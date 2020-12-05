import utils
import copy
from collections import OrderedDict

import model
import dataset

import importlib
importlib.reload(utils)
importlib.reload(model)
importlib.reload(dataset)

from utils_simple import *
import torch.quantization


def add_args(parser):
    # parser.add_argument('--model', type=str, default='one',
    #                     help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--fold_num', type=int, default=0, 
                        help='5-fold, 0 ~ 4')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--n_nets', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--comm_type', type=str, default='fedtwa', 
                            help='which type of communication strategy is going to be used: layerwise/blockwise')    
    parser.add_argument('--comm_round', type=int, default=10, 
                            help='how many round of communications we shoud use')
    args = parser.parse_args(args=[])
    return args


def start_fedavg(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          net_data_count,
                          testloader,
                          device):
    print("start fed avg")
    criterion = nn.CrossEntropyLoss()
    C = 0.1
    num_edge = int(max(C * args.n_nets, 1))
    total_data_count = 0
    for _, data_count in net_data_count.items():
        total_data_count += data_count
    print("total data: %d" % total_data_count)

    # quantize
    # fed_model.eval()
    # torch.jit.save(torch.jit.script(fed_model), './float.pth')
    
    fed_model.fuse_model()
    # modules_to_fuse = [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3']]
    # torch.quantization.fuse_modules(fed_model, modules_to_fuse, inplace=True)

    fed_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(fed_model, inplace=True)
    
    # for making shape of weight_fake_quant.scale
    train_data_set.set_idx_map([0])
    fed_model(torch.from_numpy(np.expand_dims(train_data_set[0][0], axis=0)).float())

    edges, _, _ = init_models(args.n_nets, args)
    # edges = [copy.deepcopy(fed_model) for net_cnt in range(args.n_nets)]
    for edge_now in edges:
        edge_now.fuse_model()
        edge_now.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(edge_now, inplace=True)
        edge_now(torch.from_numpy(np.expand_dims(train_data_set[0][0], axis=0)).float())

    # print('quantized \n', edges[edge_index].conv1)
    # end

    for cr in range(1, args.comm_round + 1):
        print("Communication round : %d" % (cr))

        np.random.seed(cr)  # make sure for each comparison, select the same clients each round
        selected_edge = np.random.choice(args.n_nets, num_edge, replace=False)
        print("selected edge", selected_edge)

        for edge_progress, edge_index in enumerate(selected_edge):
            train_data_set.set_idx_map(data_idx_map[edge_index])
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))

            edges[edge_index] = copy.deepcopy(fed_model)
            edges[edge_index].to(device)
            edges[edge_index].train()
            edge_opt = optim.Adam(params=edges[edge_index].parameters(), lr=args.lr)
            # train
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                edge_opt.zero_grad()
                # edge_opt[edge_index].zero_grad()
                edge_pred = edges[edge_index](inputs)

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                # edge_opt[edge_index].step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    break
            edges[edge_index].to('cpu')
            # print(edge_index)
            # local_state = edges[edge_index].state_dict()
            # for key in edges[edge_index].state_dict().keys():
            #     if 'activation_post_process' in key or 'fake_quant' in key:
            #         print(key, local_state[key])
            # print()
        # return
        # cal weight using fed avg
        update_state = OrderedDict()
        for k, edge in enumerate(edges):
            local_state = edge.state_dict()
            for key in fed_model.state_dict().keys():
                # if 'zero_point' in key:
                #     print(local_state[key])
                if 'activation_post_process' in key or 'fake_quant' in key:
                    if k == 0:
                        update_state[key] = local_state[key]
                    else:
                        update_state[key] += local_state[key]
                elif 'enable' in key:
                    update_state[key] = local_state[key]
                else:
                    if k == 0:
                        update_state[key] = local_state[key] * (net_data_count[k] / total_data_count)
                    else:
                        update_state[key] += local_state[key] * (net_data_count[k] / total_data_count)
            # break
        for key in update_state.keys():
            if 'enable' in key:
                continue
            if 'activation_post_process' in key or 'fake_quant' in key:
                # print(key, update_state[key], update_state[key].type())
                # print(key, update_state[key])
                if torch.is_floating_point(update_state[key]):
                    update_state[key] = update_state[key] / args.n_nets
                else:
                    update_state[key] = torch.floor_divide(update_state[key], args.n_nets)
                # print(update_state[key])

        fed_model.load_state_dict(update_state)
        if cr % 1 == 0:
            fed_model.to(device)
            fed_model.eval()

            total_loss = 0.0
            cnt = 0
            step_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    inputs, labels = data
                    inputs, labels = inputs.float().to(device), labels.long().to(device)

                    outputs = fed_model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    cnt += inputs.shape[0]

                    corr_sum = torch.sum(preds == labels.data)
                    step_acc += corr_sum.double()
                    running_loss = loss.item() * inputs.shape[0]
                    total_loss += running_loss
                    if i % 200 == 0:
                      print('test [%4d] loss: %.3f' % (i, loss.item()))
                      break
            print((step_acc / cnt).item())
            print(total_loss / cnt)
            fed_model.to('cpu')
            quantized_fed_model = torch.quantization.convert(fed_model.eval(), inplace=False)
            torch.jit.save(torch.jit.script(quantized_fed_model), './quan.pth')



def start_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    args = add_args(argparse.ArgumentParser())

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data...")
    kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt'}
    train_data_set, data_idx_map, _, net_data_count, test_data_set = dataset.GetCanDatasetUsingTxtKwarg(args.n_nets, args.fold_num, **kwargs)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    # run_benchmark('./quan.pth', testloader)
    # run_benchmark('./float.pth', testloader)

    fed_model = model.Net()
    args.comm_type = 'fedavg'
    if args.comm_type == "fedavg":
        start_fedavg(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            device)

if __name__ == "__main__":
    start_train()
