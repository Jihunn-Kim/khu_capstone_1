import utils
import copy
from collections import OrderedDict

import model
import dataset

import importlib
importlib.reload(utils)
importlib.reload(model)
importlib.reload(dataset)

from utils import *


def add_args(parser):
    # parser.add_argument('--model', type=str, default='moderate-cnn',
    #                     help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--fold_num', type=int, default=0, 
                        help='5-fold, 0 ~ 4')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
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
                          edges,
                          device):
    print("start fed avg")
    criterion = nn.CrossEntropyLoss()
    C = 0.1
    num_edge = int(max(C * args.n_nets, 1))
    total_data_count = 0
    for _, data_count in net_data_count.items():
        total_data_count += data_count
    print("total data: %d" % total_data_count)

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
                edge_pred = edges[edge_index](inputs)

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break
            edges[edge_index].to('cpu')

        # cal weight using fed avg
        update_state = OrderedDict()
        for k, edge in enumerate(edges):
            local_state = edge.state_dict()
            for key in fed_model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * net_data_count[k] / total_data_count
                else:
                    update_state[key] += local_state[key] * net_data_count[k] / total_data_count

        fed_model.load_state_dict(update_state)
        if cr % 10 == 0:
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
                      # break
            print((step_acc / cnt).data)
            print(total_loss / cnt)
            fed_model.to('cpu')


def start_fedprox(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          testloader,
                          device):
    print("start fed prox")
    criterion = nn.CrossEntropyLoss()
    mu = 0.001
    C = 0.1
    num_edge = int(max(C * args.n_nets, 1))
    fed_model.to(device)

    for cr in range(1, args.comm_round + 1):
        print("Communication round : %d" % (cr))
        edge_weight_dict = {}
        fed_weight_dict = {}
        for fed_name, fed_param in fed_model.named_parameters():
          edge_weight_dict[fed_name] = []
          fed_weight_dict[fed_name] = fed_param

        np.random.seed(cr)  # make sure for each comparison, select the same clients each round
        selected_edge = np.random.choice(args.n_nets, num_edge, replace=False)
        print("selected edge", selected_edge)

        total_data_length = 0
        edge_data_len = []
        for edge_progress, edge_index in enumerate(selected_edge):
            train_data_set.set_idx_map(data_idx_map[edge_index])
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))
            total_data_length += len(train_data_set)
            edge_data_len.append(len(train_data_set))

            edge_model = copy.deepcopy(fed_model)
            edge_model.to(device)
            edge_opt = optim.Adam(params=edge_model.parameters(),lr=args.lr)
            # train
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                edge_opt.zero_grad()
                edge_pred = edge_model(inputs)

                edge_loss = criterion(edge_pred, labels)
                # prox term
                fed_prox_reg = 0.0
                for edge_name, edge_param in edge_model.named_parameters():
                    fed_prox_reg += ((mu / 2) * torch.norm((fed_weight_dict[edge_name] - edge_param))**2)
                edge_loss += fed_prox_reg
                
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break

            edge_model.to('cpu')
            # save edge weight
            for edge_name, edge_param in edge_model.named_parameters():
                edge_weight_dict[edge_name].append(edge_param)

        fed_model.to('cpu')
        # cal weight, / number of edge
        for fed_name, fed_param in fed_model.named_parameters():
            fed_param.data.copy_( sum(weight / num_edge for weight in edge_weight_dict[fed_name]) )
        fed_model.to(device)

        if cr % 10 == 0:
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
                      # break
            print((step_acc / cnt).data)
            print(total_loss / cnt)


def start_fedtwa(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          net_data_count,
                          testloader,
                          edges,
                          device):
    # TEFL, without asynchronous model update
    print("start fed temporally weighted aggregation")
    criterion = nn.CrossEntropyLoss()
    time_stamp = [0 for worker in range(args.n_nets)]
    twa_exp = math.e / 2.0
    C = 0.1
    num_edge = int(max(C * args.n_nets, 1))
    total_data_count = 0
    for _, data_count in net_data_count.items():
        total_data_count += data_count
    print("total data: %d" % total_data_count)
    
    for cr in range(1, args.comm_round + 1):
        print("Communication round : %d" % (cr))
        
        np.random.seed(cr)  # make sure for each comparison, select the same clients each round
        selected_edge = np.random.choice(args.n_nets, num_edge, replace=False)
        print("selected edge", selected_edge)

        for edge_progress, edge_index in enumerate(selected_edge):
            time_stamp[edge_index] = cr
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
                edge_pred = edges[edge_index](inputs)

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break
            edges[edge_index].to('cpu')

        # cal weight using time stamp
        update_state = OrderedDict()
        for k, edge in enumerate(edges):
            local_state = edge.state_dict()
            for key in fed_model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * (net_data_count[k] / total_data_count) * math.pow(twa_exp, -(cr - time_stamp[k]))
                else:
                    update_state[key] += local_state[key] * (net_data_count[k] / total_data_count) * math.pow(twa_exp, -(cr - time_stamp[k]))

        fed_model.load_state_dict(update_state)
        if cr % 10 == 0:
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
                      # break
            print((step_acc / cnt).data)
            print(total_loss / cnt)
            fed_model.to('cpu')


def start_feddw(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          net_data_count,
                          testloader,
                          local_test_loader,
                          edges,
                          device):
    print("start fed Node-aware Dynamic Weighting")
    worker_selected_frequency = [0 for worker in range(args.n_nets)]
    criterion = nn.CrossEntropyLoss()
    H = 0.5
    P = 0.5
    G = 0.1
    R = 0.1
    alpha, beta, gamma = 30.0/100.0, 50.0/100.0, 20.0/100.0
    num_edge = int(max(G * args.n_nets, 1))
    
    # cal data weight for selecting participants
    total_data_count = 0
    for _, data_count in net_data_count.items():
        total_data_count += data_count
    print("total data: %d" % total_data_count)

    total_data_weight = 0.0
    net_weight_dict = {}
    for net_key, data_count in net_data_count.items():
        net_data_count[net_key] = data_count / total_data_count
        net_weight_dict[net_key] = total_data_count / data_count
        total_data_weight += net_weight_dict[net_key]
    
    for net_key, data_count in net_weight_dict.items():
        net_weight_dict[net_key] = net_weight_dict[net_key] / total_data_weight
    # end

    worker_local_accuracy = [0 for worker in range(args.n_nets)]
    
    for cr in range(1, args.comm_round + 1):
        print("Communication round : %d" % (cr))

        # select participants
        candidates = []
        sum_frequency = sum(worker_selected_frequency)
        if sum_frequency == 0:
            sum_frequency = 1
        for worker_index in range(args.n_nets):
          candidates.append((H * worker_selected_frequency[worker_index] / sum_frequency + (1 - H) * net_weight_dict[worker_index], worker_index))
        candidates = sorted(candidates)[:int(R * args.n_nets)]
        candidates = [temp[1] for temp in candidates]

        np.random.seed(cr)
        selected_edge = np.random.choice(candidates, num_edge, replace=False)
        # end select

        # weighted frequency
        avg_selected_frequency = sum(worker_selected_frequency) / len(worker_selected_frequency)
        weighted_frequency = [P * (avg_selected_frequency - worker_frequency) for worker_frequency in worker_selected_frequency]
        frequency_prime = min(weighted_frequency)
        weighted_frequency = [frequency + frequency_prime + 1 for frequency in weighted_frequency]
        # end weigthed

        print("selected edge", selected_edge)
        for edge_progress, edge_index in enumerate(selected_edge):
            worker_selected_frequency[edge_index] += 1
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
                edge_pred = edges[edge_index](inputs)

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break

            # get edge accuracy using subset of testset
            edges[edge_index].eval()
            print("[%2d/%2d] edge: %d, cal accuracy" % (edge_progress, len(selected_edge), edge_index))
            cnt = 0
            step_acc = 0.0
            with torch.no_grad():
                for inputs, labels in local_test_loader:
                  inputs, labels = inputs.float().to(device), labels.long().to(device)

                  outputs = edges[edge_index](inputs)
                  _, preds = torch.max(outputs, 1)

                  loss = criterion(outputs, labels)
                  cnt += inputs.shape[0]

                  corr_sum = torch.sum(preds == labels.data)
                  step_acc += corr_sum.double()
                  # break

            worker_local_accuracy[edge_index] = (step_acc / cnt).item()
            print(worker_local_accuracy[edge_index])
            edges[edge_index].to('cpu')

        # cal weight dynamically
        sum_accuracy = sum(worker_local_accuracy)
        sum_weighted_frequency = sum(weighted_frequency)
        update_state = OrderedDict()
        for k, edge in enumerate(edges):
            local_state = edge.state_dict()
            for key in fed_model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] \
                    * (net_data_count[k] * alpha \
                    + worker_local_accuracy[k] / sum_accuracy * beta \
                    + weighted_frequency[k] / sum_weighted_frequency * gamma)
                else:
                    update_state[key] += local_state[key] \
                    * (net_data_count[k] * alpha \
                    + worker_local_accuracy[k] / sum_accuracy * beta \
                    + weighted_frequency[k] / sum_weighted_frequency * gamma)

        fed_model.load_state_dict(update_state)
        if cr % 10 == 0:
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
                    # break
          print((step_acc / cnt).data)
          print(total_loss / cnt)
          fed_model.to('cpu')


def start_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    args = add_args(argparse.ArgumentParser())

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data...")
    # kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt',
    #         "./dataset/Fuzzy_dataset.csv" : './Fuzzy_dataset.txt',
    #         "./dataset/RPM_dataset.csv" : './RPM_dataset.txt',
    #         "./dataset/gear_dataset.csv" : './gear_dataset.txt'
    # }
    kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt'}
    train_data_set, data_idx_map, net_class_count, net_data_count, test_data_set = dataset.GetCanDatasetUsingTxtKwarg(args.n_nets, args.fold_num, **kwargs)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    fed_model = model.Net()
    args.comm_type = 'feddw'
    if args.comm_type == "fedavg":
        edges, _, _ = init_models(args.n_nets, args)
        start_fedavg(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            edges,
                            device)
    elif args.comm_type == "fedprox":
        start_fedprox(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            testloader,
                            device)
    elif args.comm_type == "fedtwa":
        edges, _, _ = init_models(args.n_nets, args)
        start_fedtwa(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            edges,
                            device)
    elif args.comm_type == "feddw":
        local_test_set = copy.deepcopy(test_data_set)
        # mnist train 60,000 / test 10,000 / 1,000
        # CAN train ~ 13,000,000 / test 2,000,000 / for speed 40,000
        local_test_idx = np.random.choice(len(local_test_set), len(local_test_set) // 50, replace=False)
        local_test_set.set_idx_map(local_test_idx)
        local_test_loader = torch.utils.data.DataLoader(local_test_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

        edges, _, _ = init_models(args.n_nets, args)
        start_feddw(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            local_test_loader,
                            edges,
                            device)

if __name__ == "__main__":
    start_train()