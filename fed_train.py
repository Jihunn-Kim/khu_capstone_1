import copy
import argparse
import time
import math
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn

import model
import utils
import dataset

# for google colab reload
import importlib
importlib.reload(model)
importlib.reload(utils)
importlib.reload(dataset)


## paramter

# shared
criterion = nn.CrossEntropyLoss()
C = 0.1
#

# prox
mu = 0.001
#

# time weight
twa_exp = 1.1
#

# dynamic weight
H = 0.5
P = 0.1
G = 0.1
R = 0.1
alpha, beta, gamma = 40.0/100.0, 40.0/100.0, 20.0/100.0
#

## end


def add_args(parser):
    parser.add_argument('--packet_num', type=int, default=1,
                        help='packet number used in training, 1 ~ 3')
    parser.add_argument('--dataset', type=str, default='can',
                        help='dataset used for training, can or syncan')
    parser.add_argument('--fold_num', type=int, default=0, 
                        help='5-fold, 0 ~ 4')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--n_nets', type=int, default=100,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--comm_type', type=str, default='fedprox', 
                            help='type of communication, [fedavg, fedprox, fedtwa, feddw, edge]')  
    parser.add_argument('--comm_round', type=int, default=50, 
                            help='how many round of communications we shoud use')
    parser.add_argument('--weight_save_path', type=str, default='./weights', 
                            help='model weight save path')
    args = parser.parse_args(args=[])
    return args


def test_model(fed_model, args, testloader, device):
    fed_model.to(device)
    fed_model.eval()

    cnt = 0
    step_acc = 0.0
    with torch.no_grad():
        packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, packet_state = fed_model(inputs, packet_state)
            packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

            _, preds = torch.max(outputs, 1)

            cnt += inputs.shape[0]
            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            if i % 200 == 0:
                print('test [%4d/%4d] acc: %.3f' % (i, len(testloader), (step_acc / cnt).item()))
                # break
    fed_accuracy = (step_acc / cnt).item()
    print('test acc', fed_accuracy)
    fed_model.to('cpu')
    fed_model.train()
    torch.save(fed_model.state_dict(), os.path.join(args.weight_save_path, '%s_%d_%.4f.pth' % (args.comm_type, cr, fed_accuracy)))


def start_fedavg(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          net_data_count,
                          testloader,
                          edges,
                          device):
    print("start fed avg")

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
            sampler = dataset.BatchIntervalSampler(len(train_data_set), args.batch_size)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, sampler=sampler,
                                                    shuffle=False, num_workers=2, drop_last=True)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))

            edges[edge_index] = copy.deepcopy(fed_model)
            edges[edge_index].to(device)
            edges[edge_index].train()
            edge_opt = optim.Adam(params=edges[edge_index].parameters(), lr=args.lr)

            # train
            packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                edge_pred, packet_state = edges[edge_index](inputs, packet_state)
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                edge_opt.zero_grad()

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
                    update_state[key] = local_state[key] * (net_data_count[k] / total_data_count)
                else:
                    update_state[key] += local_state[key] * (net_data_count[k] / total_data_count)
        
        fed_model.load_state_dict(update_state)
        if cr % 10 == 0:
            test_model(fed_model, args, testloader, device)


def start_fedprox(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          testloader,
                          device):
    print("start fed prox")

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

        for edge_progress, edge_index in enumerate(selected_edge):
            train_data_set.set_idx_map(data_idx_map[edge_index])
            sampler = dataset.BatchIntervalSampler(len(train_data_set), args.batch_size)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, sampler=sampler,
                                                    shuffle=False, num_workers=2, drop_last=True)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))

            edge_model = copy.deepcopy(fed_model)
            edge_model.to(device)
            edge_model.train()
            edge_opt = optim.Adam(params=edge_model.parameters(),lr=args.lr)
            # train
            packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                edge_pred, packet_state = edge_model(inputs, packet_state)
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                edge_opt.zero_grad()

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
            test_model(fed_model, args, testloader, device)
            fed_model.to(device)


def start_fedtwa(fed_model, args,
                          train_data_set,
                          data_idx_map,
                          net_data_count,
                          testloader,
                          edges,
                          device):
    # TEFL, without asynchronous model update
    print("start fed temporally weighted aggregation")

    time_stamp = [0 for worker in range(args.n_nets)]
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
            sampler = dataset.BatchIntervalSampler(len(train_data_set), args.batch_size)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, sampler=sampler,
                                                    shuffle=False, num_workers=2, drop_last=True)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))

            edges[edge_index] = copy.deepcopy(fed_model)
            edges[edge_index].to(device)
            edges[edge_index].train()
            edge_opt = optim.Adam(params=edges[edge_index].parameters(), lr=args.lr)

            # train
            packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                edge_pred, packet_state = edges[edge_index](inputs, packet_state)
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                edge_opt.zero_grad()

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break
            edges[edge_index].to('cpu')

        # cal weight using time stamp
        # in paper, cr - time_stamp[k] used, but error is high
        update_state = OrderedDict()
        for k, edge in enumerate(edges):
            local_state = edge.state_dict()
            for key in fed_model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * (net_data_count[k] / total_data_count) * math.pow(twa_exp, -(cr -2 - time_stamp[k]))
                else:
                    update_state[key] += local_state[key] * (net_data_count[k] / total_data_count) * math.pow(twa_exp, -(cr -2 - time_stamp[k]))

        fed_model.load_state_dict(update_state)
        if cr % 10 == 0:
            test_model(fed_model, args, testloader, device)


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
            sampler = dataset.BatchIntervalSampler(len(train_data_set), args.batch_size)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, sampler=sampler,
                                                    shuffle=False, num_workers=2, drop_last=True)
            print("[%2d/%2d] edge: %d, data len: %d" % (edge_progress, len(selected_edge), edge_index, len(train_data_set)))

            edges[edge_index] = copy.deepcopy(fed_model)
            edges[edge_index].to(device)
            edges[edge_index].train()
            edge_opt = optim.Adam(params=edges[edge_index].parameters(), lr=args.lr)

            # train
            packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                edge_pred, packet_state = edges[edge_index](inputs, packet_state)
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                edge_opt.zero_grad()

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break

            # get edge accuracy using subset of testset
            edges[edge_index].eval()
            print("[%2d/%2d] edge: %d, cal local accuracy" % (edge_progress, len(selected_edge), edge_index))
            cnt = 0
            step_acc = 0.0
            with torch.no_grad():
                packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
                for inputs, labels in local_test_loader:
                  inputs, labels = inputs.float().to(device), labels.long().to(device)

                  edge_pred, packet_state = edges[edge_index](inputs, packet_state)
                  packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                  _, preds = torch.max(edge_pred, 1)

                  loss = criterion(edge_pred, labels)
                  cnt += inputs.shape[0]

                  corr_sum = torch.sum(preds == labels.data)
                  step_acc += corr_sum.double()
                  # break

            worker_local_accuracy[edge_index] = (step_acc / cnt).item()
            print('edge local accuracy', worker_local_accuracy[edge_index])
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
          test_model(fed_model, args, testloader, device)


def start_only_edge(args,
                          train_data_set,
                          data_idx_map,
                          testloader,
                          edges,
                          device):
    print("start only edge")
    total_epoch = int(args.comm_round * C)

    for cr in range(1, total_epoch + 1):
        print("Edge round : %d" % (cr))
        edge_accuracy_list = []

        for edge_index, edge_model in enumerate(edges):
            train_data_set.set_idx_map(data_idx_map[edge_index])
            sampler = dataset.BatchIntervalSampler(len(train_data_set), args.batch_size)
            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, sampler=sampler,
                                                    shuffle=False, num_workers=2, drop_last=True)
            print("edge[%2d/%2d] data len: %d" % (edge_index, len(edges), len(train_data_set)))

            edge_model.to(device)
            edge_model.train()
            edge_opt = optim.Adam(params=edge_model.parameters(),lr=args.lr)

            # train
            packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
            for data_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                edge_pred, packet_state = edge_model(inputs, packet_state)
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                edge_opt.zero_grad()

                edge_loss = criterion(edge_pred, labels)
                edge_loss.backward()

                edge_opt.step()
                edge_loss = edge_loss.item()
                if data_idx % 100 == 0:
                    print('[%4d] loss: %.3f' % (data_idx, edge_loss))
                    # break

            # test
            # if cr < 4:
            #     continue
            edge_model.eval()
            total_loss = 0.0
            cnt = 0
            step_acc = 0.0
            with torch.no_grad():
                packet_state = torch.zeros(args.batch_size, model.STATE_DIM).to(device)
                for i, (inputs, labels) in enumerate(testloader):
                    inputs, labels = inputs.float().to(device), labels.long().to(device)
                    
                    outputs, packet_state = edge_model(inputs, packet_state)
                    packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

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
            edge_accuracy = (step_acc / cnt).item()
            edge_accuracy_list.append(edge_accuracy)
            print("edge[%2d/%2d] acc: %.4f" % (edge_index, len(edges), edge_accuracy))
            edge_model.to('cpu')

        # if cr < 4:
        #     continue
        edge_accuracy_avg = sum(edge_accuracy_list) / len(edge_accuracy_list)
        torch.save(edges[0].state_dict(), os.path.join(weight_path, 'edge_%d_%.4f.pth' % (cr, edge_accuracy_avg)))



def start_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    args = add_args(argparse.ArgumentParser())

    # make weight folder
    os.makedirs(args.weight_save_path, exist_ok=True)

    # for reproductivity
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data...")
    if args.dataset == 'can':
        train_data_set, data_idx_map, net_data_count, test_data_set = dataset.GetCanDataset(args.n_nets, args.fold_num, args.packet_num, "./dataset/Mixed_dataset.csv", "./dataset/Mixed_dataset_1.txt")
    elif args.dataset == 'syncan':
        train_data_set, data_idx_map, net_data_count, test_data_set = dataset.GetCanDataset(args.n_nets, args.fold_num, args.packet_num, "./dataset/test_mixed.csv", "./dataset/Mixed_dataset_1.txt")
    
    sampler = dataset.BatchIntervalSampler(len(test_data_set), args.batch_size)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size, sampler=sampler,
                                            shuffle=False, num_workers=2, drop_last=True)

    if args.dataset == 'can':
        fed_model = model.OneNet(args.packet_num)
        edges = [model.OneNet(args.packet_num) for _ in range(args.n_nets)]
    elif args.dataset == 'syncan':
        fed_model = model.OneNet(args.packet_num)
        edges = [model.OneNet(args.packet_num) for _ in range(args.n_nets)]

    if args.comm_type == "fedavg":
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
        start_fedtwa(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            edges,
                            device)
    elif args.comm_type == "feddw":
        local_test_set = copy.deepcopy(test_data_set)
        # in paper, mnist train 60,000 / test 10,000 / 1,000 - 10%
        # CAN train ~ 1,400,000 / test 300,000 / for speed 15,000 - 5%
        local_test_idx = [idx for idx in range(0, len(local_test_set) // 20)]
        local_test_set.set_idx_map(local_test_idx)
        sampler = dataset.BatchIntervalSampler(len(local_test_set), args.batch_size)
        local_test_loader = torch.utils.data.DataLoader(local_test_set, batch_size=args.batch_size, sampler=sampler,
                                                shuffle=False, num_workers=2, drop_last=True)

        start_feddw(fed_model, args,
                            train_data_set,
                            data_idx_map,
                            net_data_count,
                            testloader,
                            local_test_loader,
                            edges,
                            device)
    elif args.comm_type == "edge":
        start_only_edge(args,
                            train_data_set,
                            data_idx_map,
                            testloader,
                            edges,
                            device)

if __name__ == "__main__":
    start_train()