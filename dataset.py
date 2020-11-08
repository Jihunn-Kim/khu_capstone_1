import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import const
from torch.utils.data.sampler import Sampler

# 0, batch * 1, batch * 2 ...
class BatchIntervalSampler(Sampler):

    def __init__(self, data_length, batch_size):
        # data length 가 batch size 로 나뉘게 만듦
        if data_length % batch_size != 0:
            data_length = data_length - (data_length % batch_size)

        self.indices =[]
        # print(data_length)
        batch_group_interval = int(data_length / batch_size)
        for group_idx in range(batch_group_interval):
            for local_idx in range(batch_size):
                self.indices.append(group_idx + local_idx * batch_group_interval)
        # print('sampler init', self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def unpack_bits(x, num_bits):
    """
    Args:
        x (int): bit로 변환할 정수
        num_bits (int): 표현할 비트수 
    """
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def CsvToText(csv_file):
    target_csv = pd.read_csv(csv_file)
    text_file_name = csv_file.split('/')[-1].split('.')[0] + '.txt'
    print(text_file_name)
    target_text = open(text_file_name, mode='wt', encoding='utf-8')

    i = 0
    datum = [ [], [] ]
    print(len(target_csv))

    while i + const.CAN_ID_BIT - 1 < len(target_csv):

        is_regular = True
        for j in range(const.CAN_ID_BIT):
            l = target_csv.iloc[i + j]
            b = l[2]
            r = (l[b+2+1] == 'R')

            if not r:
                is_regular = False
                break
        
        if is_regular:
            target_text.write("%d R\n" % i)
        else:
            target_text.write("%d T\n" % i)

        i+=1
        if (i % 5000 == 0):
            print(i)

    target_text.close()
    print('done')


def GetCanDataset(total_edge, fold_num, csv_path, txt_path):
    csv = pd.read_csv(csv_path)
    txt = open(txt_path, "r")
    lines = txt.read().splitlines()

    idx = 0
    datum = []
    label_temp = []
    while idx < len(csv) // 2:
        line = lines[idx]
        if not line:
            break

        if line.split(' ')[1] == 'R':
            datum.append((idx, 1))
            label_temp.append(1)
        else:
            datum.append((idx, 0))
            label_temp.append(0)

        idx += 1
        if (idx % 1000000 == 0):
            print(idx)

    fold_length = int(len(label_temp) / 5)
    train_datum = []
    train_label_temp = []
    for i in range(5):
        if i != fold_num:
            train_datum += datum[i*fold_length:(i+1)*fold_length]
            train_label_temp += label_temp[i*fold_length:(i+1)*fold_length]
        else:
            test_datum = datum[i*fold_length:(i+1)*fold_length]


    N = len(train_label_temp)
    train_label_temp = np.array(train_label_temp)

    proportions = np.random.dirichlet(np.repeat(1, total_edge))
    proportions = np.cumsum(proportions)
    idx_batch = [[] for _ in range(total_edge)]
    data_idx_map = {}
    prev = 0.0
    for j in range(total_edge):
        idx_batch[j] = [idx for idx in range(int(prev * N), int(proportions[j] * N))]
        prev = proportions[j]
        data_idx_map[j] = idx_batch[j]

    _, net_data_count = record_net_data_stats(train_label_temp, data_idx_map)

    return CanDataset(csv, train_datum), data_idx_map, net_data_count, CanDataset(csv, test_datum, False)


class CanDataset(Dataset):

    def __init__(self, csv, datum, is_train=True):
        self.csv = csv
        self.datum = datum
        if is_train:
          self.idx_map = []
        else:
          self.idx_map = [idx for idx in range(len(self.datum))]

    def __len__(self):
        return len(self.idx_map)

    def set_idx_map(self, data_idx_map):
        self.idx_map = data_idx_map

    def __getitem__(self, idx):
        # print(idx)
        start_i = self.datum[self.idx_map[idx]][0]
        is_regular = self.datum[self.idx_map[idx]][1]

        packet = np.zeros((1, const.CAN_DATA_LEN))
        data_len = self.csv.iloc[start_i, 1]
        for j in range(data_len):
            data_value = int(self.csv.iloc[start_i, 2 + j], 16) / 255.0
            packet[0, j] = data_value

        return (packet, is_regular)


# def GetCanDataset(total_edge, fold_num, csv_path, txt_path):
#     csv = pd.read_csv(csv_path)
#     txt = open(txt_path, "r")
#     lines = txt.read().splitlines()
#     frame_size = const.CAN_FRAME_LEN
#     idx = 0
#     datum = []
#     label_temp = []
#     while idx + frame_size  - 1 < len(csv) // 2:
#         # csv_row = csv.iloc[idx + frame_size - 1]
#         # data_len = csv_row[1]
#         # is_regular = (csv_row[data_len + 2] == 'R')

#         # if is_regular:
#         #     datum.append((idx, 1))
#         #     label_temp.append(1)
#         # else:
#         #     datum.append((idx, 0))
#         #     label_temp.append(0)
#         line = lines[idx]
#         if not line:
#             break

#         if line.split(' ')[1] == 'R':
#             datum.append((idx, 1))
#             label_temp.append(1)
#         else:
#             datum.append((idx, 0))
#             label_temp.append(0)

#         idx += 1
#         if (idx % 1000000 == 0):
#             print(idx)

#     fold_length = int(len(label_temp) / 5)
#     train_datum = []
#     train_label_temp = []
#     for i in range(5):
#         if i != fold_num:
#             train_datum += datum[i*fold_length:(i+1)*fold_length]
#             train_label_temp += label_temp[i*fold_length:(i+1)*fold_length]
#         else:
#             test_datum = datum[i*fold_length:(i+1)*fold_length]

#     min_size = 0
#     output_class_num = 2
#     N = len(train_label_temp)
#     train_label_temp = np.array(train_label_temp)
#     data_idx_map = {}

#     # proportions = np.random.dirichlet(np.repeat(1, total_edge))
#     # proportions = np.cumsum(proportions)
#     # idx_batch = [[] for _ in range(total_edge)]
#     # prev = 0.0
#     # for j in range(total_edge):
#     #     idx_batch[j] = [idx for idx in range(int(prev * N), int(proportions[j] * N))]
#     #     prev = proportions[j]
#     #     np.random.shuffle(idx_batch[j])
#     #     data_idx_map[j] = idx_batch[j]
        
#     while min_size < 512:
#         idx_batch = [[] for _ in range(total_edge)]
#         # for each class in the dataset
#         for k in range(output_class_num):
#             idx_k = np.where(train_label_temp == k)[0]
#             np.random.shuffle(idx_k)
#             proportions = np.random.dirichlet(np.repeat(1, total_edge))
#             ## Balance
#             proportions = np.array([p*(len(idx_j)<N/total_edge) for p,idx_j in zip(proportions,idx_batch)])
#             proportions = proportions/proportions.sum()
#             proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
#             idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
#             min_size = min([len(idx_j) for idx_j in idx_batch])

#     for j in range(total_edge):
#         np.random.shuffle(idx_batch[j])
#         data_idx_map[j] = idx_batch[j]

#     _, net_data_count = record_net_data_stats(train_label_temp, data_idx_map)

#     return CanDataset(csv, train_datum), data_idx_map, net_data_count, CanDataset(csv, test_datum, False)


# class CanDataset(Dataset):

#     def __init__(self, csv, datum, is_train=True):
#         self.csv = csv
#         self.datum = datum
#         self.is_train = is_train
#         if self.is_train:
#           self.idx_map = []
#         else:
#           self.idx_map = [idx for idx in range(len(self.datum))]

#     def __len__(self):
#         return len(self.idx_map)

#     def set_idx_map(self, data_idx_map):
#         self.idx_map = data_idx_map

#     def __getitem__(self, idx):
#         start_i = self.datum[self.idx_map[idx]][0]
#         if self.is_train:
#             is_regular = self.datum[self.idx_map[idx]][1]
#             l = np.zeros((const.CAN_FRAME_LEN, const.CAN_DATA_LEN))
#             '''
#                 각 바이트 값은 모두 normalized 된다.
#                 0 ~ 255 -> 0.0 ~ 1.0
#             '''
#             for i in range(const.CAN_FRAME_LEN):
#                 data_len = self.csv.iloc[start_i + i, 1]
#                 for j in range(data_len):
#                     k = int(self.csv.iloc[start_i + i, 2 + j], 16) / 255.0
#                     l[i][j] = k
#             l = np.reshape(l, (1, const.CAN_FRAME_LEN, const.CAN_DATA_LEN))
#         else:
#             l = np.zeros((const.CAN_DATA_LEN))
#             data_len = self.csv.iloc[start_i, 1]
#             is_regular = self.csv.iloc[start_i, data_len + 2] == 'R'
#             if is_regular:
#                 is_regular = 1
#             else:
#                 is_regular = 0
#             for j in range(data_len):
#                 k = int(self.csv.iloc[start_i, 2 + j], 16) / 255.0
#                 l[j] = k
#             l = np.reshape(l, (1, const.CAN_DATA_LEN))

#         return (l, is_regular)


def record_net_data_stats(label_temp, data_idx_map):
    net_class_count = {}
    net_data_count= {}

    for net_i, dataidx in data_idx_map.items():
        unq, unq_cnt = np.unique(label_temp[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_class_count[net_i] = tmp
        net_data_count[net_i] = len(dataidx)
    print('Data statistics: %s' % str(net_class_count))
    return net_class_count, net_data_count


def GetCanDatasetUsingTxtKwarg(total_edge, fold_num, **kwargs):
    csv_list = []
    total_datum = []
    total_label_temp = []
    csv_idx = 0
    for csv_file, txt_file in kwargs.items():
        csv = pd.read_csv(csv_file)
        csv_list.append(csv)

        txt = open(txt_file, "r")
        lines = txt.read().splitlines()

        idx = 0
        local_datum = []
        while idx + const.CAN_ID_BIT - 1 < len(csv):
            line = lines[idx]
            if not line:
                break

            if line.split(' ')[1] == 'R':
                local_datum.append((csv_idx, idx, 1))
                total_label_temp.append(1)
            else:
                local_datum.append((csv_idx, idx, 0))
                total_label_temp.append(0)

            idx += 1
            if (idx % 1000000 == 0):
                print(idx)

        csv_idx += 1
        total_datum += local_datum

    fold_length = int(len(total_label_temp) / 5)
    datum = []
    label_temp = []
    for i in range(5):
        if i != fold_num:
            datum += total_datum[i*fold_length:(i+1)*fold_length]
            label_temp += total_label_temp[i*fold_length:(i+1)*fold_length]
        else:
            test_datum = total_datum[i*fold_length:(i+1)*fold_length]

    min_size = 0
    output_class_num = 2
    N = len(label_temp)
    label_temp = np.array(label_temp)
    data_idx_map = {}

    while min_size < 512:
        idx_batch = [[] for _ in range(total_edge)]
        # for each class in the dataset
        for k in range(output_class_num):
            idx_k = np.where(label_temp == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(1, total_edge))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/total_edge) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(total_edge):
        np.random.shuffle(idx_batch[j])
        data_idx_map[j] = idx_batch[j]

    net_class_count, net_data_count = record_net_data_stats(label_temp, data_idx_map)

    return CanDatasetKwarg(csv_list, datum), data_idx_map, net_class_count, net_data_count, CanDatasetKwarg(csv_list, test_datum, False)


class CanDatasetKwarg(Dataset):

    def __init__(self, csv_list, datum, is_train=True):
        self.csv_list = csv_list
        self.datum = datum
        if is_train:
          self.idx_map = []
        else:
          self.idx_map = [idx for idx in range(len(self.datum))]

    def __len__(self):
        return len(self.idx_map)

    def set_idx_map(self, data_idx_map):
        self.idx_map = data_idx_map

    def __getitem__(self, idx):
        csv_idx = self.datum[self.idx_map[idx]][0]
        start_i = self.datum[self.idx_map[idx]][1]
        is_regular = self.datum[self.idx_map[idx]][2]

        l = np.zeros((const.CAN_ID_BIT, const.CAN_ID_BIT))
        for i in range(const.CAN_ID_BIT):
            id_ = int(self.csv_list[csv_idx].iloc[start_i + i, 1], 16)
            bits = unpack_bits(np.array(id_), const.CAN_ID_BIT)
            l[i] = bits
        l = np.reshape(l, (1, const.CAN_ID_BIT, const.CAN_ID_BIT))

        return (l, is_regular)


def GetCanDatasetUsingTxt(csv_file, txt_path, length):
    csv = pd.read_csv(csv_file)
    txt = open(txt_path, "r")
    lines = txt.read().splitlines()

    idx = 0
    datum = [ [], [] ]
    while idx + const.CAN_ID_BIT - 1 < len(csv):
        if len(datum[0]) >= length//2 and len(datum[1]) >= length//2:
            break

        line = lines[idx]
        if not line:
            break

        if line.split(' ')[1] == 'R':
            if len(datum[0]) < length//2:
                datum[0].append((idx, 1))
        else:
            if len(datum[1]) < length//2:
                datum[1].append((idx, 0))

        idx += 1
        if (idx % 5000 == 0):
            print(idx, len(datum[0]), len(datum[1]))
            
    l = int((length // 2) * 0.9)
    return CanDataset(csv, datum[0][:l] + datum[1][:l]), \
            CanDataset(csv, datum[0][l:] + datum[1][l:])


if __name__ == "__main__":
    kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt'}
    test_data_set = dataset.GetCanDatasetUsingTxtKwarg(-1, -1, False, **kwargs)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    for x, y in testloader:
      print(x)
      print(y)
      break
