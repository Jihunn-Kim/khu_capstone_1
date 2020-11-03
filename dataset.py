import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import const

'''
def int_to_binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
'''

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


# def CsvToNumpy(csv_file):
#     target_csv = pd.read_csv(csv_file)
#     inputs_save_numpy = 'inputs_' + csv_file.split('/')[-1].split('.')[0].split('_')[0] + '.npy'
#     labels_save_numpy = 'labels_' + csv_file.split('/')[-1].split('.')[0].split('_')[0] + '.npy'
#     print(inputs_save_numpy, labels_save_numpy)

#     i = 0
#     inputs_array = []
#     labels_array = []
#     print(len(target_csv))

#     while i + const.CAN_ID_BIT - 1 < len(target_csv):

#         is_regular = True
#         for j in range(const.CAN_ID_BIT):
#             l = target_csv.iloc[i + j]
#             b = l[2]
#             r = (l[b+2+1] == 'R')

#             if not r:
#                 is_regular = False
#                 break

#         inputs = np.zeros((const.CAN_ID_BIT, const.CAN_ID_BIT))
#         for idx in range(const.CAN_ID_BIT):
#             can_id = int(target_csv.iloc[i + idx, 1], 16)
#             inputs[idx] = unpack_bits(np.array(can_id), const.CAN_ID_BIT)
#         inputs = np.reshape(inputs, (1, const.CAN_ID_BIT, const.CAN_ID_BIT))
        
#         if is_regular:
#             labels = 1
#         else:
#             labels = 0

#         inputs_array.append(inputs)
#         labels_array.append(labels)

#         i+=1
#         if (i % 5000 == 0):
#             print(i)
#         # break

#     inputs_array = np.array(inputs_array)
#     labels_array = np.array(labels_array)
#     np.save(inputs_save_numpy, arr=inputs_array)
#     np.save(labels_save_numpy, arr=labels_array)
#     print('done')


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


def GetCanDataset(csv_file, length):
    csv = pd.read_csv(csv_file)
    
    i = 0
    datum = [ [], [] ]

    while i + const.CAN_ID_BIT - 1 < len(csv):
        if len(datum[0]) >= length//2 and len(datum[1]) >= length//2:
            break

        is_regular = True
        for j in range(const.CAN_ID_BIT):
            l = csv.iloc[i + j]
            b = l[2]
            r = (l[b+2+1] == 'R')

            if not r:
                is_regular = False
                break
        
        if is_regular:
            if len(datum[0]) < length//2:
                datum[0].append((i, 1))
        else:
            if len(datum[1]) < length//2:
                datum[1].append((i, 0))
        i+=1
        if (i % 5000 == 0):
            print(i, len(datum[0]), len(datum[1]))
            
    l = int((length // 2) * 0.9)
    return CanDataset(csv, datum[0][:l] + datum[1][:l]), \
            CanDataset(csv, datum[0][l:] + datum[1][l:])


class CanDataset(Dataset):

    def __init__(self, csv, datum):
        self.csv = csv
        self.datum = datum

    def __len__(self):
        return len(self.datum)

    def __getitem__(self, idx):
        start_i = self.datum[idx][0]
        is_regular = self.datum[idx][1]

        l = np.zeros((const.CAN_ID_BIT, const.CAN_ID_BIT))
        for i in range(const.CAN_ID_BIT):
            id = int(self.csv.iloc[start_i + i, 1], 16)
            bits = unpack_bits(np.array(id), const.CAN_ID_BIT)
            l[i] = bits
        l = np.reshape(l, (1, const.CAN_ID_BIT, const.CAN_ID_BIT))

        return (l, is_regular)


if __name__ == "__main__":
    kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt'}
    test_data_set = dataset.GetCanDatasetUsingTxtKwarg(-1, -1, False, **kwargs)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    for x, y in testloader:
      print(x)
      print(y)
      break
