import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import const


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


def GetCanDataset(total_edge, fold_num, packet_num, csv_path, txt_path):
    csv = pd.read_csv(csv_path)
    txt = open(txt_path, "r")
    lines = txt.read().splitlines()

    idx = 0
    datum = []
    label_temp = []
    # [cur_idx ~ cur_idx + packet_num)
    while idx + packet_num - 1 < len(csv) // 2:
        line = lines[idx + packet_num - 1]
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

    return CanDataset(csv, train_datum, packet_num), data_idx_map, net_data_count, CanDataset(csv, test_datum, packet_num, False)


class CanDataset(Dataset):

    def __init__(self, csv, datum, packet_num, is_train=True):
        self.csv = csv
        self.datum = datum
        self.packet_num = packet_num
        if is_train:
          self.idx_map = []
        else:
          self.idx_map = [idx for idx in range(len(self.datum))]

    def __len__(self):
        return len(self.idx_map) - self.packet_num + 1

    def set_idx_map(self, data_idx_map):
        self.idx_map = data_idx_map

    def __getitem__(self, idx):
        # [cur_idx ~ cur_idx + packet_num)
        start_i = self.datum[self.idx_map[idx]][0]
        is_regular = self.datum[self.idx_map[idx]][1]

        packet = np.zeros((const.CAN_DATA_LEN * self.packet_num))
        for next_i in range(self.packet_num):
            packet = np.zeros((const.CAN_DATA_LEN * self.packet_num))
            data_len = self.csv.iloc[start_i + next_i, 1]
            for j in range(data_len):
                data_value = int(self.csv.iloc[start_i + next_i, 2 + j], 16) / 255.0
                packet[j + const.CAN_DATA_LEN * next_i] = data_value

        return torch.from_numpy(packet).float(), is_regular


if __name__ == "__main__":
    pass
