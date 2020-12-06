import numpy as np
import pandas as pd
import const
import math

csv = pd.read_csv("./fuzzy_normal_dataset.csv")
arr = []
idx = 0
while idx < 10000 and idx < len(csv):
    now_row = csv.iloc[idx].tolist()
    data_len = now_row[1]
    for row_idx in range(data_len + 3, len(now_row)):
        now_row[row_idx] = '00'

    del now_row[1]

    arr.append(now_row)
    idx += 1
    if idx % 1000 == 0:
        print(idx)

arr = np.array(arr)
np.save('./fuzzy_normal_numpy.npy', arr)


save_load = np.load('./fuzzy_normal_numpy.npy')
print(len(save_load))
print(save_load[0])


csv = pd.read_csv("./fuzzy_abnormal_dataset.csv")
arr = []
idx = 0
while idx < 10000 and idx < len(csv):
    now_row = csv.iloc[idx].tolist()
    data_len = now_row[1]
    for row_idx in range(data_len + 3, len(now_row)):
        now_row[row_idx] = '00'

    del now_row[1]

    arr.append(now_row)
    idx += 1
    if idx % 1000 == 0:
        print(idx)

arr = np.array(arr)
np.save('./fuzzy_abnormal_numpy.npy', arr)


# for packet number = 1
csv = pd.read_csv("./fuzzy_normal_dataset.csv")
arr = []
idx = 0
while idx < 10000 and idx < len(csv):
    packet = np.zeros((1, const.CAN_DATA_LEN * 1))
    for next_i in range(1):
        data_len = int(csv.iloc[idx + next_i, 1])
        for j in range(data_len):
            data_value = int(csv.iloc[idx + next_i, 2 + j], 16) / 255.0
            packet[0][j + const.CAN_DATA_LEN * next_i] = data_value
    arr.append(packet)

    idx += 1
    if idx % 1000 == 0:
        print(idx)

arr = np.array(arr)
np.save('./fuzzy_tensor_normal_numpy.npy', arr)

save_load = np.load('./fuzzy_tensor_normal_numpy.npy')
print(len(save_load))
print(save_load[0])


# for packet number = 1
csv = pd.read_csv("./fuzzy_abnormal_dataset.csv")
arr = []
idx = 0
while idx < 10000 and idx < len(csv):
    packet = np.zeros((1, const.CAN_DATA_LEN * 1))
    for next_i in range(1):
        data_len = int(csv.iloc[idx + next_i, 1])
        for j in range(data_len):
            data_value = int(csv.iloc[idx + next_i, 2 + j], 16) / 255.0
            packet[0][j + const.CAN_DATA_LEN * next_i] = data_value
    arr.append(packet)

    idx += 1
    if idx % 1000 == 0:
        print(idx)

arr = np.array(arr)
np.save('./fuzzy_tensor_abnormal_numpy.npy', arr)