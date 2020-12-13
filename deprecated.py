#### utils ####
# for mixed dataset
def CsvToTextCNN(csv_file):
    target_csv = pd.read_csv(csv_file)
    file_name, extension = os.path.splitext(csv_file)
    print(file_name, extension)
    target_text = open(file_name + '_CNN8.txt', mode='wt', encoding='utf-8')

    idx = 0
    print(len(target_csv))

    while idx + const.CNN_FRAME_LEN - 1 < len(target_csv):

        is_regular = True
        for j in range(const.CNN_FRAME_LEN):
            l = target_csv.iloc[idx + j]
            b = l[1]
            r = (l[b+2] == 'R')

            if not r:
                is_regular = False
                break
        
        if is_regular:
            target_text.write("%d R\n" % idx)
        else:
            target_text.write("%d T\n" % idx)

        idx += 1
        if idx % 300000 == 0:
            print(idx)

    target_text.close()
    print('done')



#### dataset ####
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


def GetCanDataset(total_edge, fold_num, csv_path, txt_path):
    csv = pd.read_csv(csv_path)
    txt = open(txt_path, "r")
    lines = txt.read().splitlines()
    frame_size = const.CAN_FRAME_LEN
    idx = 0
    datum = []
    label_temp = []
    while idx + frame_size  - 1 < len(csv) // 2:
        # csv_row = csv.iloc[idx + frame_size - 1]
        # data_len = csv_row[1]
        # is_regular = (csv_row[data_len + 2] == 'R')

        # if is_regular:
        #     datum.append((idx, 1))
        #     label_temp.append(1)
        # else:
        #     datum.append((idx, 0))
        #     label_temp.append(0)
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

    min_size = 0
    output_class_num = 2
    N = len(train_label_temp)
    train_label_temp = np.array(train_label_temp)
    data_idx_map = {}

    # proportions = np.random.dirichlet(np.repeat(1, total_edge))
    # proportions = np.cumsum(proportions)
    # idx_batch = [[] for _ in range(total_edge)]
    # prev = 0.0
    # for j in range(total_edge):
    #     idx_batch[j] = [idx for idx in range(int(prev * N), int(proportions[j] * N))]
    #     prev = proportions[j]
    #     np.random.shuffle(idx_batch[j])
    #     data_idx_map[j] = idx_batch[j]
        
    while min_size < 512:
        idx_batch = [[] for _ in range(total_edge)]
        # for each class in the dataset
        for k in range(output_class_num):
            idx_k = np.where(train_label_temp == k)[0]
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

    _, net_data_count = record_net_data_stats(train_label_temp, data_idx_map)

    return CanDataset(csv, train_datum), data_idx_map, net_data_count, CanDataset(csv, test_datum, False)


class CanDataset(Dataset):

    def __init__(self, csv, datum, is_train=True):
        self.csv = csv
        self.datum = datum
        self.is_train = is_train
        if self.is_train:
          self.idx_map = []
        else:
          self.idx_map = [idx for idx in range(len(self.datum))]

    def __len__(self):
        return len(self.idx_map)

    def set_idx_map(self, data_idx_map):
        self.idx_map = data_idx_map

    def __getitem__(self, idx):
        start_i = self.datum[self.idx_map[idx]][0]
        if self.is_train:
            is_regular = self.datum[self.idx_map[idx]][1]
            l = np.zeros((const.CAN_FRAME_LEN, const.CAN_DATA_LEN))
            '''
                각 바이트 값은 모두 normalized 된다.
                0 ~ 255 -> 0.0 ~ 1.0
            '''
            for i in range(const.CAN_FRAME_LEN):
                data_len = self.csv.iloc[start_i + i, 1]
                for j in range(data_len):
                    k = int(self.csv.iloc[start_i + i, 2 + j], 16) / 255.0
                    l[i][j] = k
            l = np.reshape(l, (1, const.CAN_FRAME_LEN, const.CAN_DATA_LEN))
        else:
            l = np.zeros((const.CAN_DATA_LEN))
            data_len = self.csv.iloc[start_i, 1]
            is_regular = self.csv.iloc[start_i, data_len + 2] == 'R'
            if is_regular:
                is_regular = 1
            else:
                is_regular = 0
            for j in range(data_len):
                k = int(self.csv.iloc[start_i, 2 + j], 16) / 255.0
                l[j] = k
            l = np.reshape(l, (1, const.CAN_DATA_LEN))

        return (l, is_regular)


def GetCanDatasetCNN(total_edge, fold_num, csv_path, txt_path):
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

    return CanDatasetCNN(csv, train_datum), data_idx_map, net_data_count, CanDatasetCNN(csv, test_datum, False)


class CanDatasetCNN(Dataset):

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
        start_i = self.datum[self.idx_map[idx]][0]
        is_regular = self.datum[self.idx_map[idx]][1]
        
        packet = np.zeros((const.CNN_FRAME_LEN, const.CNN_FRAME_LEN))
        for i in range(const.CNN_FRAME_LEN):
            data_len = self.csv.iloc[start_i + i, 1]
            for j in range(data_len):
                k = int(self.csv.iloc[start_i + i, 2 + j], 16) / 255.0
                packet[i][j] = k
        packet = np.reshape(packet, (1, const.CNN_FRAME_LEN, const.CNN_FRAME_LEN))
        return (packet, is_regular)


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


def GetSynCanDataset(total_edge, fold_num, packet_num, csv_path, txt_path):
    csv = pd.read_csv(csv_path)
    txt = open(txt_path, "r")
    lines = txt.read().splitlines()

    idx = 0
    datum = []
    label_temp = []
    # [cur_idx ~ cur_idx + packet_num)
    while idx + packet_num - 1 < len(csv):
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

    return CanSynDataset(csv, train_datum, packet_num), data_idx_map, net_data_count, CanSynDataset(csv, test_datum, packet_num, False)


class CanSynDataset(Dataset):

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

        packet = np.zeros((const.SYNCAN_DATA_LEN * self.packet_num))
        for next_i in range(self.packet_num):
            data_id = float(self.csv.iloc[start_i + next_i, 1][2:])
            packet[const.SYNCAN_DATA_LEN * next_i] = (data_id / 10.0)

            data_len = 5
            for j in range(1, data_len):
                data_value = float(self.csv.iloc[start_i + next_i, 1 + j])
                packet[j + const.SYNCAN_DATA_LEN * next_i] = data_value

        return torch.from_numpy(packet).float(), is_regular


# for syncan dataset
def CsvToText_SynCAN(csv_file):
    target_csv = pd.read_csv(csv_file)
    file_name, extension = os.path.splitext(csv_file)
    print(file_name, extension)
    target_text = open(file_name + '.txt', mode='wt', encoding='utf-8')

    idx = 0
    print(len(target_csv))

    while idx < len(target_csv):
        csv_row = target_csv.iloc[idx]
        is_regular = (int(csv_row[0]) == 0)

        if is_regular:
            target_text.write("%d R\n" % idx)
        else:
            target_text.write("%d T\n" % idx)

        idx += 1
        if (idx % 1000000 == 0):
            print(idx)

    target_text.close()
    print('done')


def Mix_Six_SynCANDataset():
    normal_csv = pd.read_csv('./dataset/test_normal.csv')
    normal_idx = 0
    target_len = len(normal_csv)

    save_csv = open('./dataset/test_mixed.csv', 'w')
    save_csv_file = csv.writer(save_csv)

    other_csv = [pd.read_csv('./dataset/test_continuous.csv'),
                  pd.read_csv('./dataset/test_flooding.csv'),
                  pd.read_csv('./dataset/test_plateau.csv'),
                  pd.read_csv('./dataset/test_playback.csv'),
                  pd.read_csv('./dataset/test_suppress.csv')]
    other_csv_idx = [0, 0, 0, 0, 0]

    while normal_idx < target_len:
        np.random.seed(normal_idx)
        selected_csv = np.random.choice([0, 1, 2, 3, 4], 5, replace=True)
        all_done = True
        for csv_idx in selected_csv:
          now_csv = other_csv[csv_idx]
          now_idx = other_csv_idx[csv_idx]

          start_normal_idx = now_idx
          while now_idx < len(now_csv):
            csv_row_ahead = now_csv.iloc[now_idx + 1]
            label_ahead = csv_row_ahead[0]

            csv_row_behind = now_csv.iloc[now_idx]
            label_behind = csv_row_behind[0]

            if label_ahead == 1 and label_behind == 0:
              print(now_idx, 'start error')
              add_normal_len = (now_idx - start_normal_idx) // 9
              start_abnormal_idx = now_idx + 1
            elif label_ahead == 0 and label_behind == 1:
              print(now_idx, 'end error')
              add_abnormal_len = (now_idx - start_abnormal_idx) // 6

              for _ in range(6):
                  # done
                  if normal_idx + add_normal_len >= target_len:
                      save_csv.close()
                      return

                  # write normal
                  for idx in range(normal_idx, normal_idx + add_normal_len):
                    row = normal_csv.iloc[idx]
                    row = row.fillna(0)
                    if len(row) != 7:
                      continue
                    save_csv_file.writerow(row[0:1].append(row[2:]))
                  normal_idx += add_normal_len
                  # write abnormal
                  for idx in range(start_abnormal_idx, start_abnormal_idx + add_abnormal_len):
                    row = now_csv.iloc[idx]
                    row = row.fillna(0)
                    if len(row) != 7:
                      continue
                    save_csv_file.writerow(row[0:1].append(row[2:]))
                  start_abnormal_idx += add_abnormal_len

              other_csv_idx[csv_idx] = now_idx + 1
              # check other csv not end
              all_done = False
              break
              
            now_idx += 1

        if all_done:
            break

    save_csv.close()
    
def test():
    x = np.arange(10, 51, 10)
    y = [0, 0, 0, 0, 0]
    y2 = [1, 1, 1, 1, 1]
    y3 = [2, 2, 2, 2, 2]
    y4 = [3, 3, 3, 3, 3]
    plt.figure(figsize=[9, 7])
    plt.plot(x,y,'b',label='fed avg')
    plt.plot(x,y2,'r',label='fed prox')
    plt.plot(x,y3,'g',label='fed dynamic')
    plt.plot(x,y4,'m',label='fed timstamp')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()

    plt.savefig('./savefig_default.png')
