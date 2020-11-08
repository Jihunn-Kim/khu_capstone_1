import pandas as pd
import numpy as np
import csv
import os
import const

# xxx
def CsvToText(csv_file):
    target_csv = pd.read_csv(csv_file)
    file_name, extension = os.path.splitext(csv_file)
    print(file_name, extension)
    target_text = open(file_name + '_4.txt', mode='wt', encoding='utf-8')

    idx = 0
    print(len(target_csv))

    while idx + const.CAN_FRAME_LEN - 1 < len(target_csv):
        csv_row = target_csv.iloc[idx + const.CAN_FRAME_LEN - 1]
        data_len = csv_row[1]
        is_regular = (csv_row[data_len + 2] == 'R')

        if is_regular:
            target_text.write("%d R\n" % idx)
        else:
            target_text.write("%d T\n" % idx)

        idx += 1
        if (idx % 1000000 == 0):
            print(idx)

    target_text.close()
    print('done')


def Mix_Four_CANDataset():
    Dos_csv = pd.read_csv('./dataset/DoS_dataset.csv')
    Other_csv = [pd.read_csv('./dataset/Fuzzy_dataset.csv'),
                  pd.read_csv('./dataset/RPM_dataset.csv'),
                  pd.read_csv('./dataset/gear_dataset.csv')]
    Other_csv_idx = [0, 0, 0]
    
    save_csv = open('./dataset/Mixed_dataset.csv', 'w')
    save_csv_file = csv.writer(save_csv)

    # DoS 유해 트래픽 주기를 바꿈
    # DoS 다음 세번의 Dos 자리를 다른 유해 트래픽으로 바꿈
    # DoS / (Fuzzy, RPM, gear) 중 3번 순서 랜덤, 뽑히는 개수 랜덤 / Dos ...
    dos_idx = 0
    dos_preriod = 3
    while dos_idx < len(Dos_csv):
        dos_row = Dos_csv.iloc[dos_idx]
        number_of_data = dos_row[2]
        is_regular = (dos_row[number_of_data + 3] == 'R')
        dos_row.dropna(inplace=True)

        if is_regular:
            save_csv_file.writerow(dos_row[1:])
        else:
            if dos_preriod == 3:
                save_csv_file.writerow(dos_row[1:])
                np.random.seed(dos_idx)
                selected_edge = np.random.choice([0, 1, 2], 3, replace=True)
            else:
                selected_idx = selected_edge[dos_preriod]
                local_csv = Other_csv[selected_idx]
                local_idx = Other_csv_idx[selected_idx]

                while True:
                    local_row = local_csv.iloc[local_idx]
                    local_number_of_data = local_row[2]
                    is_injected = (local_row[local_number_of_data + 3] == 'T')
                    local_idx += 1
                    if is_injected:
                        local_row.dropna(inplace=True)
                        save_csv_file.writerow(local_row[1:])
                        break
                Other_csv_idx[selected_idx] = local_idx

            dos_preriod -= 1
            if dos_preriod == -1:
                dos_preriod = 3

        dos_idx += 1
        if dos_idx % 100000 == 0:
            print(dos_idx)
            # break
    save_csv.close()

