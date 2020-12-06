import pandas as pd
import csv

original_csv = pd.read_csv('./Fuzzy_dataset.csv')

normal_csv = open('./fuzzy_normal_dataset.csv', 'w', newline='', encoding='utf-8')
normal_csv_file = csv.writer(normal_csv)

abnormal_csv = open('./fuzzy_abnormal_dataset.csv', 'w', newline='', encoding='utf-8')
abnormal_csv_file = csv.writer(abnormal_csv)

idx = 0

normal_first = False
abnormal_first = False
while idx < len(original_csv) // 30:
    original_row = original_csv.iloc[idx]
    number_of_data = original_row[2]
    is_regular = (original_row[number_of_data + 3] == 'R')
    original_row.dropna(inplace=True)
    if is_regular:
        if not normal_first and number_of_data != 8:
            idx += 1
            continue
        normal_first = True
        normal_csv_file.writerow(original_row[1:])
    else:
        if not abnormal_first and number_of_data != 8:
            idx += 1
            continue
        abnormal_first = True
        abnormal_csv_file.writerow(original_row[1:])

    idx += 1
    if idx % 500000 == 0:
        print(idx)