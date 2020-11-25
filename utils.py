import pandas as pd
import numpy as np
import csv
import os
import const
from matplotlib import pyplot as plt


def run_benchmark_cnn():
    import sys
    sys.path.append("/content/drive/My Drive/capstone1/CAN/torch2trt")
    from torch2trt import torch2trt
    import model
    import time
    import torch
    import dataset
    import torch.nn as nn

    test_model = model.CnnNet()
    test_model.eval().cuda()

    batch_size = 1
    # _, _, _, test_data_set = dataset.GetCanDataset(100, 0, "./dataset/Mixed_dataset.csv", "./dataset/Mixed_dataset_1.txt")
    
    # sampler = dataset.BatchIntervalSampler(len(test_data_set), batch_size)
    # testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=sampler,
    #                                         shuffle=False, num_workers=2, drop_last=True)
    
    # create model and input data
    # for inputs, labels in testloader:
    #     trt_x = inputs.float().cuda()
    #     trt_state = torch.zeros((batch_size, 8 * 32)).float().cuda()
    #     trt_model = model.OneNet()
    #     trt_model.load_state_dict(torch.load(weight_path))
    #     trt_model.float().eval().cuda()

    #     trt_f16_x = inputs.half().cuda()
    #     trt_f16_state = torch.zeros((batch_size, 8 * 32)).half().cuda()
    #     trt_f16_model = model.OneNet()
    #     trt_f16_model.load_state_dict(torch.load(weight_path))
    #     trt_f16_model.half().eval().cuda()

    #     trt_int8_strict_x = inputs.float().cuda()
    #     trt_int8_strict_state = torch.zeros((batch_size, 8 * 32)).float().cuda() # match model weight
    #     trt_int8_strict_model = model.OneNet()
    #     trt_int8_strict_model.load_state_dict(torch.load(weight_path))
    #     trt_int8_strict_model.eval().cuda() # no attribute 'char'
        
    #     break

    inputs = torch.ones((batch_size, 1, const.CNN_FRAME_LEN, const.CNN_FRAME_LEN))


    trt_x = inputs.half().cuda() # ??? densenet error?
    trt_model = model.CnnNet()
    # trt_model.load_state_dict(torch.load(weight_path))
    trt_model.eval().cuda()

    trt_f16_x = inputs.half().cuda()
    trt_f16_model = model.CnnNet().half()
    # trt_f16_model.load_state_dict(torch.load(weight_path))
    trt_f16_model.half().eval().cuda()

    trt_int8_strict_x = inputs.half().cuda() # match model weight
    trt_int8_strict_model = model.CnnNet()
    # trt_int8_strict_model.load_state_dict(torch.load(weight_path))
    trt_int8_strict_model.eval().cuda() # no attribute 'char'

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(trt_model, [trt_x], max_batch_size=batch_size)
    model_trt_f16 = torch2trt(trt_f16_model, [trt_f16_x], fp16_mode=True, max_batch_size=batch_size)
    model_trt_int8_strict = torch2trt(trt_int8_strict_model, [trt_int8_strict_x], fp16_mode=False, int8_mode=True, strict_type_constraints=True, max_batch_size=batch_size)

    # testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=sampler,
    #                                         shuffle=False, num_workers=2, drop_last=True)

    with torch.no_grad():
        ### test inference time
        dummy_x = torch.ones((batch_size, 1, const.CNN_FRAME_LEN, const.CNN_FRAME_LEN)).half().cuda()
        dummy_cnt = 10000
        print('ignore data loading time, inference random data')

        check_time = time.time()
        for i in range(dummy_cnt):
            _ = test_model(dummy_x)
        print('torch model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        check_time = time.time()
        for i in range(dummy_cnt):
            _ = model_trt(dummy_x)
        print('trt model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        dummy_x = torch.ones((batch_size, 1, const.CNN_FRAME_LEN, const.CNN_FRAME_LEN)).half().cuda()
        check_time = time.time()
        for i in range(dummy_cnt):
            _ = model_trt_f16(dummy_x)
        print('trt float 16 model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        dummy_x = torch.ones((batch_size, 1, const.CNN_FRAME_LEN, const.CNN_FRAME_LEN)).char().cuda()
        check_time = time.time()
        for i in range(dummy_cnt):
            _ = model_trt_int8_strict(dummy_x)
        print('trt int8 strict model: %.6f' % ((time.time() - check_time) / dummy_cnt))
        return
        ## end

        criterion = nn.CrossEntropyLoss()
        state_temp = torch.zeros((batch_size, 8 * 32)).cuda()
        step_acc = 0.0
        step_loss = 0.0
        cnt = 0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = test_model(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('torch', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('trt', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).half().cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.half().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt_f16(inputs, state_temp)

            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('float16', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).float().cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt_int8_strict(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('int8 strict', step_acc.item() / cnt, step_loss / loss_cnt)


def run_benchmark(weight_path):
    import sys
    sys.path.append("/content/drive/My Drive/capstone1/CAN/torch2trt")
    from torch2trt import torch2trt
    import model
    import time
    import torch
    import dataset
    import torch.nn as nn

    test_model = model.OneNet()
    test_model.load_state_dict(torch.load(weight_path))
    test_model.eval().cuda()

    batch_size = 1
    _, _, _, test_data_set = dataset.GetCanDataset(100, 0, "./dataset/Mixed_dataset.csv", "./dataset/Mixed_dataset_1.txt")
    
    sampler = dataset.BatchIntervalSampler(len(test_data_set), batch_size)
    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=sampler,
                                            shuffle=False, num_workers=2, drop_last=True)
    
    # create model and input data
    for inputs, labels in testloader:
        # inputs = torch.cat([inputs, inputs, inputs], 1)

        trt_x = inputs.float().cuda()
        trt_state = torch.zeros((batch_size, 8 * 32)).float().cuda()
        trt_model = model.OneNet()
        trt_model.load_state_dict(torch.load(weight_path))
        trt_model.float().eval().cuda()

        trt_f16_x = inputs.half().cuda()
        trt_f16_state = torch.zeros((batch_size, 8 * 32)).half().cuda()
        trt_f16_model = model.OneNet().half()
        trt_f16_model.load_state_dict(torch.load(weight_path))
        trt_f16_model.half().eval().cuda()

        trt_int8_strict_x = inputs.float().cuda()
        trt_int8_strict_state = torch.zeros((batch_size, 8 * 32)).float().cuda() # match model weight
        trt_int8_strict_model = model.OneNet()
        trt_int8_strict_model.load_state_dict(torch.load(weight_path))
        trt_int8_strict_model.eval().cuda() # no attribute 'char'
        
        break

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(trt_model, [trt_x, trt_state], max_batch_size=batch_size)
    model_trt_f16 = torch2trt(trt_f16_model, [trt_f16_x, trt_f16_state], fp16_mode=True, max_batch_size=batch_size)
    model_trt_int8_strict = torch2trt(trt_int8_strict_model, [trt_int8_strict_x, trt_int8_strict_state], fp16_mode=False, int8_mode=True, strict_type_constraints=True, max_batch_size=batch_size)

    testloader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=sampler,
                                            shuffle=False, num_workers=2, drop_last=True)

    with torch.no_grad():
        ### test inference time
        dummy_x = torch.ones((batch_size, 8)).cuda()
        dummy_state = torch.zeros(batch_size, model.STATE_DIM).cuda()
        dummy_cnt = 10000
        print('ignore data loading time, inference random data')
        
        check_time = time.time()
        for i in range(dummy_cnt):
            _, _ = test_model(dummy_x, dummy_state)
        print('torch model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        check_time = time.time()
        for i in range(dummy_cnt):
            _, _ = model_trt(dummy_x, dummy_state)
        print('trt model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        dummy_x = torch.ones((batch_size, 8)).half().cuda()
        dummy_state = torch.zeros(batch_size, model.STATE_DIM).half().cuda()
        check_time = time.time()
        for i in range(dummy_cnt):
            _, _ = model_trt_f16(dummy_x, dummy_state)
        print('trt float 16 model: %.6f' % ((time.time() - check_time) / dummy_cnt))

        dummy_x = torch.ones((batch_size, 8)).char().cuda()
        dummy_state = torch.zeros(batch_size, model.STATE_DIM).char().cuda()
        check_time = time.time()
        for i in range(dummy_cnt):
            _, _ = model_trt_int8_strict(dummy_x, dummy_state)
        print('trt int8 strict model: %.6f' % ((time.time() - check_time) / dummy_cnt))
        return
        ## end

        criterion = nn.CrossEntropyLoss()
        state_temp = torch.zeros((batch_size, 8 * 32)).cuda()
        step_acc = 0.0
        step_loss = 0.0
        cnt = 0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = test_model(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('torch', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('trt', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).half().cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.half().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt_f16(inputs, state_temp)

            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('float16', step_acc.item() / cnt, step_loss / loss_cnt)

        state_temp = torch.zeros((batch_size, 8 * 32)).float().cuda()
        step_acc = 0.0
        cnt = 0
        step_loss = 0.0
        loss_cnt = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            normal_outputs, state_temp = model_trt_int8_strict(inputs, state_temp)
            
            _, preds = torch.max(normal_outputs, 1)
            edge_loss = criterion(normal_outputs, labels)
            step_loss += edge_loss.item()
            loss_cnt += 1

            corr_sum = torch.sum(preds == labels.data)
            step_acc += corr_sum.double()
            cnt += batch_size
        print('int8 strict', step_acc.item() / cnt, step_loss / loss_cnt)


def drawGraph(x_value, x_label, y_axis, y_label):
    pass


def CsvToTextOne(csv_file):
    target_csv = pd.read_csv(csv_file)
    file_name, extension = os.path.splitext(csv_file)
    print(file_name, extension)
    target_text = open(file_name + '_1.txt', mode='wt', encoding='utf-8')

    idx = 0
    print(len(target_csv))

    while idx < len(target_csv):
        csv_row = target_csv.iloc[idx]
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
                    save_csv_file.writerow(row[0:1].append(row[2:]))
                  normal_idx += add_normal_len
                  # write abnormal
                  for idx in range(start_abnormal_idx, start_abnormal_idx + add_abnormal_len):
                    row = now_csv.iloc[idx]
                    row = row.fillna(0)
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
