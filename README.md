# khu_capstone_1

## 연합학습 기반 유해트래픽 탐지
- Pytorch
- CAN protocol 유해 트래픽 데이터 셋
- FedAvg, FedProx, Fed using timestamp, Fed dynamic weight 논문 구현 및 성능 비교

## Model train
- Install [PyTorch](http://pytorch.org)
- Train model
```bash
python3 fed_train.py --packet_num 3 --fold_num 0 --batch_size 128 --lr 0.001 --n_nets 100 --comm_type fedprox --comm_round 50
```
