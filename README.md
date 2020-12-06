# khu_capstone_1

## 자율자동차를 위한 연합학습 기반 유해 트래픽 탐지

![figure2](images\figure2.png)

![figure3](images\figure3.png)

### 서론

  자동차와 같은 ECU 기반 시스템에서 사용하는 표준 프로토콜인 CAN 은 송수신에 대한 인증 체계가 존재하지 않는다. 특히 현대에는 무인 자동차와 같이 자동차 내 ECU 간 네트워크도 외부로 개방된 형태가 등장하고 있다. 자동차 내 네트워크에 침입이 가능한 위험이 조성됨에 따라, 인증체계가 부재한 프로토콜인 CAN의 낮은 보안성을 보완해줄 침입 탐지 및 차단 시스템(IDS/IPS)의 필요성이 대두된다.

  본 연구에서는 데이터의 외부 유출과 실시간 탐지를 위한 방안을 제시한다. 엣지(자동차)마다 자체 모델을 개별적으로 학습한 뒤, 학습 데이터가 아닌 모델 가중치를 중앙 서버와 연동하는 ‘연합 학습’ 을 유해 트래픽 탐지 시스템에 적용해보고자 한다. 또한, 실시간 처리 방식을 구현하기 위하여 RNN 을 기반으로 한 모델을 유해 트래픽 탐지 시스템에서 이용해보고자 한다.

## 모델 성능

단일 엣지 정확도 및 추론 시간

|                  | CNN   | Stateful (Ours) |
| :--------------: | ----- | --------------- |
| 추론 시간 (msec) | 19.49 | 0.49            |
|    정확도 (%)    | 99%   | 98%             |

연합학습 적용 정확도

| 연합학습 방법 | CNN   | Stateful (Ours) |
| ------------- | ----- | --------------- |
| 엣지 평균     | 57.95 | 90.94           |
| FED AVG       | 60.36 | 96.88           |
| FED PROX      | 91.63 | 97.50           |
| FED Timestamp | 60.35 | 96.74           |
| FED Dynamic   | 94.65 | 97.45           |

## 실험 환경

- Google Colab or Linux (Ubuntu)

- Pytorch >= 1.6.0
- Python >= 3.6
- NVIDIA GPU + CUDA CuDNN

## 모델 학습
- Train model
```bash
python3 fed_train.py --packet_num 3 --fold_num 0 --batch_size 128 --lr 0.001 --n_nets 100 --comm_type fedprox --comm_round 50
```

## 사용한 데이터셋
- [CAN Dataset](https://sites.google.com/a/hksecurity.net/ocslab/Dataset/CAN-intrusion-dataset)