# 추론시간 개선 - 양자화 시도

## Pytorch quantization
- Pytorch 가 제공하는 라이브러리로 양자화 학습.
- 하지만 cpu 에서만 실행 가능, 또한 모델의 채널 수를 신중하게 고르지 않으면 cpu 에서 조차 속도 개선이 미미함. 
- 양자화 과정으로 학습된 모델은 pytorch model -> onnx -> tensorRT 변환이 불가능하여 gpu 에서 실행 불가능.

## TensorRT
- Google Colab - install_tensorRT

- 양자화 학습을 사용하지 않고, 라이브러리를 활용하여 모델의 정밀도 감소 및 양자화 시도. 

- 모델에 따라 속도 차이가 크고 아래 단계의 정밀도가 더 빠른 경우가 있었음

- 정확한 이해가 필요해 보임 (사용법 미숙, 입력 값은 float 등)

  | Inference Time(msec) | Densenet - 32 packet | Ours - 1 packet |
  | -------------------- | -------------------- | --------------- |
  | Torch - float32      | 19.49                | 0.49            |
  | TensorRT - float32   | 4.30                 | 0.37            |
  | TensorRT - float16   | 4.32                 | 0.35            |
  | TensorRT - int8      | 3.70                 | 0.41            |

  