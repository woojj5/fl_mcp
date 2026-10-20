# 설정 및 데이터 요구사항

## YAML 실험 설정

`config.yaml`은 다음 실험 설정 그룹을 제공한다. 현재 `skills/skill.py config`는 파일을 읽어 요약하지만 서버·클라이언트 실행 시 YAML을 자동 적용하지 않는다.

| 그룹 | 항목 |
|---|---|
| 최상위 | `task_id`, `seed`, `model` |
| `data` | 데이터셋·경로·분할 유형·학습 종류·클라이언트 수·Dirichlet alpha·최소 샘플·클라이언트 가중치 |
| `server` | 라운드 수·참여 수·집계 전략·테스트 주기·체크포인트·배치 크기·최소 클라이언트·전략별 파라미터·차등 프라이버시 옵션 |
| `client` | 배치 크기·로컬 epoch·옵티마이저·학습률·momentum·weight decay·로컬 평가 |
| `system` | GPU 수·분산 실행·Flower 시뮬레이션·자원 이질성 설정 |
| `tracking` | SQLite 파일·로그 파일·로그 수준 |

기본값은 task `fl_cifar10`, seed 42, `simple_cnn`, CIFAR-10, IID, 클라이언트 10개, Dirichlet alpha 0.5, 서버 20 라운드·라운드당 5개 클라이언트, `fedavg`, 클라이언트 5 epoch·SGD·학습률 0.01이다.

문서화된 YAML 분할 값은 `iid`, `dir`, `class`, `niid`이며, 클라이언트 CLI 값은 `iid`, `dirichlet`, `class_based`다. 두 인터페이스의 명칭 차이는 설정 검증 시 통일해야 한다. 현재 CLI가 YAML 파일을 실제로 로딩하는 경로는 확인되지 않았다.

## 데이터셋 및 분할

README와 설정 파일은 CIFAR-10, CIFAR-100, MNIST, FashionMNIST, SVHN을 기본 데이터셋 예시로 든다. 사용자 제공 데이터 업로드 지원 형식은 README 설명 기준 ZIP(ImageFolder 또는 flat), NPZ, CSV다.

- ImageFolder ZIP은 `train/class명/이미지`와 선택적 `test/` 디렉터리를 사용한다.
- Flat ZIP은 클래스별 이미지 파일을 포함하고 기본 80/20 분할을 적용한다고 README가 설명한다.
- NPZ는 `x_train`, `y_train`, `x_test`, `y_test` 키를 사용한다.
- CSV는 첫 열을 라벨, 나머지를 입력 특성으로 설명한다.
- 데이터 분할은 IID, Dirichlet, class-based 방식을 제공하는 것이 목표다.

분산 클라이언트와 시뮬레이션은 torchvision 내장 데이터셋을 로드하며 IID, Dirichlet, class-based 방식으로 분할한다. 클라이언트의 `/upload-dataset`은 ZIP ImageFolder, NPZ(`x_train`, `y_train`), CSV(첫 열 라벨, 뒤 열 이미지 픽셀)를 저장하고 이후 로컬 학습에서 사용한다. 사용자 데이터 채널은 MNIST/FashionMNIST 설정일 때 1, 그 외는 3이어야 한다.

## 모델과 집계 전략

런타임 모델은 `simple_cnn`, `lenet`, `mlp`/`logistic`, torchvision `resnet18/34/50/101`이다. `fedavg_cnn`, `fedbn_cnn`, `floco_net`, `rnn`, `gru` 이름은 기본 CNN/MLP 구조를 공유하는 호환 별칭이다.

서버 CLI는 FedAvg, FedProx, FedAdam, FedYogi, FedAvgM, FedMedian, Krum, DP-FedAvg를 허용한다. YAML에만 적힌 FedTrimmedAvg, Bulyan, qFedAvg 등은 현재 서버 CLI에서 사용할 수 없다.

## 학습 재현성

실험을 비교할 때 task ID, seed, 모델, 데이터셋, 데이터 분할과 클라이언트 수, 라운드 설정, 집계 전략, 로컬 epoch, optimizer, learning rate, batch size 및 체크포인트 위치를 함께 기록한다.
