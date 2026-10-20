# 구현 현황과 후속 과제

## 구현된 기능

- `skills/skill.py`의 FL 명령과 `run.py` 대시보드·서버·클라이언트 진입점
- PyTorch 클라이언트 학습, 모델 직렬화, 서버 집계, 로컬 시뮬레이션 UI
- SQLite 실험/라운드 기록과 분산 노드 상태 화면
- 클라이언트의 ZIP ImageFolder, NPZ, 이미지 픽셀 CSV 데이터 업로드 및 후속 로컬 학습
- Bearer token API 인증, 대시보드 Basic 인증, TLS 필수 서비스 실행과 인증서 검증
- 개발용 인증서 생성 스크립트, Compose 보안 설정, `.env.example`

## 실행 검증

`python skills/skill.py lint`가 저장소의 Python 파일 구문을 확인한다. `python skills/skill.py test`의 통합 테스트는 다음을 검사한다.

- 토큰 없는 서버 API 요청이 401로 거부되는지
- NPZ 업로드 데이터가 클라이언트 로컬 학습에 쓰이는지
- 두 클라이언트 업데이트가 서버에서 한 라운드로 집계되고 SQLite에 기록되는지
- 자체 서명 로컬 인증서를 신뢰한 HTTPS 요청이 성공하는지
- 대시보드 Basic 인증과 시뮬레이션 실행 UI가 제공되는지
- 시뮬레이션이 라운드를 수행하고 기록을 저장하는지

실제 공개 데이터셋 다운로드, GPU, Docker 전체 구동 및 외부 CA TLS handshake는 이 통합 테스트에서 검증하지 않는다.

## 알려진 제한 및 후속 과제

- CSV는 첫 열 라벨과 28×28 grayscale 또는 32×32 RGB 픽셀을 전제로 한다. 임의 표 데이터는 지원하지 않는다.
- 모델 이름 `fedavg_cnn`, `fedbn_cnn`, `floco_net`, `rnn`, `gru`는 간소화된 CNN/MLP 호환 별칭이다.
- FedAvgM, FedAdam, FedYogi, Krum 구현은 시연용이며 DP-FedAvg는 형식적인 차등 프라이버시 보장을 제공하지 않는다.
- `config.yaml`은 스킬에서 확인할 수 있지만 server/client 실행 옵션으로 자동 적용되지 않는다.
- Compose는 기본 클라이언트 두 개를 정의한다. 추가 노드는 Compose 서비스를 추가하거나 별도 실행해야 한다.
- 라운드 상태는 서버 메모리에 있고 대시보드 시뮬레이션 진행도 프로세스 재시작 후 복구하지 않는다.
- 개발용 자체 서명 인증서 대신 운영 CA 인증서와 비밀정보 회전 체계를 적용해야 한다.
- DB 백업/보존, API rate limit, 장기 운영 장애 복구는 별도 작업이다.
