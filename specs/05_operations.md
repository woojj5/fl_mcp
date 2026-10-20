# 배포 및 운영

## 준비

```bash
pip install -r requirements.txt
python scripts/generate_dev_cert.py
```

복사한 `.env.example`을 `.env`로 두고 `FL_API_TOKEN`, `FL_DASHBOARD_PASSWORD`를 강한 임의 값으로 교체한다. 독립 Python 실행에는 환경변수로 `FL_API_TOKEN`, `FL_DASHBOARD_USER`, `FL_DASHBOARD_PASSWORD`, `FL_TLS_CERT`, `FL_TLS_KEY`, `FL_TLS_CA`를 설정해야 한다. `.env`는 Compose가 자동으로 읽으며 Python은 자동 로딩하지 않는다.

## Python 실행

```bash
python run.py                         # TLS 대시보드
python run.py server --port 8080
python run.py client --server https://localhost:8080 --client-id client_0 --port 8081
```

모든 서비스는 인증서·키가 없으면 시작하지 않는다. 자체 서명 개발 인증서는 로컬 테스트에만 사용하고 외부 배포에는 CA 인증서를 사용한다.

## Docker Compose

`.env.example`을 `.env`로 복사하고 비밀정보를 바꾼 뒤 `python scripts/generate_dev_cert.py`와 `docker compose up --build`를 실행한다.

| 서비스 | 컨테이너 포트 | 호스트 포트 |
|---|---:|---:|
| `fl-server` | 8080 | 8080 |
| `fl-client-0` | 8081 | 8081 |
| `fl-client-1` | 8082 | 8082 |
| `fl-dashboard` | 5000 | 5000 |

Compose 환경변수는 인증정보(`FL_API_TOKEN`, `FL_DASHBOARD_USER`, `FL_DASHBOARD_PASSWORD`)와 모델·데이터셋·라운드·클라이언트 수·분할·로컬 학습 옵션이다. 보안 변수는 필수이며 누락 시 Compose 설정이 실패한다.

## 저장 데이터와 업로드

- Docker named volume `fl-data`를 `/data`에 연결하며 서버와 대시보드가 `/data/server_tracking.db`를 공유한다.
- 클라이언트 데이터셋은 각 노드가 설정한 로컬 경로에 저장한다.
- ZIP ImageFolder, NPZ(`x_train`, `y_train`), 이미지 픽셀 CSV를 클라이언트의 인증된 `POST /upload-dataset`에 전송할 수 있다.
- 서버에는 가중치 업데이트와 집계 지표가 전송되며 원본 학습 샘플은 보내지 않는다.
- Compose의 `certs/` 디렉터리에는 TLS 인증서와 개인키가 필요하다. 생성된 개발 개인키와 `.env`는 버전 관리에 포함하지 않는다.

## 보안 및 운영 범위

서버·클라이언트 API는 Bearer token, 대시보드는 Basic 인증을 사용한다. HTTPS 클라이언트는 `FL_TLS_CA` 인증서를 검증한다. 개발용 생성 인증서는 공개 CA 신뢰를 제공하지 않는다. 운영 배포 전에 CA/인증서 갱신, secret manager, 방화벽, rate limit, DB 백업 및 로그 보존을 구성한다.

라운드 상태는 서버 메모리에 있으므로 재시작하면 복원되지 않는다. 대시보드 시뮬레이션은 대시보드 프로세스의 백그라운드 스레드에서 실행한다.
