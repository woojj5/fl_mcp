# CLI 및 HTTP 인터페이스

## 실행 명령

| 명령 | 동작 | 기본값 |
|---|---|---|
| `python run.py` 또는 `dashboard` | TLS 대시보드 | `0.0.0.0:5000`, `./tracking.db` |
| `python run.py server` | TLS 분산 서버 | `0.0.0.0:8080`, simple_cnn, CIFAR-10, 20 라운드, 최소 2 클라이언트 |
| `python run.py client` | TLS 분산 클라이언트 | `0.0.0.0:8081`, client_0, HTTPS 서버, IID, 자동 시작 |

대시보드 옵션은 `--host`, `--port`, `--db`, `--fl-server-url`, `--fl-client-urls`다. 서버 옵션은 `--model`, `--num-classes`, `--dataset`, `--rounds`, `--min-clients`, `--clients-per-round`, `--strategy`, `--db`, `--task-id`다.

클라이언트는 `--client-id`, `--server`, `--my-address`, `--host`, `--port`, `--model`, `--num-classes`, `--dataset`, `--data-dir`, `--partition-id`, `--num-clients-total`, `--split-type`, `--alpha`, `--batch-size`, `--local-epoch`, `--lr`, `--optimizer`, `--gpu`, `--seed`, `--no-auto-start`를 받는다. 분할은 `iid`, `dirichlet`, `class_based`; optimizer는 `sgd`, `adam`이다.

## 스킬 실행기

실행 형식은 `python skills/skill.py <명령> [옵션]`이다.

| 명령 | 동작 |
|---|---|
| `list`, `specs`, `config`, `check`, `lint`, `test` | 스킬 목록, 문서, YAML, 저장소, Python 구문, unittest 관리 |
| `dashboard [인자...]` | 대시보드 실행 |
| `server [인자...]` | FL 서버 실행 |
| `client [인자...]` | FL 클라이언트 실행 |
| `compose [config\|up\|down\|ps] [--build]` | Docker Compose 조작 |

## 대시보드 경로

- `/`: 시뮬레이션 설정 폼 및 학습 기록
- `/deployment`: FL 서버와 클라이언트 상태 모니터
- `/api/simulation/start`: JSON 설정으로 시뮬레이션 시작
- `/api/simulation/status`: 시뮬레이션 진행 상태
- `/api/metrics`: SQLite 학습 지표
- `/api/deployment`: 원격 노드 상태 JSON

## FL 서버 API

| 메서드 | 경로 | 목적 |
|---|---|---|
| GET | `/health` | 공개 상태 점검 |
| GET | `/status` | 서버 상태와 라운드 정보 |
| POST | `/register` | 클라이언트 등록 |
| GET | `/global-model` | 현재 전역 모델 다운로드 |
| POST | `/upload-weights` | 클라이언트 업데이트 업로드 |
| POST | `/predict` | 전역 모델 예측 |
| GET | `/admin/clients` | 등록 클라이언트 목록 |
| POST | `/admin/next-round` | 준비된 업데이트로 라운드 강제 진행 |
| POST | `/admin/reset` | 서버 메모리 상태 초기화 |

## FL 클라이언트 API

| 메서드 | 경로 | 목적 |
|---|---|---|
| GET | `/health` | 공개 상태 점검 |
| GET | `/status` | 클라이언트 상태 및 학습 지표 |
| POST | `/train` | 학습 라운드 1회 수동 실행 |
| POST | `/predict` | 로컬 모델 예측 |
| POST | `/upload-dataset` | multipart `file` 필드로 ZIP·NPZ·CSV 업로드 후 로컬 학습에 사용 |
| POST | `/admin/start` | 자동 학습 루프 시작 |
| POST | `/admin/stop` | 자동 학습 루프 정지 |

등록 payload는 `{client_id, address}`, 업데이트 payload는 `{client_id, round, weights, samples, loss, accuracy}`다. `weights`는 Base64 인코딩 NPZ 문자열이다. 사용자 데이터는 각 클라이언트에 별도로 업로드하며 서버로 전송하지 않는다.

## 인증과 암호화

- `/health` 외 FL 서버·클라이언트 API: `Authorization: Bearer <FL_API_TOKEN>`
- 대시보드: HTTP Basic 인증 (`FL_DASHBOARD_USER`, `FL_DASHBOARD_PASSWORD`)
- 모든 실행 진입점: `FL_TLS_CERT`, `FL_TLS_KEY`가 필요하고 HTTPS로 제공
- 내부 HTTPS 요청: `FL_TLS_CA`로 서버 인증서를 검증
- Docker Compose는 `.env`의 API 토큰·대시보드 계정과 `certs/server.crt`, `certs/server.key`를 요구
