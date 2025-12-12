
# 🛡️ Tank Challenge 시뮬레이터 - 전장상황인식

  YOLO 기반 실시간 전장 상황 인식, 자율주행, 자동 포격 시뮬레이션 시스템

---
## 📌 프로젝트 설명

이 프로젝트는 **Unity 기반 전장 시뮬레이터에서 전차가 스스로 주변 상황을 인식하고, 적 전차를 탐지하여 자동으로 포격 및 회피 기동까지 수행하는 전장상황인식 시스템**을 구현하는 것을 목표로 한다.


프로젝트는 아래의 문제를 해결하기 위해 설계되었다:

### ✅ 해결하려는 문제

- 전장 환경에서 **실시간 객체 인식 성능 부족(FPS 저하, 잔상)**
- 적군과 아군 보병, 차량 및 전차를 정확히 분류하기 어려움
- 단순 Detect API는 전투 시나리오에 활용하기 부족
- 실시간 추론 + 경량화 모델 + HUD 시각화 + 자율주행/사격 로직까지 통합된 시스템의 부재
    
### 프로젝트가 제공하는 해결책

- YOLO 모델 파인튜닝 및 경량화(ONNX/TensorRT 가능)
- 실시간 객체 인식 + 거리 추정 + HUD 시각화
- A* 기반 장애물 회피 자율 주행
- 적 전차 자동 조준/발사 + 회피 → 복귀 → 재조준 사이클 구현
- Docker 기반으로 Unity ↔ AI 서버 양방향 통신 자동화
    
---

## 🎮 전투 시나리오 흐름

> 프로젝트는 실제 전투 상황을 반영한 시나리오 기반으로 제작되었다:

1. 전차가 시작 지점에서 출발
2. 주변 객체(아군/적군 보병, 차량, 바위, 적 전차)를 YOLO로 실시간 탐지
3. 탐지된 객체의 거리 추정 → HUD에 표시
4. **A\* 경로 탐색**을 통해 강을 건너면서 장애물(바위/차량)을 회피
5. 강 건너편에서 적군 전차 3대 발견
6. 자동 조준 후 포격 → 장전 시간 대기
7. 장전 중에는 회피 기동 → 복귀 → 재조준
8. 3대 전차를 모두 격멸할 때까지 반복
9. 모든 전차를 제거하면 아군 베이스캠프로 이동 후 시나리오 종료
    
이 시스템은 단순 객체인식을 넘어,
> “보는 것(Inference)” → “판단하는 것(Decision)” → “행동(Action)”
\
> 통합 구현한 **전장 인공지능 파이프라인**이다.

---

## 🚀 프로젝트 특징 (Key Features)

### 🔥 1. YOLO 기반 전장 객체 인식 (5 Classes)

- blue(아군), red(적군), rock, car, tank
- Fine-tuning 데이터셋 직접 구축
- Roboflow 기반 버전 관리 및 Auto Labeling 파이프라인 구축
    
### ⚙️ 2. ONNX / TensorRT로 모델 경량화 가능

- TensorRT 적용 시 GPU 환경에서 FPS 향상
- ONNX로 CPU 기반 추론 테스트 가능
    
### 🛰️ 3. 실시간 HUD 시각화

- 객체명, 거리, 전장 상황 요약
- 아군/적군 색상 고정
- Real-time overlay 화면 구성

### 🛞4. A* 알고리즘 기반 자율주행

- 지도(Map) JSON에서 장애물 좌표 파싱
- 바위, 차량 등 장애물 자동 회피
- Waypoint 기반 전술 이동
### 🎯 5. 자동 포격 시스템

- 적 전차 인식 → 조준 → 발사 → 회피 → 복귀 루틴
- 발사 후 장전 시간 고려해 전술적 회피 기동 수행

### 🐳 6. Docker 기반 AI 서버 아키텍처

- Flask 서버로 Unity와 REST API 통신
- `/get_action`, `/info`, `/update_bullet` API로 실시간 전술 판단 지원

### 📱 7. 모바일 클라이언트

- RAG와 Super-Resolution

---

## 📁 **프로젝트 구조**

```yaml
project/
├── detector_gui                   # HUD + YOLO 시각화 모듈
│   ├── detect_module
│   │   ├── gui.py                 # 화면 HUD 렌더링
│   │   ├── vision.py              # 객체 인식 및 거리 추정 로직
│   │   └── __init__.py
│   ├── detect.py                  # 메인 GUI 실행 파일
│   ├── requirements.txt
│   └── weights
│       └── 5cls_v7.pt             # 파인튜닝된 YOLO 모델
│
├── docker-compose.yml             # 전장 서버 + Detector 통합 실행
│
└── flask_server                   # 전술 판단 서버 (Docker)
    ├── Dockerfile
    ├── requirements.txt
    ├── data/output.csv
    ├── log_data/output.csv
    ├── map
    │   └── MAP.map
    └── server_module
        ├── combat.py              # 자동 조준·포격 계산
        ├── navigation.py          # A* 기반 자율주행
        └── server.py              # Flask API 서버
```

---

## 🛠️  설치 및 실행 방법  

### 1. 프로젝트 클론
```bash
git clone https://github.com/Seyun-lab/Acorn_1Team.git
cd Acorn_1Team/project
```

### 2. Detector 환경 설정

```bash
cd detector_gui
pip install -r requirements.txt
```

### 3. Flask 서버(Docker) 실행

```bash
cd ..
docker-compose up --build
```

### 4. Unity 시뮬레이터 실행 후 detect.py 실행

```
python detector_gui/detect.py
```

Unity에서 탱크가 스스로 이동, 탐지, 포격하며 시나리오가 시작된다.

---

## 🖥️  프로젝트 사용 예시 (Screenshots)

**HUD 예시 / 객체 인식 예시 / 포격 시나리오**

→ (이미지 또는 영상 첨부 자리)

---

## 👥 팀원

| Name     | Role                                                                                                  | Github / Email                |
| -------- | ----------------------------------------------------------------------------------------------------- | ----------------------------- |
| 박세윤(팀장)  | 형상관리 운영 및 기획, 검증 테스트, UX 어플 구현                                                                        | https://github.com/Seyun-lab  |
| 윤천희(부팀장) | 이미지 데이터 수집 및 가공, 객체 인식 모델 학습, 코드 통합, <br>이미지 초해상화                                                     | https://github.com/gyeoulnamu |
| 김소영      | waypoint이동 구현, 포격 고도화(roll보정, 빚맞춤시 보정사격), <br>LIDAR 포인트클라우드분석                                         | soyoungkim0327@gmail.com      |
| 박석번      | Docker 기반 프로젝트 아키텍처 설계, 이미지 데이터셋 버전 관리 및 품질 관리, <br>Auto Labeling 파이프라인 구축, 맵 제작                      | https://github.com/bbun-550   |
| 박성재      | Auto Labeling 구현, 전장상황인식 HUD 구현, <br>YOLO 기반 객체 크기 기반 거리 추정 구현                                        | https://github.com/OiSungJae  |
| 박태응      | 이동 알고리즘 분석 , LIDAR 이용한 객체 탐지, <br>pitch에 따른 포 탄착 지점 분석                                                | eyfuoufye@naver.com           |
| 심태훈      | 강화학습(PPO, SAC), 코드 모듈화(주행, 포격, 서버), <br>주행 알고리즘(A*, pure pursuit)적용, 포격 후 회피기동 계획 및 구현, <br>주행시나리오 계획 | https://github.com/shimtaehun |
| 정윤희      | 시뮬레이터 상 가상 이미지 센서 구현, 이미지 데이터 수집 및 라벨링 <br>최종 발표 담당                                                   | https://github.com/yunhee103  |


---

## 📚 참고 자료

- [Tank Challenge Simulator Docs](https://bangbaedong-vallet-co-ltd.gitbook.io/tank-challenge)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Label Studio Docs](https://labelstud.io/guide/)
- [Roboflow Docs](https://docs.roboflow.com/)


---

## ⚖️ License

본 프로젝트는 MIT License를 따릅니다.