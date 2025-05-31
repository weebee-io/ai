# AI 프로젝트: [Weebee]

> 금융 이해도 분석 / 랭크 배정 / 예측 모델링에 대한 설명

---

## 📂 프로젝트 구조

현재 상황 추후 게속 업데이트 해야함!
```bash
finance-segmentation-api/
├── app/
│   ├── __init__.py
│   ├── main.py               # 진입점
│   ├── core/
│   │   └── config.py         # 설정‧환경변수
│   ├── models/
│   │   └── ml_model.py       # joblib 모델 로드
│   ├── schemas/
│   │   ├── input.py          # Pydantic 입력 스키마
│   │   └── output.py         # 예측 결과 스키마
│   ├── services/
│   │   └── predict_service.py# 비즈니스 로직
│   ├── controllers/
│   │   └── predict_controller.py   # HTTP 관점의 컨트롤러
│   └── routers/
│       └── predict_router.py # URL → Controller 매핑
├── models/                   # (선택) ML 학습 산출물 보관 디렉터리
│   └── test
│       └── rf_model.joblib
├── tests/
│   └── test_predict.py
└── requirements.txt

```

---

## 👥 팀원 소개

| 이름 | 역할 | GitHub |
|------|------|--------|
| 박혁준 | API 개발/배포 | [@hyukjunmon](https://github.com/hyukjunmon) |
| 오세현 | 데이터 수집/분석 | [@MoominHunter](https://github.com/MoominHunter) |
| 정창훈 | 모델 개발/학습 | [@muramasa404](https://github.com/muramasa404) |

---

## 🎯 프로젝트 목표

- ✅ 금융 소비자 금융 이해도 자동 분류
- ✅ 비지도 → 지도 학습 기반 모델 설계
- ✅ 예측 결과 기반 API 서비스 제공

---

## 🚀 실행 방법 (로컬 개발)

## 파이썬 3.11로 설치해야함 

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. FastAPI 실행
uvicorn app.main:app --reload
```

→ [http://localhost:8000/docs](http://localhost:8000/docs) 에서 Swagger UI 확인 가능

---

## 도커허브에 배포 및 ec2로 돌리는 법 까지.
```
# 1. 도커 빌드
docker build -t finance-segmentation-api:latest .

# 2. 도커 로그인
docker login

# 3. 이미지 태그 지정
docker tag finance-segmentation-api:latest yourusername/finance-segmentation-api:latest

# 4. 이미지 업로드
docker push yourusername/finance-segmentation-api:latest

# 5. ec2에서 이미지 내려받기
docker pull yourusername/finance-segmentation-api:latest

# 6. 기존 컨테이너 중지 및 제거
docker stop finance-api
docker rm finance-api

# 7. 새 이미지로 컨테이너 재시작
docker run -d --name finance-api -p 8080:8080 --restart=always yourusername/finance-segmentation-api:latest
```
---
## 📊 사용 기술

- **언어**: Python 3.11
- **ML 모델**: scikit-learn (RandomForest, KMeans), joblib
- **웹 프레임워크**: FastAPI
- **시각화**: matplotlib, seaborn
- **배포**: Docker, GitHub Actions (선택)
- **기타**: Pandas, NumPy, Pydantic

---

## 🧪 테스트
일단 프로젝트 디렉토리 추가 X
```bash
pytest tests/
```

---

## 📁 데이터 출처 및 전처리

> 예: 실제 카드 사용 데이터, 케글 등 명시하면 될듯?  
> *[민감한 데이터는 포함하지 않음]*

---

## 🛠 향후 개선 계획

- [ ] XGBoost 기반 성능 개선
- [ ] 클러스터링 자동 평가 도구 추가
- [ ] MLOps 자동화 (optional)

---

## 📄 라이선스

MIT License © 2025 AI Team
