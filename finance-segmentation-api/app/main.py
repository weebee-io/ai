from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
from app.routers import predict_router
from app.controllers.predict_controller import get_kafka_service, PredictService

app = FastAPI(
    title="Financial-Literacy Segmentation API",
    version="1.0.0",
    description="Cluster prediction service for financial‑literacy personas",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router.router)

# 상황 체크
@app.get("/ping")
def ping():
    return {"status": "ok"}


# 서버 시작 시 Kafka 소비자 자동 시작
@app.on_event("startup")
async def startup_event():
    # PredictService 인스턴스 생성
    predict_service = PredictService()
    # KafkaConsumerService 인스턴스 가져오기
    kafka_service = get_kafka_service(predict_service)
    # Kafka 소비자 시작
    start_result = kafka_service.start()
    print(f"Kafka consumer startup result: {start_result}")


# 서버 종료 시 Kafka 소비자 자동 중지
@app.on_event("shutdown")
async def shutdown_event():
    # PredictService 인스턴스 생성
    predict_service = PredictService()
    # KafkaConsumerService 인스턴스 가져오기
    kafka_service = get_kafka_service(predict_service)
    # Kafka 소비자 중지
    stop_result = kafka_service.stop()
    print(f"Kafka consumer shutdown result: {stop_result}")





# from fastapi import FastAPI

# from controllers import items, users

# app = FastAPI()

# app.include_router(items.router)
# app.include_router(users.router)

# @app.get("/")
# def read_root():
#     return {"Hello": "world"}

