from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routers import predict_router

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





# from fastapi import FastAPI

# from controllers import items, users

# app = FastAPI()

# app.include_router(items.router)
# app.include_router(users.router)

# @app.get("/")
# def read_root():
#     return {"Hello": "world"}

