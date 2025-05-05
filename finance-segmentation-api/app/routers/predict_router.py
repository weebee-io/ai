from fastapi import APIRouter
from app.controllers.predict_controller import predict_controller
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut

router = APIRouter(prefix="/predict", tags=["Predict"])

router.post("/", response_model=PredictionOut)(predict_controller)
