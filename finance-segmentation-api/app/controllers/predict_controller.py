from fastapi import Depends, HTTPException, status
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut
from app.services.predict_service import PredictService

def predict_controller(
    payload: InputData,
    service: PredictService = Depends(PredictService),
) -> PredictionOut:
    try:
        return service.predict(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )
