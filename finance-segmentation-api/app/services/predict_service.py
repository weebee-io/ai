import numpy as np
from app.models.ml_model import load_model
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut

class PredictService:
    def __init__(self):
        self.model = load_model()

    def predict(self, data: InputData) -> PredictionOut:
        X = np.array([list(data.model_dump().values())])
        pred = int(self.model.predict(X)[0])

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = float(max(self.model.predict_proba(X)[0]))

        return PredictionOut(cluster=pred, proba=proba)
