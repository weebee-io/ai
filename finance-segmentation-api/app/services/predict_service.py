import numpy as np
import pandas as pd
import joblib 
from app.core.config import settings 
from app.models.ml_model import load_model
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut

class PredictService:
    def __init__(self):
        self.model = load_model()
        try:
            self.scaler = joblib.load(settings.SCALER_PATH)
        except FileNotFoundError:
            print(f"ERROR: Scaler file not found at {settings.SCALER_PATH}. Predictions may be inaccurate.")
            self.scaler = None 
        except Exception as e:
            print(f"ERROR: Could not load scaler: {e}")
            self.scaler = None 

    def predict(self, data: InputData) -> PredictionOut:
        input_dict = data.model_dump()
        feature_names = list(input_dict.keys()) 
        feature_values = [list(input_dict.values())]

        X_df = pd.DataFrame(feature_values, columns=feature_names)
        print(f"Input DataFrame (X_df):\n{X_df}")

        if self.scaler:
            try:
                X_scaled_np = self.scaler.transform(X_df)
                print(f"Scaled input (X_scaled_np):\n{X_scaled_np}")
            except Exception as e:
                print(f"ERROR: Could not scale data: {e}")
                X_scaled_np = X_df.values 
        else:
            X_scaled_np = X_df.values
            print("WARNING: Scaler not loaded. Using unscaled data for prediction.")

        # Get the predicted class index from the model (e.g., 0, 1, or 2)
        predicted_class_index_raw = self.model.predict(X_scaled_np)[0]
        predicted_class_index = int(predicted_class_index_raw) # Ensure it's an integer

        print(f"Predicted class index: {predicted_class_index}") # Log the predicted class index

        proba_for_response = None # This will store the probability of the predicted class for the API response
        if hasattr(self.model, "predict_proba"):
            all_class_probabilities = self.model.predict_proba(X_scaled_np)[0] # Get probabilities for all classes
            # Log all class probabilities, formatting as a list for readability
            print(f"Probabilities for all classes [BRONZE, SILVER, GOLD]: {all_class_probabilities.tolist()}")
            
            # The probability for the API response should be the probability of the predicted class
            proba_for_response = float(all_class_probabilities[predicted_class_index])
        
        # Return PredictionOut with the string representation of the predicted_class_index and its probability
        return PredictionOut(lit_level=str(predicted_class_index), proba=proba_for_response)
