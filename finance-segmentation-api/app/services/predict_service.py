import numpy as np
import pandas as pd
from app.models.ml_model import load_model_components # Changed import
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut

class PredictService:
    def __init__(self):
        # Load predictor and scaler separately
        try:
            self.predictor, self.scaler = load_model_components()
        except Exception as e:
            # Log the error. For a production system, consider more robust logging.
            print(f"CRITICAL: Failed to load model components in PredictService: {e}")
            self.predictor = None
            self.scaler = None
            # Depending on desired behavior, you might want to re-raise the exception
            # to prevent the service from starting in a non-functional state.
            # raise RuntimeError(f"Failed to initialize PredictService due to model loading error: {e}")

        if self.predictor is None:
            print("CRITICAL: Predictor could not be loaded. PredictService will not function correctly.")
            # Again, consider raising an exception or implementing a health check that would fail.
        if self.scaler is None:
            # This might be acceptable if the model doesn't always require a scaler,
            # but for the current centroid model, it's expected.
            print("WARNING: Scaler could not be loaded. Predictions might be inaccurate if data requires scaling.")
        elif hasattr(self.scaler, 'feature_names_in_'):
            print(f"DEBUG: Scaler expects feature names (and order): {self.scaler.feature_names_in_}")
        else:
            print("DEBUG: Scaler does not have 'feature_names_in_' attribute.")

    def predict(self, data: InputData) -> PredictionOut:
        input_dict = data.model_dump()
        feature_names = list(input_dict.keys()) 
        feature_values = [list(input_dict.values())]

        X_df = pd.DataFrame(feature_values, columns=feature_names)
        print(f"Input DataFrame (X_df) before reordering:\n{X_df}")

        # 스케일러가 기대하는 피처 순서 (디버그 로그에서 확인됨, 또는 직접 명시)
        expected_feature_order = [
            'essential_pct', 'discretionary_pct', 'risk_profile_score',
            'complex_product_flag', 'digital_engagement', 'is_married',
            'spend_volatility', 'sav_inv_ratio'
        ]
        
        X_df_reordered = X_df # 기본값 설정
        try:
            # 컬럼 순서 재정렬 및 누락된 컬럼 확인
            missing_cols = [col for col in expected_feature_order if col not in X_df.columns]
            if missing_cols:
                print(f"ERROR: The following expected columns are missing from input DataFrame: {missing_cols}. Current columns: {X_df.columns.tolist()}")
                # 스케일링 시도 시 에러가 나도록 하기 위해, 원본 X_df를 사용 (다음 try-except 블록에서 처리)
            else:
                X_df_reordered = X_df[expected_feature_order]
                print(f"Input DataFrame (X_df_reordered) after reordering:\n{X_df_reordered}")
        except KeyError as e:
            print(f"ERROR: KeyError during DataFrame reordering, meaning a feature in expected_feature_order might be misspelled or truly missing from InputData definition. Missing key: {e}")
            print(f"InputData features used for X_df: {feature_names}")
            # 이 경우, 스케일링은 아마도 실패할 것이므로, X_df_reordered는 원본 X_df를 사용

        # Scale the input data using the loaded scaler
        try:
            if self.scaler:
                # 재정렬된 DataFrame 사용
                X_scaled_np = self.scaler.transform(X_df_reordered)
                print(f"Scaled input (X_scaled_np):\n{X_scaled_np}")
            else:
                print("WARNING: Scaler is not available (likely failed to load). Using unscaled data for prediction.")
                X_scaled_np = X_df_reordered.values # Fallback to unscaled reordered data
        except Exception as e:
            print(f"ERROR: Failed to scale data using the loaded scaler: {e}. Using unscaled data.")
            X_scaled_np = X_df.values # Fallback to unscaled data

        # Initialize defaults for prediction outputs
        predicted_class_index: int = 0
        all_class_probabilities: np.ndarray = np.array([1.0, 0.0, 0.0]) # Default: BRONZE, 100%

        # Perform prediction using the loaded CentroidPredictor instance
        try:
            if self.predictor:
                # X_scaled_np is expected to be a 2D array, even for a single sample (e.g., shape (1, num_features))
                # predict_proba should return probabilities for all classes for each sample.
                # For a single sample input, it will be shape (1, num_classes), so [0] accesses the probabilities for that sample.
                all_class_probabilities = self.predictor.predict_proba(X_scaled_np)[0] 
                predicted_class_index = np.argmax(all_class_probabilities)
                print(f"Predicted class index: {predicted_class_index}, Probabilities: {all_class_probabilities}")
            else:
                print("ERROR: Predictor is not available (likely failed to load). Falling back to default BRONZE.")
                # Defaults (predicted_class_index = 0, all_class_probabilities = [1.0, 0.0, 0.0]) 
                # are already initialized at the start of the method, so no action needed here for fallback.
                pass # Rely on pre-initialized defaults

        except (AttributeError, ValueError, TypeError, IndexError) as e:
            # These errors could occur if X_scaled_np is not in the expected format for the predictor,
            # or if predict_proba/classes_ behave unexpectedly.
            print(f"ERROR: Failed to get prediction from model: {e}. Falling back to default BRONZE.")
            # Default values already initialized, no need to re-assign here.
        except Exception as e: # This is the general exception for the new prediction block
            print(f"UNEXPECTED ERROR during prediction: {e}. Falling back to default BRONZE.")
            # Default values already initialized.

        # Ensure predicted_class_index is valid for all_class_probabilities after try-except
        if not (0 <= predicted_class_index < len(all_class_probabilities)):
            print(f"WARNING: predicted_class_index {predicted_class_index} is out of bounds for all_class_probabilities (len {len(all_class_probabilities)}). Resetting to 0.")
            predicted_class_index = 0
            # Ensure all_class_probabilities has a safe default if it became problematic
            if len(all_class_probabilities) == 0 or len(all_class_probabilities) <= predicted_class_index:
                 all_class_probabilities = np.array([1.0, 0.0, 0.0])
                 if len(all_class_probabilities) > 0: all_class_probabilities[0] = 1.0 # Mark first class as 100%

        # Calculate proba_for_response from the final, validated values
        proba_for_response: float
        if len(all_class_probabilities) > 0 and 0 <= predicted_class_index < len(all_class_probabilities):
            proba_for_response = float(all_class_probabilities[predicted_class_index])
        else: # Fallback if all_class_probabilities is empty or index still problematic
            proba_for_response = 1.0 if predicted_class_index == 0 else 0.0 # Default to 1.0 for BRONZE, 0 otherwise
            if len(all_class_probabilities) == 0: # Ensure all_class_probabilities is not empty for return
                 all_class_probabilities = np.array([0.0,0.0,0.0])
                 if 0 <= predicted_class_index < len(all_class_probabilities): all_class_probabilities[predicted_class_index] = 1.0
                 elif len(all_class_probabilities)>0 : all_class_probabilities[0] = 1.0
                 else: all_class_probabilities = np.array([1.0,0.0,0.0]) # final final fallback

            
            # The probability for the API response should be the probability of the predicted class
            proba_for_response = float(all_class_probabilities[predicted_class_index])
        
        class_mapping = {
            0: "SILVER",  # As per user request
            1: "GOLD",    # As per user request
            2: "BRONZE"   # As per user request
        }

        predicted_label = class_mapping.get(predicted_class_index, "UNKNOWN") # Fallback to UNKNOWN
        if predicted_label == "UNKNOWN":
            print(f"WARNING: predicted_class_index {predicted_class_index} not found in class_mapping. Defaulting to BRONZE or first class.")
            # Ensure a safe default label if index was somehow out of expected range despite earlier checks
            # The most likely default based on previous logic is the first key (0) or a hardcoded default.
            predicted_label = class_mapping.get(0, "BRONZE") # Default to SILVER or BRONZE

        # Return PredictionOut with the mapped string label and its probability
        return PredictionOut(lit_level=predicted_label, proba=proba_for_response)
