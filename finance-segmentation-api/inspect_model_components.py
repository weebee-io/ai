import joblib
import numpy as np
import os
import mlflow
import json # For logging dicts as params

# finance-segmentation-api 디렉토리로 경로를 맞춰야 합니다.
# 이 스크립트가 finance-segmentation-api 내부에 있다고 가정합니다.
# 그렇지 않다면 sys.path를 조정해야 할 수 있습니다.
try:
    from app.core.config import get_settings
except ImportError:
    # 스크립트가 프로젝트 루트에 있고 'app'이 하위 디렉토리인 경우 PYTHONPATH를 임시로 조정
    import sys
    # 현재 스크립트의 디렉토리를 기준으로 finance-segmentation-api 디렉토리를 찾습니다.
    # 이 스크립트는 /Users/henry/Desktop/final-project/ai/finance-segmentation-api/ 에 위치할 것입니다.
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root) # finance-segmentation-api 디렉토리를 sys.path에 추가
    from app.core.config import get_settings


def inspect_components():
    mlflow.set_experiment("Model Inspection") # Sets the active experiment

    with mlflow.start_run(run_name="Inspect Centroid Model Components"):
        settings = get_settings()
        model_path_from_config = settings.MODEL_PATH
        
        # 스크립트 파일 위치를 기준으로 한 current_script_dir를 사용합니다.
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(model_path_from_config):
            model_path = os.path.join(current_script_dir, model_path_from_config)
            print(f"Constructed absolute model path: {model_path}")
        else:
            model_path = model_path_from_config

        mlflow.log_param("original_model_path_config", model_path_from_config)
        mlflow.log_param("effective_model_path", model_path)

        print(f"Loading model from: {model_path}")
        
        try:
            bundle = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}.")
            print(f"Current working directory: {os.getcwd()}")
            return # mlflow.start_run() 컨텍스트 내에서 return하면 run이 종료됨
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return

        centroids = bundle.get('centroids')
        weights = bundle.get('weights')
        
        feature_names = ['essential_pct', 'discretionary_pct', 'risk_profile_score', 
                         'complex_product_flag', 'digital_engagement', 'is_married', 
                         'spend_volatility', 'sav_inv_ratio']
        mlflow.log_param("feature_names_ordered", ", ".join(feature_names))
        mlflow.log_metric("num_features", len(feature_names))
        
        class_mapping_indices_to_names = {0: 'SILVER', 1: 'GOLD', 2: 'BRONZE'}
        mlflow.log_param("class_mapping", json.dumps(class_mapping_indices_to_names))

        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")
            print(f"Logged model artifact: {model_path} to 'model' directory in MLflow run.")
        else:
            # 이 경우는 joblib.load에서 이미 FileNotFoundError로 처리되었을 가능성이 높습니다.
            print(f"Could not log model artifact. Path not found: {model_path}")

        print("\nFeature Names (in order):")
        for i, name in enumerate(feature_names):
            print(f"{i}: {name}")

        print("\nCentroids (Segment Centers):")
        if centroids is not None:
            mlflow.log_metric("num_centroids", len(centroids))
            for i, centroid_values in enumerate(centroids):
                segment_name = class_mapping_indices_to_names.get(i, f"Unknown_Segment_{i}")
                print(f"\n--- {segment_name} (Centroid {i}) ---")
                if len(centroid_values) == len(feature_names):
                    for feature_idx, value in enumerate(centroid_values):
                        print(f"  {feature_names[feature_idx]}: {value:.4f}")
                else:
                    print(f"  Raw values: {centroid_values}")
                    print(f"  Warning: Number of centroid values ({len(centroid_values)}) does not match number of feature names ({len(feature_names)}).")
        else:
            print("Centroids not found in model bundle.")

        print("\nFeature Weights:")
        if weights is not None:
            mlflow.log_metric("num_weights", len(weights))
            if len(weights) == len(feature_names):
                for i, weight_val in enumerate(weights): # 변수명 변경: weight -> weight_val
                    print(f"  {feature_names[i]}: {weight_val:.4f}")
            else:
                print(f"  Raw values: {weights}")
                print(f"  Warning: Number of weight values ({len(weights)}) does not match number of feature names ({len(feature_names)}).")
        else:
            print("Weights not found in model bundle.")

        scaler = bundle.get('scaler')

        print("\nScaler Information:")
        if scaler is not None:
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                print("  Mean values (scaler.mean_):")
                if len(scaler.mean_) == len(feature_names):
                    for i, mean_val in enumerate(scaler.mean_):
                        print(f"    {feature_names[i]}: {mean_val:.4f}")
                else:
                    print(f"    Raw values: {scaler.mean_}")
            else:
                print("  Scaler mean_ not found or is None.")
                
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                print("  Scale values (scaler.scale_ - standard deviation):")
                if len(scaler.scale_) == len(feature_names):
                    for i, scale_val in enumerate(scaler.scale_):
                        print(f"    {feature_names[i]}: {scale_val:.4f}")
                else:
                    print(f"    Raw values: {scaler.scale_}")
            else:
                print("  Scaler scale_ not found or is None.")
                
            print("\n  Example: Scaled values if all original inputs are 0:")
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and scaler.mean_ is not None and scaler.scale_ is not None:
                if len(scaler.mean_) == len(feature_names) and len(scaler.scale_) == len(feature_names):
                    scaled_zeros = (0 - np.array(scaler.mean_)) / np.array(scaler.scale_)
                    for i, sz_val in enumerate(scaled_zeros):
                        print(f"    {feature_names[i]}: {sz_val:.4f}")
                else:
                    print("    Cannot calculate scaled zeros due to length mismatch of mean/scale and feature_names.")
            else:
                print("    Cannot calculate scaled zeros, mean_ or scale_ missing.")
        else:
            print("Scaler not found in model bundle.")

        # 모든 로깅 및 작업이 끝난 후 with 블록이 여기서 종료됩니다.

if __name__ == "__main__":
    inspect_components()
