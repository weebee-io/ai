import joblib
import numpy as np
import os

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
    settings = get_settings()
    model_path = settings.MODEL_PATH
    
    # MODEL_PATH가 상대 경로일 수 있으므로, 설정 파일의 위치를 기준으로 절대 경로를 만듭니다.
    # config.py는 app/core/config.py 이므로, settings.BASE_DIR 같은 것이 있다면 사용하면 좋지만,
    # 없다면 get_settings()가 반환하는 settings 객체에 MODEL_PATH가 이미 절대경로이거나,
    # joblib.load가 처리할 수 있는 경로이길 기대합니다.
    # 일반적으로 MODEL_PATH는 프로젝트 루트로부터의 상대 경로로 저장됩니다.
    # finance-segmentation-api 가 프로젝트 루트라고 가정합니다.
    
    # 만약 MODEL_PATH가 settings.PROJECT_ROOT 같은 것을 기준으로 한다면, 그에 맞게 조정해야 합니다.
    # 현재는 get_settings().MODEL_PATH가 유효한 경로라고 가정합니다.
    # 해당 경로가 상대경로라면, 이 스크립트의 실행 위치에 따라 달라질 수 있습니다.
    # ml_model.py에서는 이 경로를 바로 사용하므로, 동일하게 동작할 것으로 예상합니다.

    print(f"Loading model from: {model_path}")
    
    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print(f"Current working directory: {os.getcwd()}")
        # PROJECT_ROOT를 기준으로 경로를 재시도해볼 수 있습니다.
        # 예: model_path = os.path.join(project_root, settings.MODEL_PATH)
        # bundle = joblib.load(model_path)
        return

    centroids = bundle.get('centroids')
    weights = bundle.get('weights')
    
    # 이전 메모리(MEMORY[31ef0c6d...])에서 스케일러가 기대하는 특성 순서
    feature_names = ['essential_pct', 'discretionary_pct', 'risk_profile_score', 
                     'complex_product_flag', 'digital_engagement', 'is_married', 
                     'spend_volatility', 'sav_inv_ratio']
    
    # 이전 메모리(MEMORY[eb6d10fe...])에서 클래스 매핑
    class_mapping_indices_to_names = {0: 'SILVER', 1: 'GOLD', 2: 'BRONZE'}

    print("\nFeature Names (in order):")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")

    print("\nCentroids (Segment Centers):")
    if centroids is not None:
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
        if len(weights) == len(feature_names):
            for i, weight in enumerate(weights):
                print(f"  {feature_names[i]}: {weight:.4f}")
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
            
        # 예시: 모든 원본 입력이 0일 때 스케일링된 값 계산
        print("\n  Example: Scaled values if all original inputs are 0:")
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and scaler.mean_ is not None and scaler.scale_ is not None:
            if len(scaler.mean_) == len(feature_names) and len(scaler.scale_) == len(feature_names):
                # Ensure feature_names align with scaler.mean_ and scaler.scale_ if their order is fixed by the scaler
                # For now, assuming the order in feature_names is consistent with how scaler was fit
                scaled_zeros = (0 - np.array(scaler.mean_)) / np.array(scaler.scale_)
                for i, sz_val in enumerate(scaled_zeros):
                    print(f"    {feature_names[i]}: {sz_val:.4f}")
            else:
                print("    Cannot calculate scaled zeros due to length mismatch of mean/scale and feature_names.")
        else:
            print("    Cannot calculate scaled zeros, mean_ or scale_ missing.")
    else:
        print("Scaler not found in model bundle.")

if __name__ == "__main__":
    inspect_components()
