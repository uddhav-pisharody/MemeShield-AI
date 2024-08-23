import os
import torch
from src.utils.data_loader import get_data_loaders
from src.models.bert_vit import BertViTModel
from src.utils.training import train_model, evaluate_model
from src.utils.feature_extraction import extract_features
from src.utils.feature_selection import apply_feature_selection
from src.utils.classifiers import train_classifiers, evaluate_classifiers
from src.config import Config

def main():
    # Load Configuration
    config = Config()

    # Step 1: Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Step 2: Initialize Model
    print("Initializing model...")
    model = BertViTModel(config)

    # Step 3: Train Model
    if not os.path.exists(config.saved_model_path):
        print("Training model...")
        train_model(model, train_loader, val_loader, config)
        torch.save(model.state_dict(), config.saved_model_path)
    else:
        print("Loading saved model...")
        model.load_state_dict(torch.load(config.saved_model_path))

    # Step 4: Extract Features
    print("Extracting features...")
    feature_map = extract_features(model, test_loader, config)
    
    # Save extracted features to CSV
    feature_csv_path = os.path.join(config.features_dir, "features.csv")
    feature_map.to_csv(feature_csv_path, index=False)
    print(f"Features saved to {feature_csv_path}")

    # Step 5: Apply Feature Selection
    print("Applying feature selection...")
    selected_features_ga, selected_features_fa, selected_features_pso = apply_feature_selection(feature_csv_path, config)

    # Save the selected features
    ga_features_path = os.path.join(config.features_dir, "selected_features_ga.csv")
    fa_features_path = os.path.join(config.features_dir, "selected_features_fa.csv")
    pso_features_path = os.path.join(config.features_dir, "selected_features_pso.csv")
    
    selected_features_ga.to_csv(ga_features_path, index=False)
    selected_features_fa.to_csv(fa_features_path, index=False)
    selected_features_pso.to_csv(pso_features_path, index=False)
    
    print(f"Selected features saved to {ga_features_path}, {fa_features_path}, and {pso_features_path}")

    # Step 6: Train Machine Learning Classifiers
    print("Training classifiers...")
    classifiers_ga = train_classifiers(selected_features_ga, config)
    classifiers_fa = train_classifiers(selected_features_fa, config)
    classifiers_pso = train_classifiers(selected_features_pso, config)

    # Step 7: Evaluate Classifiers
    print("Evaluating classifiers...")
    evaluate_classifiers(classifiers_ga, config)
    evaluate_classifiers(classifiers_fa, config)
    evaluate_classifiers(classifiers_pso, config)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()