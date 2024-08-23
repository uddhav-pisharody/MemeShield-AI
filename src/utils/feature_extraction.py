import torch
import pandas as pd
from torch.utils.data import DataLoader

def extract_features(model, data_loader: DataLoader, config):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input'].to(config.device)
            labels.append(batch['label'].cpu().numpy())
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    label_df = pd.DataFrame(labels, columns=['label'])
    feature_df = pd.concat([feature_df, label_df], axis=1)
    
    return feature_df
