import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from config import *
from networks import DrugEfficacyModel
from data_utils import load_and_preprocess_data

def evaluate():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_PATH)
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = DrugEfficacyModel(input_dim, hidden_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'drug_efficacy_model.pth')))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    # Convert probabilities to binary predictions (threshold of 0.5)
    binary_predictions = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, binary_predictions)
    auc = roc_auc_score(y_test, predictions) #AUC is better than accuracy for imbalanced datasets

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    evaluate()
