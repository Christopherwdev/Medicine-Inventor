import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
from networks import DrugEfficacyModel
from data_utils import load_and_preprocess_data
import os

# Create model save directory if it doesn't exist
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def train():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_PATH)
    input_dim = X_train.shape[1]
    hidden_dim = 128  # Adjust as needed
    output_dim = 1  # Binary classification

    model = DrugEfficacyModel(input_dim, hidden_dim, output_dim).to(DEVICE)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(DEVICE) #Reshape y for BCELoss

    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'drug_efficacy_model.pth'))
    print(f'Model saved to {os.path.join(MODEL_SAVE_PATH, "drug_efficacy_model.pth")}')

if __name__ == "__main__":
    train()
