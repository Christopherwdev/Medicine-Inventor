# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
DATA_PATH = "drug_data.csv"  # Replace with your data path
MODEL_SAVE_PATH = "models"

# Other configurations (add as needed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
