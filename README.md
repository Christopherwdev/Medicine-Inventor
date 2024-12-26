# Medicine-Inventor: Drug Efficacy Prediction using a Neural Network

This repository contains a simple neural network model for predicting the efficacy of drugs based on their molecular features.  This is a foundational example and can be expanded upon for more complex scenarios.

## Overview

The model uses a supervised learning approach.  It takes as input a set of molecular features representing a drug and outputs a probability score indicating the likelihood of the drug being effective.  The model is trained using a binary cross-entropy loss function.

## Dependencies

* Python 3.7+
* PyTorch
* scikit-learn
* pandas
* numpy
* tqdm

Install the required packages using pip:

```bash
pip install torch scikit-learn pandas numpy tqdm
```

## Data
The model requires a CSV file containing the drug features and efficacy labels. The expected format is:

Feature 1	Feature 2	...	Efficacy
...	...	...	...
The Efficacy column should contain binary values (0 or 1) representing ineffective and effective drugs, respectively. You'll need to prepare your own dataset or obtain one from a public repository. Make sure your data is preprocessed appropriately (e.g., scaling, handling missing values).
Place your data CSV file in the root directory of the project and update the DATA_PATH variable in config.py accordingly.

## Usage
Data Preparation: Prepare your dataset in the format described above.
Training: Run the training script:

```bash
python train.py
```

This will train the model and save it to the models directory.
Evaluation: Run the evaluation script:

```bash
python evaluate.py
```

This will load the trained model and evaluate its performance on a test set. The accuracy and AUC scores will be printed to the console.

## File Structure

Copy
Medicine-Inventor/
├── config.py          # Hyperparameters and file paths
├── networks.py        # Neural network architecture
├── data_utils.py      # Data loading and preprocessing functions
├── train.py           # Training script
└── evaluate.py        # Evaluation script
└── models/            # Directory to save trained models (created during training)
└── README.md          # This file

## Model Architecture
The model uses a simple feedforward neural network with two hidden layers and a sigmoid activation function in the output layer to produce a probability score. The architecture can be modified in networks.py.

## Further Development
This is a basic example. Consider these improvements:

- More sophisticated architectures: Explore more complex neural network architectures (e.g., convolutional neural networks for image-based features, recurrent neural networks for sequential data).
- Hyperparameter tuning: Optimize hyperparameters such as learning rate, batch size, and number of layers using techniques like grid search or random search.
- Feature engineering: Carefully engineer relevant features from your data to improve model performance.
- Advanced evaluation metrics: Use more comprehensive evaluation metrics beyond accuracy and AUC.

This project provides a starting point for building more advanced drug efficacy prediction models. Remember to adapt and expand upon this foundation based on your specific needs and dataset.

You can learn more about my coding projects here: [https://sites.google.com/view/wong-kin-on-christopher/computer-science](https://sites.google.com/view/wong-kin-on-christopher/computer-science)

