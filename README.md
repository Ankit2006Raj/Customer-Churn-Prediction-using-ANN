# Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave) using an Artificial Neural Network built with TensorFlow/Keras.

## Overview

The model is trained on a dataset of 10,000 bank customers and learns patterns from features like credit score, age, balance, geography, and activity status to predict churn probability.

## Project Structure

```
├── churn_ann.py                          # Main script
├── churn_ann_results.png                 # Output visualizations
└── data/
    └── Artificial_Neural_Network_Case_Study_data.csv
```

## Model Architecture

- Input layer → Dense(64, ReLU) + BatchNorm + Dropout(0.3)
- Hidden → Dense(32, ReLU) + BatchNorm + Dropout(0.2)
- Hidden → Dense(16, ReLU) + Dropout(0.1)
- Output → Dense(1, Sigmoid)
- Optimizer: Adam | Loss: Binary Crossentropy

## Results

The model outputs:
- Test Accuracy & ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Prediction Probability Distribution
- Churn Rate by Geography

![Results](churn_ann_results.png)

## Tech Stack

- Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## Installation & Usage

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python churn_ann.py
```

## Connect with Me

- GitHub: [Ankit2006Raj](https://github.com/Ankit2006Raj)
- LinkedIn: [Ankit Raj](https://www.linkedin.com/in/ankit-raj-226a36309)
- Email: ankit9905163014@gmail.com
