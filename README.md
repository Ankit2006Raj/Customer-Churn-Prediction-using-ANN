# Customer Churn Prediction using ANN

Predicts whether a bank customer will churn using an Artificial Neural Network built with TensorFlow/Keras on a dataset of 10,000 customers.

## Project Structure

```
├── churn_ann.py                                        # Main script
├── churn_ann_results.png                               # Output visualizations
├── Image/                                              # Result screenshots
└── data/
    └── Artificial_Neural_Network_Case_Study_data.csv
```

## Model Architecture

| Layer | Details |
|-------|---------|
| Input + Hidden 1 | Dense(64, ReLU) → BatchNorm → Dropout(0.3) |
| Hidden 2 | Dense(32, ReLU) → BatchNorm → Dropout(0.2) |
| Hidden 3 | Dense(16, ReLU) → Dropout(0.1) |
| Output | Dense(1, Sigmoid) |

- Optimizer: Adam | Loss: Binary Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau

## Results

![Results](churn_ann_results.png)
<img width="1290" height="1184" alt="image" src="https://github.com/user-attachments/assets/88603ab4-89ee-4037-bc5f-9a81824dc4e9" />
<img width="1094" height="888" alt="image" src="https://github.com/user-attachments/assets/6783aca0-aa0b-4b41-9c29-dac9351a496d" />


## Tech Stack

Python · TensorFlow/Keras · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## Installation & Usage

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python churn_ann.py
```

## Connect with Me

- GitHub: [Ankit2006Raj](https://github.com/Ankit2006Raj)
- LinkedIn: [Ankit Raj](https://www.linkedin.com/in/ankit-raj-226a36309)
- Email: ankit9905163014@gmail.com
