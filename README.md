# Regression on Tabular Data - General Machine Learning

## Overview
This project focuses on performing regression analysis on tabular data using random forest regression. The goal is to predict a continuous target variable based on the input features provided in the dataset.

## Directory Structure
```
2-Regression_on_the_tabular_data-General_Machine_Learning/
│
├── data/
│   ├── hidden_test.csv
│   └── train.csv
├── notebooks/
│   └── data_analysis.ipynb
├── results/
│   ├── hypertuning_results.txt
│   ├── imputer_model.pkl
│   ├── predictions.csv
│   └── random_forest_model.pkl
├── src/
│   ├── hypertune.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Getting Started
### Prerequisites
- Python 3.8
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

### Installation
1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory:
    ```bash
    cd 2-Regression_on_the_tabular_data-General_Machine_Learning
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Model hypertune (optional)
Run the model hypertune script to find best model parameters:
```bash
python src/hypertune.py
```

### Model Training
Train the regression model using the training dataset:
```bash
python src/train.py
```

### Model Evaluation
Evaluate the trained model on the test dataset:
```bash
python src/predict.py
```

## Notebooks
- `data_analysis.ipynb`: Jupyter notebook for data analysis