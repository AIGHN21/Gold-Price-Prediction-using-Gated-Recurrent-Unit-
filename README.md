# Financial Time Series Forecasting with GRU + Attention

## Overview

This project implements a deep learning pipeline for financial time series forecasting using a hybrid **GRU + Attention** architecture.

Instead of directly predicting the next closing price, the model learns the **percentage change (return)** between consecutive days:

[
pct_t = \frac{Close_{t+1} - Close_t}{Close_t}
]

The predicted percentage is then transformed back into the predicted closing price:

[
PredictedClose = BaseClose \times (1 + PredictedPct)
]

This formulation stabilizes the learning process and allows the network to focus on market movement dynamics instead of absolute price scales.

---

# Dataset Structure

The dataset contains historical OHLCV market data.

## Input Features

For each trading day:

* Open
* High
* Low
* Close
* Volume

Additional engineered features include:

* Daily percentage return
* Rolling statistics
* Price differences
* Normalized sequences

---

# Feature Extraction


## Percentage Change Label

The target variable is defined as:

$$
y_t = \frac{Close_{t+1} - Close_t}{Close_t}
$$

This transforms raw prices into stationary relative movements.



---

## Context Window (Lookback)

The model uses a sliding context window of:

* **Lookback = 7 days**

Each training sample contains sequential information from the previous 7 trading days.

Example:

| t-6 | t-5 | t-4 | t-3 | t-2 | t-1 | t |
| --- | --- | --- | --- | --- | --- | - |

The network predicts the percentage movement for:

[
t+1
]

---

# Model Architecture

## GRU + Attention Network

The forecasting model combines:

* GRU recurrent layers
* Temporal attention mechanism
* Dense regression head

---

## Architecture Flow

```text
Input Sequence (7 days)
        ↓
Feature Embedding
        ↓
GRU Layer
        ↓
Attention Layer
        ↓
Context Vector
        ↓
Fully Connected Layer
        ↓
Predicted Percentage Change
```

---

# Why GRU?

GRU (Gated Recurrent Unit) is used because:

* Lower computational cost than LSTM
* Faster convergence
* Strong performance on sequential financial data
* Better memory efficiency

---

# Why Attention?

Attention allows the model to:

* Focus on important historical timesteps
* Learn dynamic temporal weighting
* Capture short-term market momentum
* Improve interpretability of sequence learning

Instead of treating all timesteps equally, the network learns which previous days are most relevant for predicting the next movement.

---

# Training Configuration

## Hardware

Training was performed using:

* **2× NVIDIA Tesla T4 GPUs**

---

## Frameworks

* PyTorch
* NumPy
* Pandas
* Scikit-learn

---

# Prediction Pipeline

## Step 1 — Predict Percentage Change

The model outputs:

[
\hat{pct}
]

---

## Step 2 — Convert Back to Price

Using the latest real closing price:

[
PredictedClose = BaseClose \times (1 + \hat{pct})
]

---

# Sample Prediction Results

| Date       | BaseClose | TruePct   | PredPct   | TrueClose | PredClose |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| 2022-09-15 | 1697.36   | -0.019324 | -0.001748 | 1664.56   | 1694.39   |
| 2022-09-16 | 1664.56   | 0.006278  | -0.001334 | 1675.01   | 1662.34   |
| 2022-09-19 | 1675.01   | 0.000418  | -0.000485 | 1675.71   | 1674.20   |
| 2022-09-20 | 1675.71   | -0.006535 | -0.000771 | 1664.76   | 1674.42   |
| 2022-09-21 | 1664.76   | 0.005388  | 0.000224  | 1673.73   | 1665.13   |
| 2022-09-22 | 1673.73   | -0.001535 | 0.000434  | 1671.16   | 1674.46   |
| 2022-09-23 | 1671.16   | -0.015732 | 0.000364  | 1644.87   | 1671.77   |
| 2022-09-26 | 1644.87   | -0.013661 | -0.000704 | 1622.40   | 1643.71   |
| 2022-09-27 | 1622.40   | 0.004167  | -0.002009 | 1629.16   | 1619.14   |
| 2022-09-28 | 1629.16   | 0.018856  | -0.001265 | 1659.88   | 1627.10   |

---

# Key Observations

* The model successfully learns directional movement patterns.
* Percentage-based learning stabilizes training compared to raw-price prediction.
* Attention improves temporal sensitivity over pure recurrent models.
* GRU provides efficient sequence modeling with reduced memory overhead.

---

# Future Improvements

Potential future upgrades:

* Multi-head attention
* Transformer encoder blocks
* Probabilistic forecasting
* Using Hidden Markov Model to define the hidden regime 

---

# Conclusion

This project demonstrates a sequence forecasting approach for financial markets using:

* Percentage-return learning
* Sliding temporal windows
* GRU recurrent modeling
* Attention-based temporal weighting

