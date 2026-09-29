# Traffic Patterns Forecasting — Time Series Forecasting

A complete machine-learning project for forecasting traffic patterns using the **PEMS-BAY traffic-speed benchmark**. The project combines exploratory data analysis, time-series feature engineering, a persistence baseline, Random Forest regression, and an LSTM neural network to forecast traffic speed.

## Project Overview

Traffic congestion changes over time and often follows recurring patterns based on:

- Time of day
- Day of week
- Recent traffic conditions
- Historical traffic at similar times
- Short-term and long-term temporal dependencies

This project models these patterns and generates a **one-hour traffic forecast at 5-minute intervals**.

## Dataset

The notebook uses the public **PEMS-BAY** benchmark.

- 325 traffic sensors
- 5-minute sampling interval
- Traffic-speed observations
- Historical time-series data
- Designed for traffic forecasting research

> **Important:** PEMS-BAY is historical benchmark data, not a live traffic feed. The notebook downloads the public dataset automatically.

## Project Structure

```text
Traffic_Patterns_Forecasting/
│
├── Traffic_Patterns_Forecasting_PEMS_BAY.ipynb
├── README.md
└── Traffic_Patterns_Forecasting_Presentation.pptx
```

## Main Objectives

1. Load and clean traffic sensor data.
2. Analyze recurring traffic patterns.
3. Create time-series lag and rolling features.
4. Preserve chronological order during model evaluation.
5. Establish a persistence forecasting baseline.
6. Train a Random Forest forecasting model.
7. Train an LSTM sequence model.
8. Compare models using MAE, RMSE, and R².
9. Generate a one-hour future traffic forecast.
10. Visualize historical traffic and predicted traffic.

## Technologies Used

- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Requests

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow requests jupyter
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Traffic_Patterns_Forecasting_PEMS_BAY.ipynb
```

## Data Download

The notebook automatically downloads the PEMS-BAY CSV into:

```text
traffic_data/PEMS-BAY.csv
```

If the file already exists, the notebook reuses it instead of downloading it again.

## Workflow

### 1. Data Collection

The project downloads the public PEMS-BAY benchmark and loads it into a Pandas DataFrame.

### 2. Data Cleaning

The notebook:

- Converts timestamps to datetime format
- Sorts observations chronologically
- Uses interpolation for missing observations
- Uses forward/backward filling where necessary

### 3. Exploratory Data Analysis

The project visualizes:

- Full traffic history
- Recent seven-day traffic patterns
- Average traffic speed by hour
- Average traffic speed by weekday

These plots help identify recurring daily and weekly patterns.

### 4. Feature Engineering

The Random Forest model uses temporal features including:

```text
lag_1
lag_3
lag_6
lag_12
lag_24
lag_288
lag_2016
```

These represent approximately:

| Feature | Meaning |
|---|---|
| lag_1 | Previous 5 minutes |
| lag_3 | Previous 15 minutes |
| lag_6 | Previous 30 minutes |
| lag_12 | Previous 1 hour |
| lag_24 | Previous 2 hours |
| lag_288 | Previous 1 day |
| lag_2016 | Previous 1 week |

Additional features include:

- Hour
- Minute
- Day of week
- Day of year
- Weekend indicator
- Cyclic hour features
- Cyclic weekday features
- 1-hour rolling mean
- 1-hour rolling standard deviation
- 3-hour rolling mean

## Train / Validation / Test Split

The project uses a chronological split:

```text
70% → Training
15% → Validation
15% → Testing
```

A random split is avoided because future observations must not leak into the training data.

## Models

### Persistence Baseline

The simplest model predicts that the next traffic value will be equal to the most recent observed value.

This gives a reference point for judging machine-learning models.

### Random Forest

A Random Forest Regressor learns relationships between:

- Recent traffic values
- Historical traffic values
- Calendar features
- Rolling statistics

Example configuration:

```python
RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

### LSTM

The LSTM model receives a **24-hour history window**:

```text
288 observations × 5 minutes = 24 hours
```

Architecture:

```text
Input
  ↓
LSTM(64)
  ↓
Dropout
  ↓
LSTM(32)
  ↓
Dropout
  ↓
Dense(16)
  ↓
Dense(1)
```

Early stopping is used to reduce unnecessary training.

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted traffic values.

Lower MAE indicates smaller average prediction errors.

### RMSE — Root Mean Squared Error

Penalizes larger errors more strongly than MAE.

Lower RMSE indicates better predictive accuracy.

### R² — Coefficient of Determination

Measures how much variation in the target is explained by the model.

A value closer to 1 indicates stronger explanatory performance on the evaluated data.

## Forecasting

The final LSTM forecast generates:

```text
12 predictions × 5 minutes
= 60 minutes
= 1 hour
```

The notebook produces a table containing:

```text
timestamp
forecast_speed
```

It also plots the historical traffic series together with the future forecast.

## Example Output

The project produces:

- Traffic history plots
- Recent traffic trend plots
- Feature importance chart
- LSTM training/validation loss
- Model comparison
- Actual vs predicted traffic
- Hourly traffic pattern
- Weekday traffic pattern
- One-hour future forecast

## Why Time-Series Validation Matters

Traditional random train/test splitting can cause temporal leakage.

For example, if observations from Monday afternoon appear in training while nearby future observations appear in the test set, the model may indirectly gain information about the future.

This project therefore maintains chronological order.

## Limitations

This is a research/academic forecasting project and has several limitations:

1. PEMS-BAY is historical rather than real-time.
2. Only one selected sensor is modeled by the notebook at a time.
3. Weather information is not included.
4. Holidays and special events are not explicitly modeled.
5. The LSTM forecast uses recursive prediction for the one-hour horizon.
6. Traffic speed is forecast rather than directly predicting congestion categories.
7. Real deployment would require continuous data ingestion and model monitoring.

## Possible Future Improvements

### Data

- Integrate a live traffic API
- Add weather data
- Add public-holiday calendars
- Add road incidents
- Add traffic volume and occupancy

### Models

- XGBoost
- LightGBM
- GRU
- Bidirectional LSTM
- Temporal Convolutional Networks
- Transformers
- Temporal Fusion Transformer

### Advanced Traffic Forecasting

The complete PEMS-BAY sensor network can be modeled jointly using:

- Graph Neural Networks
- Graph Convolutional Networks
- Spatio-temporal GNNs
- DCRNN-style architectures

These methods can model relationships between neighboring traffic sensors rather than treating every sensor independently.

## Project Applications

Traffic forecasting can support:

- Intelligent transportation systems
- Route planning
- Congestion monitoring
- Traffic signal optimization
- Travel-time estimation
- Urban mobility planning
- Fleet management
- Smart-city applications

## Reproducibility

The Random Forest uses:

```python
random_state=42
```

The dataset is processed chronologically and the scaler for the LSTM is fitted only using training observations.

## References

- Li, Y., Yu, R., Shahabi, C., & Liu, Y. — *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018.
- PEMS-BAY public traffic forecasting benchmark.
- Zenodo public dataset archive.
- Hugging Face processed PEMS-BAY benchmark resources.

## Author

**Traffic Patterns Forecasting — Time Series Forecasting Project**

Built with Python, Pandas, Scikit-learn and TensorFlow/Keras.
