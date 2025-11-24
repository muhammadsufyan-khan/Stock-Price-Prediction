# 📈 Stock Price Prediction 

## ✅ Task Objective
The goal of this task is to predict the **next day's closing price** of a selected stock using historical stock data.  
Short-term stock price prediction is a fundamental problem in financial data analysis and requires understanding patterns, trends, and relationships between market indicators such as Open, High, Low, and Volume.  

## 📂 Dataset Used
- **Source:** Yahoo Finance via Python `yfinance` library  
- **Stock Examples:**  
  - Apple Inc. (AAPL)  
  - Tesla Inc. (TSLA)  
  - Microsoft Corp. (MSFT)  
- **Time Period:** User-defined, for example `2023-01-01` to `2025-11-01`  
- **Features:**
  - `Open` (numeric): Opening price of the stock on that day  
  - `High` (numeric): Highest price reached during the day  
  - `Low` (numeric): Lowest price reached during the day  
  - `Volume` (numeric): Number of shares traded  
- **Target:**
  - `Next_Close` (numeric): Closing price of the next trading day  

> Note: The dataset is dynamically fetched using the `yfinance` API, so no separate CSV is required.

## 📂 GitHub Repo Structure
```
StockPricePrediction/
│
├── README.md
├── stock_price_prediction.ipynb      # Jupyter Notebook (single-cell complete)
├── stock_price_prediction.py         # Python script version
```


## 🧪 Models Applied
Two types of regression models were implemented:

1. **Linear Regression** (baseline)
   - Captures linear relationships between features and target.
   - Quick to train but may underfit for non-linear stock trends.

2. **Random Forest Regressor** (primary model)
   - Ensemble of decision trees for capturing complex non-linear relationships.
   - Robust to overfitting and can handle feature interactions effectively.
   - Generally provides better accuracy for stock price prediction than linear models.

> Users can switch between models by commenting/uncommenting the relevant lines in the code.


## 📊 Workflow and Key Steps
1. **Load Historical Stock Data**
   - Use the `yfinance` library to download stock data including Open, High, Low, Close, and Volume.
   - Inspect the data to ensure no missing values and correct time order.

2. **Feature Selection and Target Creation**
   - Select features: `Open`, `High`, `Low`, `Volume`.
   - Target variable: `Next_Close` (next day's closing price), created by shifting the `Close` column by one day.

3. **Train-Test Split**
   - Split the data into training (80%) and testing (20%) sets.
   - Shuffle is set to False to preserve chronological order for time series data.

4. **Model Training**
   - Train the regression model (Linear Regression or Random Forest) using training data.
   - Fit the model on features and target.

5. **Predictions**
   - Predict the next day's closing price on the test set.
   - Compare predicted values with actual closing prices.

6. **Evaluation Metrics**
   - **Mean Absolute Error (MAE)**: Measures average magnitude of errors.
   - **Root Mean Squared Error (RMSE)**: Measures spread of prediction errors.
   - These metrics help understand model accuracy and reliability.

7. **Visualization**
   - Line plot comparing actual vs predicted close prices over time.
   - Scatter plot showing relationship between actual and predicted prices.
   - Visual insights help identify trends and model performance consistency.


## 📈 Key Results
- **MAE:** ~ <img width="338" height="197" alt="image" src="https://github.com/user-attachments/assets/339610be-babb-4d32-ae44-b429a35924ec" />
 
- **RMSE:** ~ <img width="327" height="171" alt="image" src="https://github.com/user-attachments/assets/c96b8494-0fcd-482a-a30b-8a5d10d89e9a" />


### Observations:
- Random Forest Regressor generally performs better than Linear Regression for volatile stocks.
- Predictions closely follow the actual closing prices in stable periods, but larger errors can occur during sudden market swings.
- Scatter plots show that predicted values cluster around actual values, indicating the model captures most trends effectively.


## 📌 Insights
- Stock price is influenced by multiple factors beyond Open, High, Low, and Volume. Including technical indicators (e.g., moving averages, RSI, MACD) can improve predictions.
- Random Forest captures non-linear relationships between features, making it more effective for short-term price prediction.
- This approach is suitable for **short-term forecasting**, not long-term investment advice.
- Prediction errors are generally low, indicating the model’s potential for day-to-day trend analysis.


## ▶️ How to Run
1. Open `stock_price_prediction.ipynb` in **Google Colab** or **Jupyter Notebook**.  
2. Run all cells — the script will:
   - Download historical stock data from Yahoo Finance  
   - Create the next-day closing price target  
   - Train the regression model  
   - Predict prices and evaluate performance  
   - Plot actual vs predicted closing prices  

3. Alternatively, run the Python script version:
```bash
python stock_price_prediction.py
```
4. To test another stock, simply change the stock_symbol variable in the notebook or script.

## 🤝 Future Improvements

Include technical indicators like SMA, EMA, RSI, and MACD as features.

Hyperparameter tuning for Random Forest (number of trees, max depth, etc.).

Use more advanced models: XGBoost, LightGBM, or LSTM (for deep learning time-series prediction).

Extend prediction to multi-day forecasting instead of next-day only.

## 📜 License

MIT License
