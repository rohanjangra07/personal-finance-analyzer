import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_future_expenses(df, future_days=7):
    """
    Predicts future expenses based on daily aggregate spending using RandomForestRegressor.
    """
    if df.empty or len(df) < 3:
        return pd.DataFrame() # Not enough data for meaningful prediction
        
    # Group by Date
    daily_spend = df.groupby('Date')['Amount'].sum().reset_index().sort_values('Date')
    
    if len(daily_spend) < 7:
        print("⚠️ Low data, predictions may be unreliable")
        
    if len(daily_spend) < 3:
        return pd.DataFrame()
        
    # Prepare features
    daily_spend['DateOrdinal'] = daily_spend['Date'].apply(lambda x: x.toordinal())
    daily_spend['DayOfWeek'] = daily_spend['Date'].dt.dayofweek
    daily_spend['Month'] = daily_spend['Date'].dt.month
    
    # Rolling average for smoothing
    daily_spend['RollingAvg'] = daily_spend['Amount'].rolling(3).mean()
    
    X = daily_spend[['DateOrdinal', 'DayOfWeek', 'Month']].values
    y = daily_spend['Amount'].values
    
    # Train advanced ensemble model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict for the next `future_days` days
    last_date = daily_spend['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
    future_features = np.array([[date.toordinal(), date.dayofweek, date.month] for date in future_dates])
    
    predictions = model.predict(future_features)
    
    # Ensure no negative predictions for expenses
    predictions = np.maximum(predictions, 0)
    
    # Construct results dataframe
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Amount': predictions
    })
    
    return pred_df
