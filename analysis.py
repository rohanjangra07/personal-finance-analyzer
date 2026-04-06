import pandas as pd

def load_and_clean_data(file):
    """
    Loads expense data from a CSV file and creates new features.
    """
    try:
        df = pd.read_csv(file)
        
        # Ensure correct columns exist
        required_cols = ['Date', 'Amount', 'Category', 'Description']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV must contain columns: Date, Amount, Category, Description")
            
        # Convert Date to datetimeformat
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create new features
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        df['Day'] = df['Date'].dt.day_name()
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def calculate_kpis(df):
    """
    Calculates primary Key Performance Indicators.
    """
    if df.empty:
        return 0, 0, 0
    
    total_expense = df['Amount'].sum()
    num_transactions = len(df)
    avg_expense = total_expense / num_transactions if num_transactions > 0 else 0
    
    return total_expense, num_transactions, avg_expense

def get_category_analysis(df):
    """
    Groups expenses by category.
    """
    if df.empty:
        return pd.DataFrame()
    return df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)

def get_monthly_trends(df):
    """
    Groups expenses by month (YYYY-MM).
    """
    if df.empty:
        return pd.DataFrame()
    return df.groupby('Month')['Amount'].sum().reset_index().sort_values(by='Month')

def generate_smart_insights(df):
    """
    Generates smart alerts and insights from the data.
    """
    insights = []
    
    if df.empty:
        return ["Not enough data to generate insights."]
        
    avg_expense = df['Amount'].mean()
    
    # Check for overspending on individual transactions
    overspent_transactions = df[df['Amount'] > avg_expense * 3]
    if not overspent_transactions.empty:
        insights.append(f"⚠️ You have {len(overspent_transactions)} unusually high transactions.")
        
    # Top spending category
    cat_df = get_category_analysis(df)
    if not cat_df.empty:
        top_cat = cat_df.iloc[0]['Category']
        top_amt = cat_df.iloc[0]['Amount']
        insights.append(f"💡 You spend most on {top_cat}.")
        
    # Weekend vs Weekday
    weekend = df[df['Day'].isin(['Saturday', 'Sunday'])]['Amount'].sum()
    weekday = df[~df['Day'].isin(['Saturday', 'Sunday'])]['Amount'].sum()
    if weekend > weekday:
        insights.append("⚠️ You spend more on weekends.")

    # Expense trend increase
    monthly = df.groupby('Month')['Amount'].sum()
    if len(monthly) >= 2:
        if monthly.iloc[-1] > monthly.iloc[-2]:
            insights.append("📈 Your spending increased this month.")
            
    return insights

def financial_health_score(df):
    """
    Calculates a simple 0-100 financial health score based on average expense.
    """
    if df.empty:
        return 0
    
    avg_expense = df['Amount'].mean()
    
    # Simple scoring logic
    if avg_expense < 200:
        score = 90
    elif avg_expense < 500:
        score = 70
    else:
        score = 40
        
    return score

def detect_anomalies(df):
    if df.empty:
        return pd.DataFrame()
    
    mean = df['Amount'].mean()
    std = df['Amount'].std()
    
    threshold = mean + 2 * std
    
    anomalies = df[df['Amount'] > threshold]
    
    return anomalies

def ai_financial_advisor(df):
    advice = []
    
    if df.empty:
        return ["Not enough data."]
    
    # Top category
    top_cat = df.groupby('Category')['Amount'].sum().idxmax()
    
    if top_cat == "Food":
        advice.append("🍔 Try reducing food expenses to save more.")
    elif top_cat == "Shopping":
        advice.append("🛍️ Control shopping expenses to improve savings.")
    
    # High average spending
    avg = df['Amount'].mean()
    if avg > 500:
        advice.append("⚠️ Your average spending is high. Consider budgeting.")
    
    return advice
