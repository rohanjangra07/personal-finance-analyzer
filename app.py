import streamlit as st
import pandas as pd
import plotly.express as px
from analysis import load_and_clean_data, calculate_kpis, get_category_analysis, get_monthly_trends, generate_smart_insights, financial_health_score, detect_anomalies, ai_financial_advisor
from model import predict_future_expenses

# ================================
# STREAMLIT CONFIG
# ================================
st.set_page_config(page_title="Personal Finance Analyzer", layout="wide", page_icon="💸")

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.title("💸 FinAnalyzer")
    st.markdown("---")
    
    # 1. Theme Configuration
    theme_selection = st.radio("Select Theme:", ["Dark Mode 🌙", "Light Mode ☀️"])
    
    # 2. Currency Configuration
    currency_map = {"USD ($)": "$", "EUR (€)": "€", "INR (₹)": "₹", "GBP (£)": "£"}
    selected_curr_key = st.selectbox("Select Currency:", list(currency_map.keys()))
    currency = currency_map[selected_curr_key]
    
    st.markdown("---")
    
    # 3. File Upload
    uploaded_file = st.file_uploader("Upload Expense CSV", type="csv")
    if not uploaded_file:
        st.info("No file uploaded. Using sample data.")
        uploaded_file = "sample_data.csv"

# ================================
# CUSTOM CSS
# ================================
dark_css = """
<style>
/* Streamlit root */
.stApp { background-color: #0f172a; color: #f8fafc; }
h1, h2, h3, h4, h5, h6, strong { color: #f8fafc; }
div[data-testid="stMetricValue"] { color: #10b981; font-size: 2.2rem; font-weight: bold;}
div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 1rem; }
hr { border-color: #334155; }
.stAlert { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; }
</style>
"""

light_css = """
<style>
/* Streamlit root */
.stApp { background-color: #f8fafc; color: #0f172a; }
h1, h2, h3, h4, h5, h6, strong { color: #0f172a; }
div[data-testid="stMetricValue"] { color: #059669; font-size: 2.2rem; font-weight: bold;}
div[data-testid="stMetricLabel"] { color: #475569; font-size: 1rem; }
hr { border-color: #e2e8f0; }
.stAlert { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; }
</style>
"""

# Apply Theme
if "Dark" in theme_selection:
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# ================================
# MAIN APP Logic
# ================================
try:
    df = load_and_clean_data(uploaded_file)
    
    # Navigation & Filters
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("Go to:", ["📊 Dashboard", "📉 Trends & Insights", "🔮 Predictions"])
        
        st.markdown("---")
        st.markdown("### Filters")
        
        selected_month = st.selectbox(
            "Select Month",
            ["All"] + sorted(df['Month'].unique().tolist())
        )
        
        selected_category = st.selectbox(
            "Select Category",
            ["All"] + sorted(df['Category'].unique().tolist())
        )
        
        st.markdown("### 💰 Budget Settings")
        budget = st.number_input(
            "Set Monthly Budget",
            min_value=0,
            value=5000
        )

    # Apply filters
    if selected_month != "All":
        df = df[df['Month'] == selected_month]

    if selected_category != "All":
        df = df[df['Category'] == selected_category]

    st.title("💡 AI Insights & Financial Overview")
    st.markdown("Track, analyze, and predict your upcoming expenses with precision.")
    st.markdown("---")
    
    # Shared KPIs
    tot, cnt, avg = calculate_kpis(df)
    score = financial_health_score(df)
    top_category = df.groupby('Category')['Amount'].sum().idxmax() if not df.empty else "N/A"
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Total Expenses", value=f"{currency}{tot:,.2f}")
    kpi2.metric(label="Transactions", value=f"{cnt}")
    kpi3.metric(label="Avg Expense", value=f"{currency}{avg:,.2f}")
    kpi4.metric(label="💚 Health Score", value=f"{score}/100")
    
    st.metric("Top Category", top_category)
    st.markdown("---")
    
    if page == "📊 Dashboard":
        # Budget Check
        if tot > budget:
            st.error("⚠️ You have exceeded your budget!")
        else:
            st.success("✅ You are within your budget")
            
        # Layout Split for Dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Category Wise Expense (Bar Chart)")
            cat_df = get_category_analysis(df)
            if not cat_df.empty:
                fig_bar = px.bar(cat_df, x="Category", y="Amount", 
                                 template="plotly_dark" if "Dark" in theme_selection else "plotly_white",
                                 color="Category", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_bar.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        with col2:
            st.subheader("Smart Insights")
            insights = generate_smart_insights(df)
            for i in insights:
                if "Warning" in i:
                    st.warning(i)
                elif "Insight" in i:
                    st.info(i)
                else:
                    st.write(i)

            st.markdown("---")
            st.subheader("Expense Distribution")
            if not cat_df.empty:
                fig_pie = px.pie(cat_df, values="Amount", names="Category", 
                                 template="plotly_dark" if "Dark" in theme_selection else "plotly_white",
                                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
                
        st.markdown("---")
        st.subheader("Recent Transactions")
        display_df = df[['Date', 'Amount', 'Category', 'Description']].sort_values(by='Date', ascending=False)
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"{currency}{x:,.2f}")
        # remove use_container_width to fix deprecation warning
        st.dataframe(display_df, hide_index=True)

        st.markdown("---")
        st.subheader("🚨 Unusual Transactions")
        anomalies = detect_anomalies(df)
        if not anomalies.empty:
            st.warning(f"{len(anomalies)} unusual transactions detected!")
            st.dataframe(anomalies[['Date', 'Amount', 'Category']], hide_index=True)
        else:
            st.success("No unusual spending detected.")
            
        st.markdown("---")
        st.subheader("🧠 AI Financial Advisor")
        advice_list = ai_financial_advisor(df)
        for tip in advice_list:
            st.info(tip)

    elif page == "📉 Trends & Insights":
        st.subheader("📅 Daily Spending Trend (With Smoothing)")
        daily_df = df.groupby('Date')['Amount'].sum().reset_index()
        daily_df['3-Day Average'] = daily_df['Amount'].rolling(3).mean()
        
        fig_daily = px.line(
            daily_df,
            x="Date",
            y=["Amount", "3-Day Average"],
            markers=True,
            template="plotly_dark" if "Dark" in theme_selection else "plotly_white",
            line_shape="spline",
            color_discrete_sequence=["#64748b", "#10b981"] # Muted gray/blue for actual, bright green for smoothed
        )
        fig_daily.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Date",
            yaxis_title=f"Amount ({currency})",
            legend_title_text=""
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        st.markdown("---")

        st.subheader("Monthly Spending Trend")
        trend_df = get_monthly_trends(df)
        if not trend_df.empty:
            fig_trend = px.line(trend_df, x="Month", y="Amount", markers=True, 
                                template="plotly_dark" if "Dark" in theme_selection else "plotly_white",
                                line_shape="spline", color_discrete_sequence=["#10b981"])
            fig_trend.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Month", yaxis_title=f"Amount ({currency})")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data to show monthly trends.")

    elif page == "🔮 Predictions":
        st.subheader("Predictive Analytics (Next 7 Days)")
        st.markdown("This model uses **Random Forest Regressor** on your daily historical spending to capture non-linear patterns and forecast your next 7 days.")
        
        # Add warning for low data
        daily_len = len(df.groupby('Date')['Amount'].sum())
        if daily_len < 7:
            st.warning("⚠️ Prediction accuracy may be low due to limited historical data. Add more data for better accuracy.")
            
        pred_df = predict_future_expenses(df)
        if not pred_df.empty:
            # Combine historical and future
            hist_df = df.groupby('Date')['Amount'].sum().reset_index().sort_values('Date').tail(14)
            hist_df['Type'] = 'Historical'
            pred_df['Type'] = 'Predicted'
            pred_df = pred_df.rename(columns={'Predicted_Amount': 'Amount'})
            
            combined = pd.concat([hist_df, pred_df])
            fig_pred = px.bar(combined, x="Date", y="Amount", color="Type",
                              template="plotly_dark" if "Dark" in theme_selection else "plotly_white",
                              color_discrete_map={"Historical": "#94a3b8", "Predicted": "#3b82f6"},
                              barmode="group")
            fig_pred.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show Data table of predictions
            st.markdown("##### Predicted Spending Breakdown")
            pred_display = pred_df[['Date', 'Amount']].copy()
            pred_display['Date'] = pred_display['Date'].dt.strftime('%Y-%m-%d')
            pred_display['Amount'] = pred_display['Amount'].apply(lambda x: f"{currency}{x:,.2f}")
            st.dataframe(pred_display, hide_index=True)
            
        else:
            st.info("Need more daily transaction data for predictions.")

except Exception as e:
    st.error(f"Failed to load application: {e}")
