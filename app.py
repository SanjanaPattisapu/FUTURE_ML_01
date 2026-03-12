import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.title('📈Sales Forecasting Dashboard')
uploaded_file=st.file_uploader('Upload Sales CSV File')
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file, encoding='latin1')
    st.subheader('Data Preview')
    st.write(df.head())
    print(df.columns)
    df['Order Date']=pd.to_datetime(df['Order Date'])
    df['month']=df['Order Date'].dt.month
    df['year']=df['Order Date'].dt.year
    X=df[['month','year']]
    y=df['Sales']
    model=LinearRegression()
    model.fit(X,y)
    df['Predicted_Sales']=model.predict(X)
    st.subheader('Sales vs Predicted Sales')
    fig, ax=plt.subplots()
    ax.plot(df['Order Date'],df['Sales'],label='Actual Sales')
    ax.plot(df['Order Date'],df['Predicted_Sales'],label='Predicted_Sales')
    ax.legend()
    st.pyplot(fig)
    st.subheader('Forecast Future Sales')
    month = st.number_input("Enter Month",1,12)
    year = st.number_input("Enter Year",2024,2030)
    future = pd.DataFrame([[month,year]],columns=['month','year'])
    prediction = model.predict(future)
    st.write("Predicted Sales:",prediction[0])
    st.subheader("Dataset Information")
    st.write(df.info())
    st.subheader("Sales Statistics")
    st.write(df['Sales'].describe())
    df['month'] = df['Order Date'].dt.month
    monthly_sales = df.groupby('month')['Sales'].sum()
    st.subheader("Monthly Sales")
    st.bar_chart(monthly_sales)
    st.subheader("Sales Insights")
    st.write("Highest Sales:", df['Sales'].max())
    st.write("Lowest Sales:", df['Sales'].min())
    st.sidebar.header("Filter Data")
    selected_month = st.sidebar.selectbox(
    "Select Month",
    df['month'].unique()
    )
    filtered_df = df[df['month'] == selected_month]
    st.write(filtered_df)
    st.subheader("Sales Trend")
    st.line_chart(df.set_index('Order Date')['Sales'])
    