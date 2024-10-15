import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# CSS สำหรับเปลี่ยนสีพื้นหลัง
page_bg_img = '''
<style>
body {
    background-color: #f5f5f5;
}
</style>
'''

# แทรก CSS ไปยังแอป Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# 1. การเตรียมข้อมูล
data = pd.read_csv('palm averageprice 2017_2023.csv')

# แปลงชื่อเดือนเป็นตัวเลข
month_mapping = {
    "january": 1, "february": 2, "march": 3, "april": 4, 
    "may": 5, "june": 6, "july": 7, "august": 8, 
    "september": 9, "october": 10, "november": 11, "december": 12
}
data['Month'] = data['Month'].map(month_mapping)

X = data[['Year', 'Month']]
y = data['AveragePrice']

# แบ่งข้อมูลออกเป็นชุดการฝึก (train) และทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Random Forest และฝึกโมเดล
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(model, 'Palm_averageprice_model.pkl')

# ฟังก์ชันทำนายราคาปาล์มน้ำมันและคำนวณช่วงความน่าจะเป็น
def predict_oil_price(features):
    prediction = model.predict([features])[0]  # ทำนายราคา

    uncertainty = 0.05 * prediction  # คำนวณความคลาดเคลื่อน (สมมติใช้ 5% ของการทำนาย)

    lower_bound = prediction - uncertainty
    upper_bound = prediction + uncertainty

    return prediction, lower_bound, upper_bound

# ฟังก์ชันแสดงกราฟย้อนหลังและทำนายอนาคตแบบรายเดือน
def plot_past_future_prices_by_month(prediction, month_number):
    past_data = data[data['Month'] == month_number]
    past_years = past_data['Year']
    past_prices = past_data['AveragePrice']

    future_year = past_data['Year'].max() + 1
    future_price = prediction

    plt.figure(figsize=(10, 6))

    # แสดงข้อมูลราคาย้อนหลัง
    plt.plot(past_years, past_prices, label="Past Prices", color="blue", marker="o")

    # แสดงจุดทำนายอนาคต
    plt.plot(future_year, future_price, label="Predicted Future Price", color="red", linestyle="--", marker="x")

    # เส้นประสีแดงจากข้อมูลสุดท้ายไปยังการทำนาย
    plt.plot([past_years.max(), future_year], [past_prices.iloc[-1], future_price], color="red", linestyle="--", label="Prediction Transition")

    # เพิ่มตัวเลขค่าแสดงด้านบนจุดในอดีต
    for i in range(len(past_years)):
        plt.text(past_years.iloc[i], past_prices.iloc[i] + 0.1, f'{past_prices.iloc[i]:.2f}', ha='center', color='blue')

    # เพิ่มตัวเลขค่าแสดงด้านบนจุดทำนาย
    plt.text(future_year, future_price + 0.1, f'{future_price:.2f}', ha='center', color='red')

    plt.xlabel("Year")
    plt.ylabel("Palm Oil Price (Baht)")
    plt.title(f"Palm Oil Prices for Month {month_number} (Past and Future Prediction)")
    plt.legend()
    st.pyplot(plt)

    # คำนวณค่า R²
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # แสดงค่า R² ใต้กราฟ
    st.markdown(f"<h4>ค่า R² สำหรับโมเดลนี้คือ: <b>{r2:.2f}</b></h4>", unsafe_allow_html=True)


# ส่วนของการทำนายและแสดงผล
st.markdown("<h2 style='text-align: center;'>ทำนายราคาเฉลี่ยของผลปาล์มน้ำมัน</h2>", unsafe_allow_html=True)

# ผู้ใช้เลือกเดือนที่ต้องการทำนาย
month = st.selectbox('📅 Month : กรุณาเลือกเดือนที่ท่านต้องการทราบ', list(month_mapping.keys()))

if st.button('💡 Predict Oil Price'):
    month_number = month_mapping[month]
    
    # ดึงปีล่าสุดจากข้อมูลและบวก 1 เพื่อทำนายปีถัดไป
    last_year = data['Year'].max()
    next_year = last_year + 1

    # ข้อมูลที่ใช้ทำนาย (ปีถัดไป, เดือน)
    features = [next_year, month_number]
    prediction, lower_bound, upper_bound = predict_oil_price(features)

    # แสดงฤดูกาลตามเดือน
    st.markdown(f'<h3 style="color:#66bb6a;">🌿 Predicted Palm Oil Price for {next_year}: <b><span style="color:yellow;">{prediction:.2f}</span> Baht</b></h3>', unsafe_allow_html=True)
    st.markdown(f'<h4 style="color:#66bb6a;">🌟 Prediction Interval: <b>{lower_bound:.2f} - {upper_bound:.2f} Baht</b></h4>', unsafe_allow_html=True)

    plot_past_future_prices_by_month(prediction, month_number)
