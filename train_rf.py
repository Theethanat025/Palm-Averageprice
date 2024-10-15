import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. การเตรียมข้อมูล
# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('palm averageprice 2017_2023.csv')

# แปลงชื่อเดือนเป็นตัวเลข
month_mapping = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}
data['Month'] = data['Month'].map(month_mapping)

# สร้างฟีเจอร์ปีและเดือนเพื่อใช้ทำนาย
X = data[['Year', 'Month']]
y = data['AveragePrice']

# แบ่งข้อมูลออกเป็นชุดการฝึก (train) และทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. การสร้างโมเดล Random Forest
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# 3. การประเมินผลและวัดความแม่นยำ
# ทำนายด้วย Random Forest
y_pred_rf = random_forest_model.predict(X_test)

# คำนวณ Mean Squared Error (MSE) และ R-squared (R²)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# แสดงผลการประเมิน
print("Random Forest - Mean Squared Error:", mse_rf)
print("Random Forest - R-squared:", r2_rf)
