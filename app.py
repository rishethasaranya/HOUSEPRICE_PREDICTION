import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="House Rent Predictor", page_icon="🏠", layout="centered")

st.markdown("<h1 style='text-align: center;'>🏠 House Rent Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Hyderabad_House_Data.csv")
    return data

data = load_data()

# -------------------- PREPROCESSING --------------------
data['Washrooms'] = pd.to_numeric(data['Washrooms'], errors='coerce')
data['Washrooms'] = data['Washrooms'].fillna(data['Washrooms'].median())

data['Tennants'] = pd.to_numeric(data['Tennants'], errors='coerce')
data['Tennants'] = data['Tennants'].fillna(data['Tennants'].median())

data['Area'] = data['Area'].astype(str).str.extract('(\d+)')
data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
data['Area'] = data['Area'].fillna(data['Area'].median())

data['Price'] = (
    data['Price']
    .astype(str)
    .str.replace(r'[^\d]', '', regex=True)
)
data['Price'] = data['Price'].astype(float)

# -------------------- MODEL TRAINING --------------------
x = data.drop('Price', axis=1)
y = data['Price']

x = pd.get_dummies(x, drop_first=True)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

# -------------------- INPUT SECTION --------------------
st.subheader("Enter House Details")

area = st.number_input("Area (sqft)", min_value=100, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
washrooms = st.number_input("Washrooms", min_value=1, step=1)

st.markdown("")

# -------------------- PREDICTION --------------------
if st.button("Predict Rent 💰"):

    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms]
    })

    new_house = pd.get_dummies(new_house)
    new_house = new_house.reindex(columns=x.columns, fill_value=0)

    new_scaled = scaler.transform(new_house)
    predicted_price = model.predict(new_scaled)

    st.success(f"Estimated House Rent: ₹ {predicted_price[0]:,.0f} per month")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
