import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

from xgboost import XGBRegressor

# ----------------------------
# 1. LOAD DATA
# ----------------------------
data = pd.read_csv(r"C:\Users\Sayali\Downloads\Property_Price_Train (2).csv")

# ----------------------------
# 2. SPLIT FEATURES & TARGET
# ----------------------------
X = data.drop('Sale_Price', axis=1)
y = data['Sale_Price']

# ----------------------------
# 3. COLUMN TYPES
# ----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# ----------------------------
# 4. PREPROCESSING
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# ----------------------------
# 5. XGBOOST MODEL
# ----------------------------
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ----------------------------
# 6. FULL PIPELINE
# ----------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# ----------------------------
# 7. TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 8. TRAIN MODEL
# ----------------------------
model.fit(X_train, y_train)

# ----------------------------
# 9. PREDICTION
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 10. EVALUATION
# ----------------------------
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# ----------------------------
# 11. SAVE MODEL
# ----------------------------
pickle.dump(model, open("xgboost_property_model.pkl", "wb"))

print("XGBoost Model Saved Successfully!")

import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open("xgboost_property_model.pkl", "rb"))

st.title("Property Price Prediction (XGBoost)")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

input_df = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms
}])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Price: {prediction[0]}")

    import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))