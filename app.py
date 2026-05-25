import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Property Price Predictor", layout="centered")

# -----------------------------
# SESSION STATE
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

# -----------------------------
# LOAD MODEL + COLUMNS
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))

# -----------------------------
# STEP 1 : USER INFO
# -----------------------------
if st.session_state.step == 1:

    st.title("👤 Self Information")

    name = st.text_input("Your Name")
    email = st.text_input("Email")

    city = st.selectbox("City", ["Pune", "Mumbai", "Delhi"])

    if st.button("Continue ➡"):
        st.session_state.step = 2
        st.rerun()

# -----------------------------
# FUNCTION: CREATE INPUT
# -----------------------------
def create_input():

    # create empty dataframe with correct columns
    input_df = pd.DataFrame(0, index=[0], columns=cols)

    # -----------------------------
    # NUMERIC INPUTS
    # -----------------------------
    if "Lot_Size" in cols:
        input_df.loc[0, "Lot_Size"] = area

    if "Garage_Area" in cols:
        input_df.loc[0, "Garage_Area"] = garage

    if "Bedroom_Above_Grade" in cols:
        input_df.loc[0, "Bedroom_Above_Grade"] = bedrooms

    if "Rooms_Above_Grade" in cols:
        input_df.loc[0, "Rooms_Above_Grade"] = bedrooms + bathrooms

    if "Total_Basement_Area" in cols:
        input_df.loc[0, "Total_Basement_Area"] = basement_area

    if "First_Floor_Area" in cols:
        input_df.loc[0, "First_Floor_Area"] = first_floor

    if "Construction_Year" in cols:
        input_df.loc[0, "Construction_Year"] = year_built

    # -----------------------------
    # CATEGORICAL ENCODING
    # -----------------------------
    def set_cat(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    set_cat("Neighborhood", neighborhood)
    set_cat("Roof_Design", roof_design)
    set_cat("Heating_Type", heating_type)
    set_cat("Kitchen_Quality", kitchen_quality)
    set_cat("Foundation_Type", foundation_type)
    set_cat("House_Type", property_type)
    set_cat("Balcony", balcony)
    set_cat("Furnishing", furnishing)

    return input_df


# -----------------------------
# STEP 2 : PREDICTION PAGE
# -----------------------------
if st.session_state.step == 2:

    st.title("🏠 Smart Property Price Predictor")
    st.write("Enter Property Details 👇")

    # -----------------------------
    # INPUTS
    # -----------------------------
    area = st.number_input("Lot Size", value=5000)
    bedrooms = st.number_input("Bedrooms", value=3)
    bathrooms = st.number_input("Bathrooms", value=2)
    garage = st.number_input("Garage Area", value=300)
    basement_area = st.number_input("Basement Area", value=800)
    first_floor = st.number_input("First Floor Area", value=1200)
    year_built = st.number_input("Year Built", value=2010)

    balcony = st.selectbox("Balcony", ["Yes", "No"])
    property_type = st.selectbox("Property Type", ["1Fam", "2fmCon", "Duplex", "Twnhs"])
    furnishing = st.selectbox("Furnishing Type", ["Fully Furnished", "Semi", "Unfurnished"])
    neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown"])
    roof_design = st.selectbox("Roof Design", ["Gable", "Hip", "Flat"])
    heating_type = st.selectbox("Heating Type", ["GasA", "GasW", "Grav"])
    kitchen_quality = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
    foundation_type = st.selectbox("Foundation Type", ["PConc", "CBlock", "BrkTil"])

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if st.button("Predict Price 💰"):

        input_data = create_input()

        prediction = model.predict(input_data)

        st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)}")

        st.balloons()