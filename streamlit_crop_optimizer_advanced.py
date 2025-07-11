
# Field-Aware Sustainable Crop Planning App – Advanced Version (Academic Extension)
# Developed by Vasavi Agarwal & Team

import streamlit as st
from pulp import *
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Field-Aware Crop Optimizer", layout="wide")
st.title("Field-Aware Sustainable Crop Planning Optimization")
st.markdown("This system extends our crop planning model to consider multiple fields with different soil, water, and sustainability profiles.")

# Sidebar Inputs
st.sidebar.header("General Parameters")
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

total_labor = st.sidebar.slider("Total Labor Available (man-days)", 10000, 150000, 90000)
total_equipment = st.sidebar.slider("Total Equipment Hours Available", 5000, 50000, 25000)
alpha = st.sidebar.slider("Profit Weight (α)", 0.0, 1.0, 0.8)
beta = st.sidebar.slider("Sustainability Weight (β)", 0.0, 1.0, 0.2)

# Default values
profit_per_ha = {'Wheat': 8000, 'Rice': 9500, 'Maize': 9000, 'Soyabean': 8700, 'Cotton': 9300}
water_per_ha = {'Wheat': 1200, 'Rice': 1800, 'Maize': 1000, 'Soyabean': 1100, 'Cotton': 1500}
labor_per_ha = {'Wheat': 20, 'Rice': 30, 'Maize': 25, 'Soyabean': 22, 'Cotton': 28}
equipment_per_ha = {'Wheat': 8, 'Rice': 10, 'Maize': 6, 'Soyabean': 7, 'Cotton': 9}
price_fluctuation = {'Wheat': 0.95, 'Rice': 1.1, 'Maize': 0.9, 'Soyabean': 1.05, 'Cotton': 1.0}
weather_factor = {'Wheat': 0.95, 'Rice': 0.85, 'Maize': 0.9, 'Soyabean': 0.92, 'Cotton': 0.88}

# Field-specific soil suitability (0 to 1) and water availability
st.markdown("### Field Characteristics")
field_area = {}
field_water = {}
soil_score = {}

for field in fields:
    st.subheader(field)
    field_area[field] = st.number_input(f"Area of {field} (ha)", min_value=0, value=1000)
    field_water[field] = st.number_input(f"Water available in {field} (mm)", min_value=0, value=1500000)
    soil_score[field] = {}
    for crop in crops:
        soil_score[field][crop] = st.slider(f"{crop} soil suitability in {field}", 0.0, 1.0, 0.8)

if st.button("Run Optimization"):
    model = LpProblem("Field_Aware_Crop_Planning", LpMaximize)
    land = LpVariable.dicts("Land", ((field, crop) for field in fields for crop in crops), lowBound=0, cat='Continuous')

    # Adjusted profit with market/weather
    adjusted_profit = {
        crop: profit_per_ha[crop] * price_fluctuation[crop] * weather_factor[crop]
        for crop in crops
    }

    # Objective: profit + sustainability via soil_score
    model += lpSum([
        alpha * adjusted_profit[crop] * land[field, crop] +
        beta * soil_score[field][crop] * land[field, crop]
        for field in fields for crop in crops
    ]), "Total_Profit_and_Sustainability"

    # Field-wise area constraints
    for field in fields:
        model += lpSum([land[field, crop] for crop in crops]) <= field_area[field], f"Area_{field}"
        model += lpSum([land[field, crop] * water_per_ha[crop] for crop in crops]) <= field_water[field], f"Water_{field}"

    # Global constraints
    model += lpSum([land[field, crop] * labor_per_ha[crop] for field in fields for crop in crops]) <= total_labor, "Labor"
    model += lpSum([land[field, crop] * equipment_per_ha[crop] for field in fields for crop in crops]) <= total_equipment, "Equipment"

    model.solve()

    if model.status == 1:
        st.markdown("## Optimal Land Allocation")
        result_table = []
        for field in fields:
            row = {'Field': field}
            for crop in crops:
                row[crop] = round(land[field, crop].varValue or 0, 2)
            result_table.append(row)
        st.dataframe(result_table)

        total_profit = sum(adjusted_profit[crop] * land[field, crop].varValue for field in fields for crop in crops)
        sustainability_score = sum(soil_score[field][crop] * land[field, crop].varValue for field in fields for crop in crops)
        st.write(f"Total Adjusted Profit: ₹{total_profit:,.2f}")
        st.write(f"Total Weighted Sustainability Score: {sustainability_score:.2f}")

        st.markdown("### Field-wise Allocation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        data = np.array([[land[field, crop].varValue or 0 for crop in crops] for field in fields])
        im = ax.imshow(data, cmap="YlGn", aspect="auto")
        ax.set_xticks(np.arange(len(crops)))
        ax.set_yticks(np.arange(len(fields)))
        ax.set_xticklabels(crops)
        ax.set_yticklabels(fields)
        for i in range(len(fields)):
            for j in range(len(crops)):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", color="black")
        st.pyplot(fig)
    else:
        st.error("Optimization failed. Please revise your inputs.")
