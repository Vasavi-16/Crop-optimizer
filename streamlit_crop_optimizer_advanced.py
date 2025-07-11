# Field-Aware Sustainable Crop Planning App – Updated Formula-Based Version

import streamlit as st
from pulp import *
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Field-Aware Crop Optimizer", layout="wide")
st.title("Field-Aware Sustainable Crop Planning Optimization")
st.markdown("This system extends our crop planning model to consider multiple fields with different soil, water and sustainability profiles.")

# Sidebar Inputs
st.sidebar.header("General Parameters")
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

total_labor = st.sidebar.slider("Total Labor Available (man-days)", 10000, 150000, 90000)
total_equipment = st.sidebar.slider("Total Equipment Hours Available", 5000, 50000, 25000)
alpha = st.sidebar.slider("Water Weight (α)", 0.0, 1.0, 0.5)
beta = st.sidebar.slider("Fertilizer Weight (β)", 0.0, 1.0, 0.3)
gamma = st.sidebar.slider("Profit Weight (γ)", 0.0, 1.0, 0.7)

fertilizer_cost = 25  # ₹/kg

# Crop-specific parameters
st.sidebar.markdown("### Crop Parameters (per ha)")
yield_per_ha = {}
price_per_ton = {}
fertilizer_per_ha = {}
labor_per_ha = {}
equipment_per_ha = {}
water_per_ha = {}

for crop in crops:
    st.sidebar.markdown(f"**{crop}**")
    yield_per_ha[crop] = st.sidebar.number_input(f"{crop} Yield (tons/ha)", min_value=0.0, value=3.0)
    price_per_ton[crop] = st.sidebar.number_input(f"{crop} Price (₹/ton)", min_value=0.0, value=3000.0)
    fertilizer_per_ha[crop] = st.sidebar.number_input(f"{crop} Fertilizer (kg/ha)", min_value=0.0, value=100.0)
    water_per_ha[crop] = st.sidebar.number_input(f"{crop} Water (liters/ha)", min_value=0.0, value=1200000.0)
    labor_per_ha[crop] = st.sidebar.number_input(f"{crop} Labor (man-days/ha)", min_value=0.0, value=20.0)
    equipment_per_ha[crop] = st.sidebar.number_input(f"{crop} Equipment (hours/ha)", min_value=0.0, value=8.0)

# Field-specific inputs
st.markdown("### Field Characteristics")
field_area = {}
field_water = {}
rainfall_index = {}

for field in fields:
    st.subheader(field)
    field_area[field] = st.number_input(f"Area of {field} (ha)", min_value=0, value=1000)
    field_water[field] = st.number_input(f"Water available in {field} (liters)", min_value=0, value=150000000)
    rainfall_index[field] = st.slider(f"Rainfall Index in {field} (0–1)", 0.0, 1.0, 0.7)

if st.button("Run Optimization"):
    model = LpProblem("Field_Aware_Crop_Planning", LpMaximize)
    land = LpVariable.dicts("Land", ((field, crop) for field in fields for crop in crops), lowBound=0, cat='Continuous')

    # Calculate Costs based on formulas
    profit_cost = {}
    sustain_cost = {}
    hybrid_cost = {}

    for field in fields:
        for crop in crops:
            Y = yield_per_ha[crop]
            P = price_per_ton[crop]
            F = fertilizer_per_ha[crop]
            W = water_per_ha[crop]
            RI = rainfall_index[field]

            profit_cost[(field, crop)] = -(Y * P) - (F * fertilizer_cost)
            sustain_cost[(field, crop)] = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))
            hybrid_cost[(field, crop)] = gamma * profit_cost[(field, crop)] + sustain_cost[(field, crop)]

    # Objective function: maximize negative hybrid cost (i.e., minimize cost)
    model += lpSum([
        -hybrid_cost[(field, crop)] * land[(field, crop)]
        for field in fields for crop in crops
    ]), "Total_Hybrid_Cost"

    # Constraints
    for field in fields:
        model += lpSum([land[field, crop] for crop in crops]) <= field_area[field], f"Area_{field}"
        model += lpSum([land[field, crop] * water_per_ha[crop] for crop in crops]) <= field_water[field], f"Water_{field}"

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

        total_cost = sum(hybrid_cost[(field, crop)] * land[field, crop].varValue for field in fields for crop in crops)
        st.write(f"Total Hybrid Cost Value: ₹{total_cost:,.2f}")

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

