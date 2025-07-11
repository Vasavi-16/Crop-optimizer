import streamlit as st
from pulp import *

st.set_page_config(page_title="Field-Aware Crop Optimizer", layout="wide")
st.title("Field-Aware Sustainable Crop Planning Optimization")
st.markdown("This system extends our crop planning model to consider multiple fields with different soil, water and sustainability profiles.")

# Sidebar: Essential Controls
st.sidebar.header("Available Resources")
total_labor = st.sidebar.slider("Total Labor (man-days)", 10000, 150000, 90000, step=1000)
total_equipment = st.sidebar.slider("Total Equipment Hours", 5000, 50000, 25000, step=500)

st.sidebar.header("Weight Parameters")
alpha = st.sidebar.slider("Water Scarcity Weight (α)", 0.0, 1.0, 0.5)
beta = st.sidebar.slider("Fertilizer Usage Weight (β)", 0.0, 1.0, 0.3)
gamma = st.sidebar.slider("Profit Importance Weight (γ)", 0.0, 1.0, 0.7)

fertilizer_cost = 25  # ₹/kg

# Crops and Fields
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

# Fixed crop parameters (per ha)
yield_per_ha = {'Wheat': 3.2, 'Rice': 4.5, 'Maize': 3.8, 'Soyabean': 2.5, 'Cotton': 2.2}
price_per_ton = {'Wheat': 3000, 'Rice': 2800, 'Maize': 2700, 'Soyabean': 3500, 'Cotton': 4000}
fertilizer_per_ha = {'Wheat': 100, 'Rice': 120, 'Maize': 90, 'Soyabean': 80, 'Cotton': 110}
water_per_ha = {'Wheat': 1200000, 'Rice': 1800000, 'Maize': 1000000, 'Soyabean': 1100000, 'Cotton': 1500000}
labor_per_ha = {'Wheat': 20, 'Rice': 30, 'Maize': 25, 'Soyabean': 22, 'Cotton': 28}
equipment_per_ha = {'Wheat': 8, 'Rice': 10, 'Maize': 6, 'Soyabean': 7, 'Cotton': 9}

# Field characteristics
st.markdown("### Field Characteristics")
field_area = {}
field_water = {}
rainfall_index = {}

for field in fields:
    st.subheader(field)
    field_area[field] = st.number_input(f"Area of {field} (ha)", min_value=0, value=1000, step=100)
    field_water[field] = st.number_input(f"Water available in {field} (liters)", min_value=0, value=150000000, step=1000000)
    rainfall_index[field] = st.slider(f"Rainfall Index in {field} (0–1)", 0.0, 1.0, 0.7)

if st.button("Run Optimization"):
    model = LpProblem("Field_Aware_Crop_Planning", LpMaximize)
    land = LpVariable.dicts("Land", ((field, crop) for field in fields for crop in crops), lowBound=0, cat='Continuous')

    hybrid_score = {}

    for field in fields:
        for crop in crops:
            Y = yield_per_ha[crop]
            P = price_per_ton[crop]
            F = fertilizer_per_ha[crop]
            W = water_per_ha[crop]
            RI = rainfall_index[field]

            # Profit and sustainability cost
            profit_cost = -(Y * P) - (F * fertilizer_cost)
            sustain_cost = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))
            hybrid_cost = gamma * profit_cost + sustain_cost

            # FINAL SCORE (inverted cost)
            hybrid_score[(field, crop)] = -hybrid_cost

    # Objective Function: maximize hybrid score
    model += lpSum([hybrid_score[(field, crop)] * land[(field, crop)] for field in fields for crop in crops])

    # Constraints
    for field in fields:
        model += lpSum([land[field, crop] for crop in crops]) <= field_area[field], f"Area_{field}"
        model += lpSum([land[field, crop] * water_per_ha[crop] for crop in crops]) <= field_water[field], f"Water_{field}"

    model += lpSum([land[field, crop] * labor_per_ha[crop] for field in fields for crop in crops]) <= total_labor, "Labor"
    model += lpSum([land[field, crop] * equipment_per_ha[crop] for field in fields for crop in crops]) <= total_equipment, "Equipment"

    model.solve()

    if model.status == 1:
        st.markdown("## Optimal Land Allocation (ha)")
        result_table = []
        total_score = 0
        for field in fields:
            row = {'Field': field}
            for crop in crops:
                area = land[field, crop].varValue or 0
                row[crop] = round(area, 2)
                total_score += hybrid_score[(field, crop)] * area
            result_table.append(row)
        st.dataframe(result_table)

        st.success(f" Total Hybrid Score (Optimized Objective): {total_score:,.2f}")
    else:
        st.error(" Optimization failed. Please revise your inputs.")

