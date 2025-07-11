import streamlit as st
from pulp import *
import pandas as pd

st.set_page_config(page_title="Field-Aware Crop Optimizer", layout="wide")
st.title("🌾 Field-Aware Sustainable Crop Planning Optimizer")
st.markdown("This system extends our crop planning model to consider multiple fields with different soil, water and sustainability profiles.")

# Crops and Fields
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

# Fixed realistic crop parameters (per hectare)
yield_per_ha = {'Wheat': 3.2, 'Rice': 4.5, 'Maize': 3.8, 'Soyabean': 2.5, 'Cotton': 2.2}
price_per_ton = {'Wheat': 3000, 'Rice': 2800, 'Maize': 2700, 'Soyabean': 3500, 'Cotton': 4000}
fertilizer_per_ha = {'Wheat': 100, 'Rice': 120, 'Maize': 90, 'Soyabean': 80, 'Cotton': 110}
water_per_ha = {'Wheat': 1200000, 'Rice': 1800000, 'Maize': 1000000, 'Soyabean': 1100000, 'Cotton': 1500000}
fertilizer_cost = 25

# Field inputs
st.subheader("📋 Field Characteristics")
field_area = {}
field_water = {}
rainfall_index = {}

for field in fields:
    st.markdown(f"**{field}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        field_area[field] = st.number_input(f"Area of {field} (ha)", min_value=0, value=1000, step=100, key=f"area_{field}")
    with col2:
        field_water[field] = st.number_input(f"Water in {field} (liters)", min_value=0, value=150_000_000, step=1_000_000, key=f"water_{field}")
    with col3:
        rainfall_index[field] = st.slider(f"Rainfall Index {field}", 0.0, 1.0, 0.75, key=f"rain_{field}")

# Weight sliders
st.sidebar.header("⚖️ Weight Parameters")
alpha = st.sidebar.slider("α: Water Scarcity", 0.0, 1.0, 0.5)
beta = st.sidebar.slider("β: Fertilizer Use", 0.0, 1.0, 0.3)
gamma = st.sidebar.slider("γ: Profit Importance", 0.0, 1.0, 0.7)

if st.button("🚀 Run Optimization"):
    model = LpProblem("Field_Aware_Crop_Planning", LpMaximize)
    land = LpVariable.dicts("Land", ((field, crop) for field in fields for crop in crops), lowBound=0, cat='Continuous')
    
    hybrid_score = {}

    # Compute score based on formula
    for field in fields:
        for crop in crops:
            Y = yield_per_ha[crop]
            P = price_per_ton[crop]
            F = fertilizer_per_ha[crop]
            W = water_per_ha[crop]
            RI = rainfall_index[field]

            profit = -(Y * P) - (F * fertilizer_cost)
            sustain = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))
            hybrid_cost = gamma * profit + sustain
            hybrid_score[(field, crop)] = -hybrid_cost  # Convert cost to score

    # Objective: maximize hybrid score
    model += lpSum([hybrid_score[(f, c)] * land[(f, c)] for f in fields for c in crops])

    # Constraints
    for f in fields:
        model += lpSum([land[f, c] for c in crops]) <= field_area[f]
        model += lpSum([land[f, c] * water_per_ha[c] for c in crops]) <= field_water[f]

    model.solve()

    if model.status == 1:
        st.subheader("📊 Optimal Land Allocation (in ha)")
        result_table = []

        total_score = 0
        for f in fields:
            row = {'Field': f}
            for c in crops:
                area = land[f, c].varValue or 0
                row[c] = round(area, 2)
                total_score += hybrid_score[(f, c)] * area
            result_table.append(row)

        df = pd.DataFrame(result_table)
        st.dataframe(df.style.format(precision=2), use_container_width=True)

        st.success(f" Total Hybrid Score (Optimized Objective): **{total_score:,.2f}**")
    else:
        st.error(" Optimization failed. Try adjusting rainfall, water, or weights.")
