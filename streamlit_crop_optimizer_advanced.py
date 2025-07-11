import streamlit as st
from pulp import *
import pandas as pd

st.set_page_config(page_title="Crop Optimizer", layout="wide")
st.title("🌾 Field-Aware Sustainable Crop Planning Optimizer")
st.markdown("This system optimizes crop planning across multiple fields based on profit, water availability, and fertilizer usage.")

# Crops and Fields
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

# Crop parameters (per hectare)
yield_per_ha = {'Wheat': 3.2, 'Rice': 4.5, 'Maize': 3.8, 'Soyabean': 2.5, 'Cotton': 2.2}
price_per_ton = {'Wheat': 3000, 'Rice': 2800, 'Maize': 2700, 'Soyabean': 3500, 'Cotton': 4000}
fertilizer_per_ha = {'Wheat': 100, 'Rice': 120, 'Maize': 90, 'Soyabean': 80, 'Cotton': 110}
water_per_ha = {'Wheat': 1200000, 'Rice': 1800000, 'Maize': 1000000, 'Soyabean': 1100000, 'Cotton': 1500000}
fertilizer_cost = 25  # ₹/kg

# Sidebar Weights
st.sidebar.header("⚖️ Weight Parameters")
alpha = st.sidebar.slider("Water Scarcity Weight (α)", 0.0, 1.0, 0.2)
beta = st.sidebar.slider("Fertilizer Usage Weight (β)", 0.0, 1.0, 0.1)
gamma = st.sidebar.slider("Profit Importance Weight (γ)", 0.0, 1.0, 0.9)

# Field Inputs
st.subheader("📋 Field Characteristics")
field_area = {}
field_water = {}
rainfall_index = {}

for field in fields:
    st.markdown(f"**{field}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        field_area[field] = st.number_input(f"Area (ha)", min_value=0, value=1000, step=100, key=f"area_{field}")
    with col2:
        field_water[field] = st.number_input(f"Water Available (liters)", min_value=0, value=200_000_000, step=10_000_000, key=f"water_{field}")
    with col3:
        rainfall_index[field] = st.slider(f"Rainfall Index (0–1)", 0.0, 1.0, 0.85, key=f"rain_{field}")

# Optimization Logic
if st.button("🚀 Run Optimization"):
    model = LpProblem("Crop_Optimization", LpMaximize)
    land = LpVariable.dicts("Land", ((f, c) for f in fields for c in crops), lowBound=0, cat='Continuous')

    hybrid_score = {}

    for field in fields:
        for crop in crops:
            Y = yield_per_ha[crop]
            P = price_per_ton[crop]
            F = fertilizer_per_ha[crop]
            W = water_per_ha[crop]
            RI = rainfall_index[field]

            # Compute components
            profit = Y * P
            sustain_penalty = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))

            # Final hybrid score
            hybrid_score[(field, crop)] = gamma * profit - sustain_penalty

    # Objective
    model += lpSum([hybrid_score[(f, c)] * land[(f, c)] for f in fields for c in crops])

    # Constraints
    for f in fields:
        model += lpSum([land[f, c] for c in crops]) <= field_area[f]
        model += lpSum([land[f, c] * water_per_ha[c] for c in crops]) <= field_water[f]

    model.solve()

    if model.status == 1:
        st.subheader("📊 Optimal Land Allocation (in ha)")
        result_data = []
        total_score = 0

        for f in fields:
            row = {'Field': f}
            for c in crops:
                area = land[f, c].varValue or 0
                row[c] = round(area, 2)
                total_score += hybrid_score[(f, c)] * area
            result_data.append(row)

        df = pd.DataFrame(result_data)
        st.dataframe(df.style.format(precision=2), use_container_width=True)

        st.success(f"✅ Total Hybrid Score: ₹{total_score:,.2f}")
    else:
        st.error("❌ Optimization failed. Try increasing water/rainfall or reducing weights.")
