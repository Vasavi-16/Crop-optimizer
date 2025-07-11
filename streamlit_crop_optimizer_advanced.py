import streamlit as st
from pulp import *
import pandas as pd

st.set_page_config(page_title="🌱 Simple Crop Optimizer", layout="wide")
st.title("🌾 Simple & Robust Field-Aware Crop Planner")

# === Crops & Fields ===
crops = ['Wheat', 'Rice', 'Maize', 'Soyabean', 'Cotton']
fields = ['Field A', 'Field B', 'Field C']

# === Fixed Crop Parameters (per hectare) ===
yield_per_ha = {'Wheat': 3.2, 'Rice': 4.5, 'Maize': 3.8, 'Soyabean': 2.5, 'Cotton': 2.2}
price_per_ton = {'Wheat': 3000, 'Rice': 2800, 'Maize': 2700, 'Soyabean': 3500, 'Cotton': 4000}
fertilizer_per_ha = {'Wheat': 100, 'Rice': 120, 'Maize': 90, 'Soyabean': 80, 'Cotton': 110}
water_per_ha = {'Wheat': 1_200_000, 'Rice': 1_800_000, 'Maize': 1_000_000, 'Soyabean': 1_100_000, 'Cotton': 1_500_000}
fertilizer_cost = 25

# === Default Weight Parameters ===
alpha = 0.2  # water penalty
beta = 0.1   # fertilizer penalty
gamma = 0.9  # profit importance

# === Field Input UI ===
st.subheader("📋 Field Inputs")
field_area, field_water, rainfall_index = {}, {}, {}

for f in fields:
    st.markdown(f"**{f}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        field_area[f] = st.number_input(f"Area (ha) - {f}", 100, 3000, 1000, step=100)
    with col2:
        field_water[f] = st.number_input(f"Water Available (L) - {f}", 100_000_000, 300_000_000, 200_000_000, step=10_000_000)
    with col3:
        rainfall_index[f] = st.slider(f"Rainfall Index - {f}", 0.0, 1.0, 0.85)

if st.button("🚀 Optimize Now"):
    model = LpProblem("SimpleCropPlan", LpMaximize)
    land = LpVariable.dicts("Land", ((f, c) for f in fields for c in crops), lowBound=0)

    hybrid_score = {}

    for f in fields:
        for c in crops:
            Y = yield_per_ha[c]
            P = price_per_ton[c]
            F = fertilizer_per_ha[c]
            W = water_per_ha[c]
            RI = rainfall_index[f]

            profit = Y * P
            penalty = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))
            hybrid_score[(f, c)] = gamma * profit - penalty

    model += lpSum(hybrid_score[(f, c)] * land[(f, c)] for f in fields for c in crops)

    for f in fields:
        model += lpSum(land[f, c] for c in crops) <= field_area[f]
        model += lpSum(land[f, c] * water_per_ha[c] for c in crops) <= field_water[f]

    model.solve()

    if model.status == 1:
        st.subheader("✅ Optimal Land Allocation (ha)")
        rows = []
        total_score = 0

        for f in fields:
            row = {'Field': f}
            for c in crops:
                area = land[f, c].varValue or 0
                row[c] = round(area, 2)
                total_score += hybrid_score[(f, c)] * area
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df.style.format(precision=2), use_container_width=True)
        st.success(f"Total Hybrid Score: ₹{total_score:,.2f}")
    else:
        st.error("❌ Optimization failed. Try increasing rainfall or water.")
