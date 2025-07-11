import streamlit as st
from pulp import *

st.set_page_config(page_title="Crop Planner Debug", layout="wide")
st.title(" Hybrid Cost Debug – Crop Optimization")

# Crops and Fields
crops = ['Wheat', 'Rice']
fields = ['Field A', 'Field B']

# Fixed parameters
yield_per_ha = {'Wheat': 3.0, 'Rice': 4.0}
price_per_ton = {'Wheat': 3000, 'Rice': 2800}
fertilizer_per_ha = {'Wheat': 100, 'Rice': 120}
water_per_ha = {'Wheat': 1_200_000, 'Rice': 1_800_000}
fertilizer_cost = 25

field_area = {'Field A': 1500, 'Field B': 800}
field_water = {'Field A': 140_000_000, 'Field B': 153_000_000}
rainfall_index = {'Field A': st.slider("Rainfall Index – Field A", 0.0, 1.0, 0.8),
                  'Field B': st.slider("Rainfall Index – Field B", 0.0, 1.0, 0.75)}

# Weight sliders
alpha = st.sidebar.slider("α: Water Scarcity Weight", 0.0, 1.0, 0.5)
beta = st.sidebar.slider("β: Fertilizer Usage Weight", 0.0, 1.0, 0.3)
gamma = st.sidebar.slider("γ: Profit Weight", 0.0, 1.0, 0.7)

if st.button("Run Optimization"):
    model = LpProblem("CropHybridModel", LpMaximize)
    land = LpVariable.dicts("Land", ((f, c) for f in fields for c in crops), lowBound=0, cat='Continuous')

    hybrid_score = {}

    for f in fields:
        for c in crops:
            Y = yield_per_ha[c]
            P = price_per_ton[c]
            F = fertilizer_per_ha[c]
            W = water_per_ha[c]
            RI = rainfall_index[f]

            profit = -(Y * P) - (F * fertilizer_cost)
            sustain = beta * (F * fertilizer_cost) + alpha * (W * (1 - RI))
            hybrid_cost = gamma * profit + sustain
            hybrid_score[(f, c)] = -hybrid_cost

            # Debug
            st.write(f"**{f}-{c}** | Profit = ₹{profit:.2f}, Sustain = ₹{sustain:.2f}, Hybrid = ₹{hybrid_cost:.2f}, Score = {hybrid_score[(f, c)]:.2f}")

    model += lpSum([hybrid_score[(f, c)] * land[(f, c)] for f in fields for c in crops])

    for f in fields:
        model += lpSum([land[f, c] for c in crops]) <= field_area[f]
        model += lpSum([land[f, c] * water_per_ha[c] for c in crops]) <= field_water[f]

    model.solve()

    if model.status == 1:
        st.markdown("### Optimal Land Allocation")
        total_score = 0
        for f in fields:
            for c in crops:
                val = land[f, c].varValue or 0
                total_score += hybrid_score[(f, c)] * val
                st.write(f"{f}-{c}: {val:.2f} ha")
        st.success(f"Total Hybrid Score: {total_score:,.2f}")
    else:
        st.error("Optimization failed.")
