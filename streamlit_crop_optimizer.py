
# Sustainable Crop Planning – Streamlit Web App
# Built by Vasavi Agarwal & Team

import streamlit as st
from pulp import *
import matplotlib.pyplot as plt

# App Title
st.title("🌾 Sustainable Crop Planning Optimizer")
st.markdown("Maximize profit and sustainability by optimizing land allocation for crops.")

# Input Section
st.sidebar.header("🛠️ Model Inputs")

# Crop selection
available_crops = ['Wheat', 'Rice', 'Maize']
crops = st.sidebar.multiselect("Select Crops", available_crops, default=available_crops)

# Input parameters
total_land = st.sidebar.slider("Total Land Available (ha)", 1000, 10000, 3000)
total_water = st.sidebar.slider("Total Water Available (mm)", 1000000, 10000000, 4200000)
total_labor = st.sidebar.slider("Total Labor Available (man-days)", 10000, 100000, 75000)
total_equipment = st.sidebar.slider("Total Equipment Hours Available", 5000, 50000, 20000)
alpha = st.sidebar.slider("Profit Weight (α)", 0.0, 1.0, 0.8)
beta = st.sidebar.slider("Sustainability Weight (β)", 0.0, 1.0, 0.2)

st.markdown("### 🔧 Enter Crop Parameters")

# Dynamic crop input
profit_per_ha = {}
water_per_ha = {}
labor_per_ha = {}
equipment_per_ha = {}
price_fluctuation = {}
weather_factor = {}
sustainability_score = {}

for crop in crops:
    st.subheader(f"🌱 {crop}")
    profit_per_ha[crop] = st.number_input(f"Profit per ha (₹) – {crop}", 5000, 25000, 9000)
    water_per_ha[crop] = st.number_input(f"Water per ha (mm) – {crop}", 500, 3000, 1000)
    labor_per_ha[crop] = st.number_input(f"Labor per ha (man-days) – {crop}", 10, 50, 25)
    equipment_per_ha[crop] = st.number_input(f"Equipment per ha (hours) – {crop}", 5, 20, 8)
    price_fluctuation[crop] = st.slider(f"Market Price Factor – {crop}", 0.8, 1.2, 1.0)
    weather_factor[crop] = st.slider(f"Weather Yield Factor – {crop}", 0.7, 1.0, 0.9)
    sustainability_score[crop] = st.slider(f"Sustainability Score (1–5) – {crop}", 1, 5, 3)

if st.button("🚀 Run Optimization"):

    # Build the model
    model = LpProblem("Sustainable_Crop_Planning", LpMaximize)
    land = LpVariable.dicts("LandAllocated", crops, lowBound=0)

    # Adjusted profit
    adjusted_profit = {
        crop: profit_per_ha[crop] * price_fluctuation[crop] * weather_factor[crop]
        for crop in crops
    }

    # Objective Function
    model += lpSum([
        alpha * adjusted_profit[crop] * land[crop] +
        beta * sustainability_score[crop] * land[crop]
        for crop in crops
    ]), "Combined_Objective"

    # Constraints
    model += lpSum([land[crop] for crop in crops]) <= total_land, "Land_Constraint"
    model += lpSum([water_per_ha[crop] * land[crop] for crop in crops]) <= total_water, "Water_Constraint"
    model += lpSum([labor_per_ha[crop] * land[crop] for crop in crops]) <= total_labor, "Labor_Constraint"
    model += lpSum([equipment_per_ha[crop] * land[crop] for crop in crops]) <= total_equipment, "Equipment_Constraint"

    # Solve
    model.solve()

    # Output Results
    st.markdown("## ✅ Results")
    if model.status == 1:
        land_alloc = {crop: land[crop].varValue for crop in crops}
        st.write("### Optimal Land Allocation (in ha):")
        st.write(land_alloc)

        total_profit = sum(adjusted_profit[crop] * land[crop].varValue for crop in crops)
        total_sustainability = sum(sustainability_score[crop] * land[crop].varValue for crop in crops)
        st.success(f"Total Adjusted Profit: ₹{total_profit:,.2f}")
        st.info(f"Sustainability Score (Weighted): {total_sustainability:.2f}")

        # Plotting
        fig, ax = plt.subplots()
        ax.bar(land_alloc.keys(), land_alloc.values(), color='mediumseagreen')
        ax.set_title("Land Allocation by Crop")
        ax.set_ylabel("Hectares")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        profit_vals = [adjusted_profit[crop] * land[crop].varValue for crop in crops]
        sustain_vals = [sustainability_score[crop] * land[crop].varValue for crop in crops]
        x = range(len(crops))
        ax2.bar(x, profit_vals, width=0.4, label='Profit', align='center', color='salmon')
        ax2.bar([i + 0.4 for i in x], sustain_vals, width=0.4, label='Sustainability', color='slateblue')
        ax2.set_xticks([i + 0.2 for i in x])
        ax2.set_xticklabels(crops)
        ax2.set_title("Profit vs Sustainability by Crop")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.error("❌ Optimization failed. Please check your inputs.")
