import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Sugarcane Irrigation Advisory",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Sugarcane Irrigation Advisory System")
st.caption("AI and sensor-based irrigation decision support")
st.info(
    "Educational prototype using synthetic sensor data. "
    "Recommendations are illustrative, not validated agricultural advice."
)

# Load the generated sensor dataset
data_path = Path("data/sensor_data.csv")

if data_path.exists():
    data = pd.read_csv(data_path)
else:
    st.error("Sensor data not found. Run: python generate_data.py")
    st.stop()

st.subheader("Farm and Sensor Inputs")

left, right = st.columns(2)

with left:
    plot_id = st.selectbox("Select Plot ID", data["plot_id"].tolist())
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 25)
    temperature = st.slider("Temperature (°C)", 0, 50, 32)

with right:
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    rainfall = st.slider("Recent Rainfall (mm)", 0.0, 50.0, 2.0, 0.5)
    crop_stage = st.selectbox(
        "Sugarcane Crop Stage",
        ["Germination", "Tillering", "Grand Growth", "Maturity"]
    )

st.divider()
st.subheader("Irrigation Advisory")

if st.button("Generate Advisory", type="primary"):
    # Transparent rule-based baseline for this first prototype.
    if soil_moisture < 30 and rainfall < 5:
        st.warning("Irrigation may be needed under the prototype's rules.")
        st.write("**Suggested action:** Review soil conditions and irrigation availability.")
    elif rainfall >= 5:
        st.info("Recent rainfall is recorded. Reassess before irrigating.")
    else:
        st.success("The prototype rules do not currently flag irrigation as necessary.")

    st.write(f"**Plot:** {plot_id}")
    st.write(f"**Crop stage:** {crop_stage}")
    st.write(f"**Temperature:** {temperature} °C")
    st.write(f"**Humidity:** {humidity}%")
    st.write(f"**Soil moisture:** {soil_moisture}%")
    st.write(f"**Recent rainfall:** {rainfall} mm")

st.divider()
st.subheader("Synthetic Sensor Dataset")
st.dataframe(data.head(10), use_container_width=True)

st.metric("Total Sample Records", len(data))
st.bar_chart(data["soil_moisture"])

st.caption(
    "Current version uses a rule-based baseline, not a trained ML model. "
    "Thresholds are illustrative and require agronomic validation."
)