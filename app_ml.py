import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Sugarcane Smart Irrigation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path("data/sensor_data.csv")
MODEL_PATH = Path("models/irrigation_model.joblib")

# --------------------------------------------------
# ADVANCED DARK GREEN THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top right, #12352b 0%, transparent 42%),
        linear-gradient(135deg, #07130f, #0b1715 55%, #07110e);
    color: #edf7f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07140f, #0c2119);
    border-right: 1px solid #244a39;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p {
    color: #e5f5e9;
}

h1, h2, h3 {
    color: #f2fff5 !important;
    letter-spacing: -0.5px;
}

p, label, span {
    color: inherit;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 22px;
    border: 1px solid #285b43;
    background: linear-gradient(115deg, #102e22, #10231f 65%, #173d2a);
}

.hero-tag {
    color: #6ee7a2;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(27px, 3vw, 39px);
    margin: 8px 0;
    font-weight: 800;
}

.hero p {
    color: #bad4c4;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    margin: 12px 0 15px;
    color: #e9fff0;
}

.metric-card {
    padding: 19px;
    border-radius: 16px;
    background: linear-gradient(145deg, #142b23, #101d1a);
    border: 1px solid #294c3c;
    min-height: 112px;
}

.metric-label {
    color: #a8c5b4;
    font-size: 12px;
    margin-bottom: 9px;
}

.metric-value {
    color: #72e6a0;
    font-size: 27px;
    font-weight: 800;
}

.panel {
    background: rgba(17, 38, 30, 0.92);
    border: 1px solid #294c3c;
    padding: 21px;
    border-radius: 17px;
    margin-bottom: 15px;
}

.advisory-good {
    border: 1px solid #36c878;
    background: #102e21;
    border-radius: 15px;
    padding: 20px;
}

.advisory-warn {
    border: 1px solid #e9b949;
    background: #302817;
    border-radius: 15px;
    padding: 20px;
}

.advisory-rain {
    border: 1px solid #60a5fa;
    background: #142b40;
    border-radius: 15px;
    padding: 20px;
}

.advisory-title {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 7px;
}

.small-note {
    color: #9eb9aa;
    font-size: 12px;
}

.stButton > button {
    background: linear-gradient(100deg, #20a65a, #38d77c);
    color: #06170d;
    font-weight: 800;
    border: none;
    border-radius: 11px;
    padding: 12px 18px;
    min-height: 48px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #6ee7a2;
    color: #06170d;
    border: none;
    transform: translateY(-1px);
}

div[data-baseweb="select"] > div {
    background-color: #14251e;
    border-color: #355745;
}

[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 8px;
}

hr {
    border-color: #294537;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------
st.sidebar.markdown("## 🌱 SUGARSMART")
st.sidebar.caption("Precision Irrigation Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Dataset Overview",
        "Model Insights",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### System Status")
st.sidebar.success("Application interface active")
st.sidebar.caption("Data source: synthetic demonstration dataset")

if not DATA_PATH.exists():
    st.error("Dataset missing. Run: python generate_data.py")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model missing. Run: python train_model.py")
    st.stop()

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# REUSABLE HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">Smart Agriculture • Machine Learning</div>
    <h1>🌱 Sugarcane Irrigation Advisory</h1>
    <p>Sensor-driven insights for smarter irrigation decisions
    and responsible water management.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
if page == "Dashboard":

    st.markdown('<div class="section-title">Farm & Sensor Control Center</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        plot_id = st.selectbox(
            "📍 Select Farm Plot",
            data["plot_id"].tolist()
        )

        soil_moisture = st.slider(
            "💧 Soil Moisture (%)", 0, 100, 20
        )

        temperature = st.slider(
            "🌡️ Temperature (°C)", 0, 50, 32
        )

    with right:
        humidity = st.slider(
            "🌬️ Humidity (%)", 0, 100, 60
        )

        rainfall = st.slider(
            "🌧️ Recent Rainfall (mm)", 0.0, 50.0, 0.0, 0.5
        )

        crop_stage = st.selectbox(
            "🌿 Sugarcane Crop Stage",
            [
                "Germination",
                "Tillering",
                "Grand Growth",
                "Maturity"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_prediction = st.button(
        "✦  GENERATE IRRIGATION ADVISORY",
        use_container_width=True
    )

    if run_prediction:
        input_data = pd.DataFrame([{
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall_mm": rainfall,
            "crop_stage": crop_stage
        }])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        probability = float(probabilities[classes.index(1)])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Your Irrigation Advisory</div>',
            unsafe_allow_html=True
        )

        advisory_col, probability_col = st.columns([1.5, 1])

        with advisory_col:
            if rainfall >= 5:
                st.markdown("""
                <div class="advisory-rain">
                    <div class="advisory-title">🌧️ Rainfall Caution</div>
                    Recent rainfall is recorded. Reassess soil and
                    field conditions before irrigating.
                </div>
                """, unsafe_allow_html=True)

            elif prediction == 1:
                st.markdown("""
                <div class="advisory-warn">
                    <div class="advisory-title">💧 Irrigation Potentially Needed</div>
                    The Random Forest model flags irrigation as
                    potentially needed under the entered conditions.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="advisory-good">
                    <div class="advisory-title">✓ Irrigation Not Flagged</div>
                    The model does not currently flag irrigation
                    as necessary under its learned synthetic patterns.
                </div>
                """, unsafe_allow_html=True)

        with probability_col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MODEL-ESTIMATED IRRIGATION-NEEDED PROBABILITY</div>
                <div class="metric-value">{probability:.1%}</div>
                <div class="small-note">Based on synthetic training labels</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Current Field Conditions")

        metric_cols = st.columns(4)

        metrics = [
            ("Soil Moisture", f"{soil_moisture}%", "💧"),
            ("Temperature", f"{temperature} °C", "🌡️"),
            ("Humidity", f"{humidity}%", "🌬️"),
            ("Rainfall", f"{rainfall:.1f} mm", "🌧️")
        ]

        for col, (label, value, emoji) in zip(metric_cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{emoji} {label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(
            f"Plot: {plot_id}  |  Crop stage: {crop_stage}"
        )

    st.divider()

    st.markdown('<div class="section-title">Farm Dataset at a Glance</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    summary_metrics = [
        (c1, "TOTAL RECORDS", f"{len(data):,}"),
        (c2, "UNIQUE PLOTS", f"{data['plot_id'].nunique():,}"),
        (c3, "CROP STAGES", f"{data['crop_stage'].nunique()}"),
        (c4, "SENSOR FEATURES", "5")
    ]

    for col, label, value in summary_metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, table_col = st.columns([1, 1.4])

    with chart_col:
        st.markdown("### Soil Moisture Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#10221b")
        ax.set_facecolor("#10221b")
        ax.hist(
            data["soil_moisture"],
            bins=20,
            color="#39ce79",
            edgecolor="#10221b"
        )
        ax.set_xlabel("Soil Moisture (%)", color="white")
        ax.set_ylabel("Number of Records", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#476554")
        ax.spines["left"].set_color("#476554")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with table_col:
        st.markdown("### Recent Sensor Records")
        st.dataframe(
            data.head(8),
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Educational prototype: data is synthetic and the model's "
        "predictions have not been validated using real farm measurements."
    )

# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------
elif page == "Dataset Overview":

    st.markdown("## 📊 Dataset Overview")
    st.write(
        "This prototype uses artificially generated sensor records "
        "to demonstrate the data-processing and ML workflow."
    )

    st.markdown("### Dataset Preview")
    st.dataframe(data, use_container_width=True, hide_index=True)

    st.markdown("### Numerical Summary")
    st.dataframe(
        data.select_dtypes(include="number").describe().round(2),
        use_container_width=True
    )

    st.markdown("### Irrigation Label Distribution")
    label_counts = data["irrigation_needed"].value_counts().sort_index()
    label_counts.index = [
        "No irrigation flagged" if x == 0 else "Irrigation flagged"
        for x in label_counts.index
    ]
    st.bar_chart(label_counts)

    st.caption(
        "The target labels were generated by a predefined rule, "
        "not by expert-labelled field observations."
    )

# --------------------------------------------------
# MODEL PAGE
# --------------------------------------------------
elif page == "Model Insights":

    st.markdown("## 🧠 Model Insights")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Pipeline:** Numeric scaling + crop-stage encoding + classifier")
    st.write("**Training split:** 80%")
    st.write("**Testing split:** 20%")
    st.write("**Training samples:** 800")
    st.write("**Testing samples:** 200")

    st.markdown("### Reported Test Results")
    st.metric("Synthetic test accuracy", "100%")

    st.warning(
        "This accuracy applies only to the held-out portion of the "
        "synthetic, rule-generated dataset. It is not evidence of "
        "100% accuracy on real agricultural data."
    )

    st.markdown("### Input Features")
    st.write(
        "- Soil moisture (%)\n"
        "- Temperature (°C)\n"
        "- Humidity (%)\n"
        "- Recent rainfall (mm)\n"
        "- Sugarcane crop stage"
    )

    st.markdown("### Important Limitation")
    st.write(
        "The model learns patterns from labels created by a predefined "
        "rule. It requires real sensor data and agronomic validation "
        "before being considered for real irrigation decisions."
    )

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About Project":

    st.markdown("## 🌾 About the Project")

    st.write(
        "**Project title:** Irrigation Advisory System for Sugarcane "
        "Crop Using AI and Sensor-Based Technology"
    )

    st.write(
        "The project demonstrates how sensor-like inputs can be "
        "processed by a machine-learning pipeline and displayed "
        "through an interactive decision-support dashboard."
    )

    st.markdown("### System Workflow")
    st.code(
        "Sensor inputs\n"
        "     ↓\n"
        "Data preparation\n"
        "     ↓\n"
        "Random Forest model\n"
        "     ↓\n"
        "Irrigation advisory\n"
        "     ↓\n"
        "Streamlit dashboard",
        language="text"
    )

    st.markdown("### Technologies Used")
    st.write(
        "Python · Streamlit · Pandas · Scikit-learn · "
        "Random Forest · Joblib · Matplotlib"
    )

    st.markdown("### Future Improvements")
    st.write(
        "- Connect real soil-moisture sensors.\n"
        "- Collect expert-validated irrigation labels.\n"
        "- Integrate trusted local weather data.\n"
        "- Evaluate the model across crop stages and seasons.\n"
        "- Deploy through Docker and a cloud environment."
    )

    st.info(
        "This is an academic demonstration, not a substitute "
        "for advice from an agricultural expert."
    )

st.sidebar.divider()
st.sidebar.caption("SugarSmart • Academic Cloud Computing Prototype")
