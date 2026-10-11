# SugarSmart — Sugarcane Irrigation Advisory System

## Use Case
KJS-AGR-01: Irrigation Advisory System for Sugarcane Crop using AI and Sensor-Based Technology.

## Project Description
SugarSmart is an academic prototype that uses synthetic sensor data and a Random Forest classifier to generate an irrigation advisory through a Streamlit dashboard.

Inputs include:
- Soil moisture
- Temperature
- Humidity
- Recent rainfall
- Sugarcane crop stage
- Farm plot selection

## Technologies
- Python
- Streamlit
- pandas
- scikit-learn
- joblib
- Docker and Docker Hub
- Kubernetes
- CloudAnalyst

## Run Locally
Install dependencies:

python -m pip install -r requirements.txt

Start the dashboard:

python -m streamlit run app.py

## Docker Image
atharvbadhe01/sugarsmart-irrigation:v2

The recorded local container used host port 8502 and container port 8501.

## Kubernetes
Deployment: sugarsmart-deployment
Service: sugarsmart-service
Replicas: 2
NodePort: 30085

The application was also accessed using:

kubectl port-forward service/sugarsmart-service 8501:8501

Keep the terminal open while using the forwarded URL.

## Limitations
The dataset and target labels are synthetic. Model predictions are educational demonstrations and have not been validated for real agricultural decisions.

CloudAnalyst results are simulated. Docker resource observations were collected locally and do not establish a measured cloud-cost saving.

## Academic Acknowledgement
AI assistance was used to help prepare parts of the code and documentation. The implementation and results should be reviewed and verified by the student.