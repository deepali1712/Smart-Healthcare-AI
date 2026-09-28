import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("Smart Healthcare: Diabetes Risk Predictor")


# Load and Train model from public GitHub dataset
@st.cache_data
def train_model():
  url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
  df = pd.read_csv(url)

  features = ["Age", "BloodPressure", "Glucose", "BMI"]
  X = df[features]
  y = df["Outcome"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = RandomForestClassifier(n_estimators=50, random_state=42)
  model.fit(X_train, y_train)

  acc = accuracy_score(y_test, model.predict(X_test))
  return model, X_train, acc


model, X_train, accuracy = train_model()

# Sidebar
st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: **{accuracy * 100:.1f}%**")

st.sidebar.header("Patient Vitals")
age = st.sidebar.slider("Age", 21, 81, 33)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 122, 70)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 100)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)

input_df = pd.DataFrame(
    [[age, bp, glucose, bmi]], columns=["Age", "BloodPressure", "Glucose", "BMI"]
)

# Output prediction & Feature Importance
if st.button("Assess Health Risk"):
  pred = model.predict(input_df)[0]
  prob = model.predict_proba(input_df)[0][1]

  if pred == 1:
    st.error(f"High Diabetes Risk Detected (Risk Score: {prob*100:.1f}%)")
  else:
    st.success(f"Low Diabetes Risk Detected (Risk Score: {prob*100:.1f}%)")

  # Responsible AI: Feature Importance Visualization
  st.subheader("Responsible AI Analysis: Feature Importance")
  st.write("Shows which health metric impacts the AI's decisions the most:")

  importance = pd.Series(
      model.feature_importances_, index=X_train.columns
  ).sort_values()

  fig, ax = plt.subplots(figsize=(6, 3))
  importance.plot(kind="barh", color="teal", ax=ax)
  ax.set_xlabel("Importance Score")
  st.pyplot(fig)