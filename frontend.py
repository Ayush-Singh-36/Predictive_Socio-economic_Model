# importing all the libraries and dependencies
import streamlit as st
import requests

# page configuration
st.set_page_config(
    page_title = "Income Size Predictor",
    page_icon = "💸",
    layout = "centered"
)

# main title
st.title("💵 Income Size Predictor")
st.markdown("Enter demographic data below to estimate if individual income exceeds **$50,000 USD**.")

API_URL = "http://127.0.0.1:8000/predict"

# defining categorical options based on the adult cencus dataset mapping
WORKCLASS_OPTIONS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
EDUCATION_OPTIONS = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
MARITAL_OPTIONS = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
OCCUPATION_OPTIONS = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-speciality", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
RELATIONSHIP_OPTIONS = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
RACE_OPTIONS = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
GENDER_OPTIONS = ["Male", "Female"]
COUNTRY_OPTIONS = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

# form layout for user inputs
with st.form(key = "prediction_form"):
    st.subheader("Personal Demographics")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value = 17, max_value = 100, value = 30, step = 1)
        gender = st.selectbox("Gender", options = GENDER_OPTIONS)
        race = st.selectbox("Race", options = RACE_OPTIONS)

    with col2:
        workclass = st.selectbox("WorkClass", options = WORKCLASS_OPTIONS)
        native_country = st.selectbox("Native Country", options = COUNTRY_OPTIONS)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value = 1, value = 180000, step = 1000, help = "Population sampling weight. Default average is fine.")

        st.markdown("---")
        st.subheader("Employment & Education")

    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox("Highest Education Level", options = EDUCATION_OPTIONS)
        educational_num = st.number_input("Years of Education Completed", min_value = 1, max_value = 16, value = 10, step = 1)
        occupation = st.selectbox("Occupation", options = OCCUPATION_OPTIONS)

    with col4:
        marital_status = st.selectbox("Marital Status", options = MARITAL_OPTIONS)
        relationship = st.selectbox("Relationship Status within Family", options = RELATIONSHIP_OPTIONS)
        hours_per_week = st.number_input("Working Hours Per Week", min_value = 1, max_value = 99, value = 40, step = 1)

    st.markdown("---")
    st.subheader("Financial Metrics")

    col5, col6 = st.columns(2)
    with col5:
        capital_gain = st.number_input("Capital Gains ($)", min_value = 0, value = 0, step = 100)
    with col6:
        capital_loss = st.number_input("Capital Losses ($)", min_value = 0, value = 0, step = 100)

    submit_button = st.form_submit_button(label = "Calculate Prediction")

if submit_button:
    payload = {
        "age": int(age),
        "workclass": str(workclass),
        "fnlwgt": int(fnlwgt),
        "education": str(education),
        "educational_num": int(educational_num),
        "marital_status": str(marital_status),
        "occupation": str(occupation),
        "relationship": str(relationship),
        "race": str(race),
        "gender": str(gender),
        "capital_gain": int(capital_gain),
        "capital_loss": int(capital_loss),
        "hours_per_week": int(hours_per_week),
        "native_country": str(native_country)
    }

    with st.spinner("Streaming request to processing engine..."):
        try:
            response = requests.post(API_URL, json = payload)

            if response.status_code == 200:
                data = response.json()
                label = data.get("prediction_label")
                confidence = data.get("confidence_score")
                code = data.get("prediction_code")

                st.success("Analysis Complete!")

                if code == 1:
                    st.balloons()
                    st.markdown(f"### Result: **{label}**")
                    st.metric(label = "Model Confidence Score", value = f"{confidence}%")
                else:
                    st.markdown(f"### Result: **{label}**")
                    st.metric(label = "Model Confidence Score", value = f"{confidence}%")

            else:
                st.error(f"Backend Server Error: {response.json().get('detail', 'Unknown issue')}")

        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI server. Make sure your Uvicorn instance is up and running on port 8000!")
        