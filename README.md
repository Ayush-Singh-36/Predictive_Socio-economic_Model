💸 Predictive Socio-Economic Model & Serving PipelineAn end-to-end, production-grade machine learning application designed to predict whether an individual's annual income exceeds $50,000 USD based on census demographics. The system features an optimized data preprocessing pipeline, a trained classification model, a FastAPI serving backend, and an interactive Streamlit user interface—all containerized using a single Docker architecture.

📸 Application InterfaceBelow is the user interface designed for collecting demographic attributes and delivering real-time predictions directly from the containerized machine learning microservice:

![Streamlit Frontend Workspace](assets/app_preview.png)

📊 Model Evaluation & MetricsThe core engine relies on a K-Neighbors Classifier (KNN), optimized using an automated data encoding and scaling workflow.Performance SummaryClassification Accuracy: 83.24% (or your exact calculated accuracy score)Data Split: 80% Training Set / 20% Evaluation Testing SetPreprocessing Pipeline: Categorical inputs are handled dynamically via OneHotEncoder(handle_unknown="ignore"), while all resulting inputs are normalized via StandardScaler to optimize spatial distance calculation accuracy.

Detailed Classification 

       Report      precision    recall  f1-score   support

       <=50K       0.86      0.92      0.89      4912
        >50K       0.68      0.54      0.60      1601

    accuracy                           0.83      6513
   macro avg       0.77      0.73      0.75      6513
weighted avg       0.82      0.83      0.82      6513

🎛️ Architectural Features & Pipeline InputsThe model processes 14 distinctive attributes spanning personal, professional, and financial socio-economic sectors:CategoryFeature NameDescription / Data TypeDemographicsAgeGenderRaceNative CountryNumerical & Categorical features controlling baseline cultural and age distributions.EmploymentWorkclassOccupationHours Per WeekOperational employment sectors (e.g., Private, Self-Employed) paired with workload metrics.EducationEducationEducational NumCategorical highest degree paired with continuous years of formal education completed (1–16).Family StatusMarital StatusRelationshipHousehold status indicators mapping family structures and responsibilities.FinancialsCapital GainCapital LossfnlwgtContinuous monetary investment metrics alongside population control sampling weights.

🚀 Quick Start & DeploymentThis application is completely containerized. You do not need to install Python libraries locally; Docker manages the full system footprint automatically.1. Build the Docker ImageCompile the application layers using the optimized no-cache workflow to ensure all assets are updated cleanly:Bashdocker build --no-cache -t income-prediction-app .

2. Launch the MicroservicesSpin up the detached container, exposing port 8000 for the backend API and port 8510 for the interactive dashboard:Bashdocker run -d -p 8000:8000 -p 8510:8501 --name income_app_instance income-prediction-app

3. Access the Active EndpointsInteractive Web UI Dashboard: Open http://localhost:8510FastAPI Engine Documentation: Open http://localhost:8000/docs🛠️ Tech Stack & StructureCore Logic: Python 3.11, Scikit-Learn, Pandas, NumPyServing Layer: FastAPI (Uvicorn Async Worker Server Engine)Frontend Engine: StreamlitInfrastructure: Docker (Linux-slim distribution base) utilizing an automatic multi-process shell coordinator (start.sh).