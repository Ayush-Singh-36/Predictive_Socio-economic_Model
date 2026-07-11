# Importing all the libraries and dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle as pk

# looking at dataset for once
data = pd.read_csv("Predictive_Socio-economic_Model\\data\\adult.csv")
data.info()

x = data.drop(columns = "income")
y = data["income"]

# seperating text and numerical columns form x
categorical_columns = x.select_dtypes(include = str)
numerical_columns = x.select_dtypes(exclude = str)

# process categorical columns if any exist; otherwise handle cleanly
if not categorical_columns.empty:
    encoder = OneHotEncoder(sparse_output = False, handle_unknown = "ignore")
    encoded_categorical_data = encoder.fit_transform(categorical_columns)
    x_combined = np.hstack((encoded_categorical_data, numerical_columns.values))

else:
    encoder = None
    x_combined = numerical_columns.values

#scaling the newly encoded values
scaler = StandardScaler()
scaled_values = scaler.fit_transform(x_combined)

# splitting data into train & test set
x_train, x_test, y_train, y_test = train_test_split(scaled_values, y, test_size=0.2, random_state = 42)

# training the model
model = KNeighborsClassifier()
model.fit(x_train, y_train)

#predictiong output from trained model
y_pred = model.predict(x_test)

# evaluating performance using regression metrices
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:- {accuracy * 100.:2f}\n")
report = classification_report(y_test, y_pred)
print("Classification Report\n")
print(report)

# Create a dictionary of the artifacts you need for deployment
model_artifacts = {
    "encoder": encoder,
    "scaler": scaler,
    "model": model,
    "metrices": {
        "accuracy": accuracy,
        "report": report
    }
}

# Defining the file path to save your artifacts
file_path = "Predictive_Socio-economic_Model\\predictive_socio-economic_model_artifacts.pkl"

# write the artifacts to a file using pickle
with open(file_path, "wb") as file:
    pk.dump(model_artifacts, file)

print(f"Model saved successfully to {file_path}")