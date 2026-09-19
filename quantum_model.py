import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# -----------------------------------------
# 1. Load QureAI disease dataset
# -----------------------------------------

data = pd.read_csv(
    "datasets/disease_prediction/Training.csv"
)

# Remove unwanted column
data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]

X = data.drop("prognosis", axis=1)
y = data["prognosis"]


# -----------------------------------------
# 2. Select 4 important symptoms
# -----------------------------------------

selector = SelectKBest(
    score_func=chi2,
    k=4
)

X_selected = selector.fit_transform(X, y)

selected_features = X.columns[
    selector.get_support()
].tolist()

print("\nSelected quantum features:")
for feature in selected_features:
    print("-", feature)


# -----------------------------------------
# 3. Scale features for quantum circuit
# -----------------------------------------

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_quantum = scaler.fit_transform(X_selected)


# -----------------------------------------
# 4. Use a smaller dataset
#    for quantum demonstration
# -----------------------------------------

quantum_data = pd.DataFrame(
    X_quantum,
    columns=selected_features
)

quantum_data["prognosis"] = y.values

quantum_data = (
    quantum_data
    .groupby("prognosis", group_keys=False)
    .head(10)
)

X_quantum = quantum_data[selected_features].values
y_quantum = quantum_data["prognosis"].values


# -----------------------------------------
# 5. Create quantum feature map
# -----------------------------------------

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)


# -----------------------------------------
# 6. Create quantum kernel
# -----------------------------------------

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)


# -----------------------------------------
# 7. Create Quantum SVM
# -----------------------------------------

quantum_model = QSVC(
    quantum_kernel=quantum_kernel
)


# -----------------------------------------
# 8. Train Quantum ML model
# -----------------------------------------

print("\nTraining QureAI Quantum ML model...")
print("Training samples:", len(X_quantum))

quantum_model.fit(
    X_quantum,
    y_quantum
)


# -----------------------------------------
# 9. Save quantum model
# -----------------------------------------

joblib.dump(
    quantum_model,
    "backend/quantum_model.pkl"
)

joblib.dump(
    scaler,
    "backend/quantum_scaler.pkl"
)

joblib.dump(
    selector,
    "backend/quantum_selector.pkl"
)

joblib.dump(
    selected_features,
    "backend/quantum_features.pkl"
)


print("\n===================================")
print("QureAI Quantum ML model trained!")
print("===================================")

print("\nQuantum features:")
print(selected_features)

print("\nModel saved successfully.")