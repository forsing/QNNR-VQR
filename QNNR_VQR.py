# Quantum Neural Network Regressor (QNN Regressor)
# Variational Quantum Regressor (VQR)

"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.visualization import circuit_drawer

from IPython.display import display
from IPython.display import clear_output


from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit_aer import Aer
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms import VQR
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from qiskit_machine_learning.optimizers import GradientDescent

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Load Loto data
# =========================
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto7_4512_k90.csv", header=None)

print()
print("Prvih 5 kombinacija:")
print(df.head())
print()
print("Zadnjih 5 kombinacija:")
print(df.tail())
print()
"""
Prvih 5 kombinacija:
    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37

Zadnjih 5 kombinacija:
       0   1   2   3   4   5   6
4507   5   9  10  13  16  28  34
4508  10  13  14  21  30  32  39
4509   2   7  11  23  26  38  39
4510   1   7  20  24  26  35  37
4511   2  11  20  23  33  34  35
"""

# Minimalni i maksimalni brojevi po poziciji
min_val = [1,2,3,4,5,6,7]
max_val = [33,34,35,36,37,38,39]

# Mapiranje na indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        if not df_indexed[i].between(0, max_val[i]-min_val[i]).all():
            raise ValueError(f"Kolona {i} nije u opsegu")
    return df_indexed

df_indexed = map_to_indexed_range(df, min_val, max_val)

# Poslednjih NN kombinacija
NN = 4512 # sve kombinacije
# NN = 10   # poslednjih 10 kombinacija za brzi test
df_indexed = df.tail(NN).reset_index(drop=True)
df_indexed = df_indexed.iloc[:, :7]

# Kreiranje X i y
X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values

# Za klasifikaciju: prva kolona kao labela
y_single = y_x[:,0]

# Split train/test
X_train_x, X_test_x, y_train_x_all, y_test_x_all = train_test_split(
    X_x, y_x, test_size=0.25, random_state=SEED
)
y_train_x = y_train_x_all[:,0]
y_test_x = y_test_x_all[:,0]

# Skaliranje
scaler_X = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train_x)
X_test_s = scaler_X.transform(X_test_x)

y_scalers = [MinMaxScaler().fit(y_x[:,i].reshape(-1,1)) for i in range(7)]
y_train_s_all = np.column_stack([y_scalers[i].transform(y_train_x_all[:,i].reshape(-1,1)).flatten() for i in range(7)])
y_test_s_all = np.column_stack([y_scalers[i].transform(y_test_x_all[:,i].reshape(-1,1)).flatten() for i in range(7)])

# =========================
# Feature map + ansatz
# =========================
N_QUBITS = 7
REPS = 2

feat_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=REPS, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=REPS, entanglement='linear')

# Draw circuits
feat_map.decompose().draw('mpl')
plt.show()
ansatz.decompose().draw('mpl')
plt.show()

# VQC i VQR optimizatori
init = 0.05*np.ones(ansatz.num_parameters)
optimizer = COBYLA(maxiter=300)


vqr = VQR(
    feature_map=feat_map,
    ansatz=ansatz,
    optimizer=optimizer,
    loss="cross_entropy",
    initial_point=init
)
print()
print("VQR built ✅", "params:", ansatz.num_parameters)
print()
"""
VQR built ✅ params: 21
"""

# =========================
# Quantum Neural Network Regressor
# =========================
def build_qnn_regressor(X_train, y_train):
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(feat_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        estimator=estimator,
        circuit=qc,
        input_params=list(feat_map.parameters),
        weight_params=list(ansatz.parameters)
    )

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        loss="squared_error",
        optimizer=L_BFGS_B(maxiter=10)
    )


    regressor.fit(X_train, y_train)
    return regressor


gradient = GradientDescent()  # param-shift rule

regressorQNNR = build_qnn_regressor(X_train_s, y_train_x_all[:,0])
print()
print("QNN Regressor built and trained ✅")
print()
"""
QNN Regressor built and trained ✅
"""

# Predikcija VQR sledece kombinacije
X_last_scaled = scaler_X.transform(df_indexed.values[-1].reshape(1,-1))
predicted_combination = []
for i in range(7):

    y_pred_scaled = regressorQNNR.predict(X_last_scaled)
    y_pred = y_scalers[i].inverse_transform(y_pred_scaled.reshape(-1,1))[0,0]
    y_pred = max(1, int(round(y_pred)))
    predicted_combination.append(y_pred)

print("\n=== Predvidjena QNNR sledeca loto kombinacija (7) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
print()
"""
NN=10
=== Predvidjena QNNR sledeca loto kombinacija (7) ===
1 5 8 10 13 26 33


NN=4512

"""



# =========================
# Main
# =========================
def main():

    # Train VQR regressor za 7 brojeva
    print("Training Variational Quantum Regressor ...")
    for i in range(7):
        vqr.fit(X_train_s, y_train_x_all[:,i])
    print("VQR training completed.")
    """
    Training Variational Quantum Regressor ...
    VQR training completed.
    """

    # Predikcija VQR sledece kombinacije
    X_last_scaled = scaler_X.transform(df_indexed.values[-1].reshape(1,-1))
    predicted_combination = []
    for i in range(7):
        y_pred_scaled = vqr.predict(X_last_scaled)
        y_pred = y_scalers[i].inverse_transform(y_pred_scaled.reshape(-1,1))[0,0]
        y_pred = max(1, int(round(y_pred)))
        predicted_combination.append(y_pred)

    print("\n=== Predvidjena VQR sledeca loto kombinacija (7) ===")
    print(" ".join(str(num) for num in predicted_combination))
    print()
    print()
    """
    NN=10
    === Predviđena VQR sledeca loto kombinacija (7) ===
    2 9 12 15 18 29 35

    NN=4512
    === Predviđena VQR sledeca loto kombinacija (7) ===
    3 4 5 7 8 11 14
    """


if __name__ == "__main__":
    main()
