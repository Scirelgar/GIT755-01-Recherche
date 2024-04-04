import pennylane as qml

def getQuantumDevice(n_qubits):
    return qml.device("lightning.qubit", wires=n_qubits)