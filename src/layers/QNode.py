import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

qml.enable_tape()

class QuantumLayer(nn.Module):  
    def __init__(self, size_in):
        super(QuantumLayer, self).__init__()
        self.n_qubits = size_in
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def forward(self, input, theta):
        @qml.qnode(self.dev)
        def circuit(inputs, theta):
            # TODO Embeding the data into the quantum circuit
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(theta, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        return torch.tensor(circuit(input, theta), requires_grad=True)

    #def backward(ctx, grad_output):
        # TODO Should we implement the backward pass?