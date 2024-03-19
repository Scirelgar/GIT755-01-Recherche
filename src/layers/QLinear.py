import torch
import torch.nn as nn
import pennylane as qml
import math

class QuantumLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev)
        def circuit(inputs, weights):
            # TODO Correctly embeding the inputs into the quantum circuit
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits)) # TODO Strongly entangled
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.layers = []
        weight_shapes = {"weights": (n_layers, n_qubits)}
        for i in range(math.ceil(size_in/self.n_qubits)):
            self.layers.append(qml.qnn.TorchLayer(circuit, weight_shapes))

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for layer, inputs in zip(self.layers, x_split):
            outputs.append(layer(inputs)) # theta=torch.randn(self.n_layers, self.n_qubits)
        
        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)