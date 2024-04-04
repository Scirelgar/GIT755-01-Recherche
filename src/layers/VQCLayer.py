import torch
import torch.nn as nn
import pennylane as qml
import math


def qnode1(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)


def qnode2(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)


class VQCLayer(nn.Module):
    def __init__(self, input, n_qubits, n_qdepth):
        super(VQCLayer, self).__init__()
        self.input = input
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth

        self.layers = []
        for _ in range(math.ceil(input / n_qubits)):
            # Generate a quantum node for
            self.layers.append(qnode1(n_qubits, n_qdepth))

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for layer, inputs in zip(self.layers, x_split):
            outputs.append(layer(inputs))

        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)
