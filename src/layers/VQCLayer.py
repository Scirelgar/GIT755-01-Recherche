import torch
import torch.nn as nn
import pennylane as qml
import math
import matplotlib.pyplot as plt
from pennylane import numpy as np


def qnode1(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(
            weights, wires=range(n_qubits), ranges=np.ones(3, dtype=int)
        )
        return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)


def qnode2(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    dev = qml.device(
        "lightning.qubit",
        wires=n_qubits,
    )

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    qnode_obj = qml.QNode(
        circuit, dev, interface="torch", diff_method="parameter-shift"
    )

    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)


def qnode3(n_qubits, n_qdepth):
    weight_shapes = {"weights": (0)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)


class VQCLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth):
        super(VQCLayer, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth

        self.layers = []
        for _ in range(math.ceil(size_in / n_qubits)):
            # Generate a quantum node for
            self.layers.append(qnode2(n_qubits, n_qdepth))

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for layer, inputs in zip(self.layers, x_split):
            outputs.append(layer(inputs))

        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    dev = qml.device("lightning.qubit", wires=5)

    @qml.qnode(dev)
    @qml.compile()
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(5), rotation="X")
        qml.StronglyEntanglingLayers(
            weights, wires=range(5), ranges=np.ones(3, dtype=int)
        )
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(5)]

    qml.draw_mpl(circuit)(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), torch.randn(3, 5, 3))
    plt.show()
