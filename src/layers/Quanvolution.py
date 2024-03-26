# Some of the code taken from
# https://pennylane.ai/qml/demos/tutorial_quanvolution/

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

class Quanvolution(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2, padding=0, n_layers=1):
        super(Quanvolution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires=self.kernel_size[0]*kernel_size[1])
        # Random circuit parameters
        self.circuit = self.qnode()

        
    def qnode(self): 
        weight_shapes = {"weights": (self.n_layers, self.n_qubits())}

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            for j in range(self.n_qubits()):
                qml.RY(np.pi *inputs[j], wires=j)

            qml.RX(weights[0, 0], wires=0)
            qml.RX(weights[0, 1], wires=1)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[0, 3])
            qml.RY(weights[0, 3], wires=0)
            qml.RY(weights[0, 4], wires=3)

            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits())]
        
        return qml.qnn.TorchLayer(circuit, weight_shapes)
    
    def n_qubits(self):
        return self.kernel_size[0]*self.kernel_size[1]

    def forward(self, x):
        # TODO Calculate the actual output size
        out = np.zeros((x.shape[0], int(x.shape[2]/self.kernel_size[0]), int(x.shape[3]/self.kernel_size[1]), self.n_qubits()))

        # TODO Not sure 100% about the x.shape[0] and x.shape[1] values
        for i in range(x.shape[0]):
            for j in range(0, x.shape[2], self.kernel_size[0]):
                for k in range(0, x.shape[3], self.kernel_size[1]):
                    # Process a region of the image with a quantum circuit
                    q_results = self.circuit(
                        np.array([
                            x[i, 0, j, k],
                            x[i, 0, j, k + 1],
                            x[i, 0, j + 1, k],
                            x[i, 0, j + 1, k + 1]
                        ])
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(self.kernel_size[0]*self.kernel_size[1]):
                        out[i, j // 2, k // 2, c] = q_results[c]

        return out