# Some of the code taken from
# https://pennylane.ai/qml/demos/tutorial_quanvolution/

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

class QuanvolutionLayer(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2, padding=0, n_qdepth=1):
        super(QuanvolutionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_qdepth = n_qdepth
        self.circuit = self.qnode()
        
    def qnode(self):
        n_qubits = self.kernel_size[0]*self.kernel_size[1]
        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (self.n_qdepth, n_qubits)}

        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            #for j in range(n_qubits):
            #    qml.RY(np.pi *inputs[j], wires=j)

            qml.RX(weights[0, 0], wires=0)
            qml.RX(weights[0, 1], wires=1)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[0, 3])
            qml.RY(weights[0, 2], wires=0)
            qml.RY(weights[0, 3], wires=3)

            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
        return qml.qnn.TorchLayer(circuit, weight_shapes)
    
    def forward(self, x):
        n_qubits = self.kernel_size[0]*self.kernel_size[1]
        out = np.zeros((
            x.shape[0], 
            n_qubits, 
            int(x.shape[2]/self.kernel_size[0]), 
            int(x.shape[3]/self.kernel_size[1])))

        # Loop over the batch size
        for i in range(x.shape[0]):
            # Loop over the coordinates of the top-left pixel of 2X2 squares
            for j in range(0, x.shape[2], self.kernel_size[0]):
                for k in range(0, x.shape[3], self.kernel_size[1]):
                    # Process a region of the image with a quantum circuit
                    q_results = self.circuit(
                        torch.tensor([
                            x[i, 0, j, k],
                            x[i, 0, j, k + 1],
                            x[i, 0, j + 1, k],
                            x[i, 0, j + 1, k + 1]
                        ])
                    )

                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(n_qubits):
                        out[i, c, j // 2, k // 2] = q_results[c]

        return torch.tensor(out)