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

        n_wires = kernel_size[0]*kernel_size[1]
        dev = qml.device("default.qubit", wires=n_wires)
        # Random circuit parameters
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_wires))
        
        # TODO Better way to implement the circuit than declare it in the initialisation??
        @qml.qnode(dev)
        def circuit(phi):
            # Encoding of 4 classical input values
            for j in range(4):
                qml.RY(np.pi * phi[j], wires=j)

            # Random quantum circuit
            RandomLayers(rand_params, wires=list(range(4)))

            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(4)]
        
        self.circuit = circuit

    def forward(self, x):
        # TODO Calculate the actual output size
        out = np.zeros((14, 14, 4))

        # TODO Not sure 100% about the x.shape[0] and x.shape[1] values
        for j in range(0, x.shape[0], self.kernel_size[0]):
            for k in range(0, x.shape[1], self.kernel_size[1]):
                # Process a region of the image with a quantum circuit
                q_results = self.circuit(
                    [
                        x[j, k, 0],
                        x[j, k + 1, 0],
                        x[j + 1, k, 0],
                        x[j + 1, k + 1, 0]
                    ]
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(self.kernel_size[0]*self.kernel_size[1]):
                    out[j // 2, k // 2, c] = q_results[c]
        return out
    
    def backward(self, x):
        # TODO Figre out if we need to implement the backward pass
        pass