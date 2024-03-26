import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

SAVE_PATH = "../../data/processed/HQNN-Quanv/"

class QuanvolutionalLayer():

    def __init__(self) -> None:
        
        self.num_qubits = 4
        self.num_layers = 2
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        
        self.rand_paramsrand_params = np.random.uniform(high=2 * np.pi, size=(self.num_layers, self.num_qubits))

        pass

    def qnode(self, phi ):    
        @qml.qnode(self.device)
        def circuit(self, phi):
            for j in range(self.num_qubits):
                qml.RY(np.pi *phi[j], wires=j)

            RandomLayers(self.rand_params, wires=list(range(self.num_qubits)))

            return [qml.expval(qml.PauliZ(j)) for j in range(self.num_qubits)]
        
        return circuit(phi)

    def quanv(self, image):
        """Convolves the input image with many applications of the same quantum circuit."""
        out = np.zeros((14, 14, self.num_qubits))
         # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = self.qnode(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0]
                    ]
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(4):
                    out[j // 2, k // 2, c] = q_results[c]
        return out
    
    def preprocess(self, img):
        q_train_images = []
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(train_images):
            print("{}/{}        ".format(idx + 1, n_train), end="\r")
            q_train_images.append(self.quanv(img))
        q_train_images = np.asarray(q_train_images)

        q_test_images = []
        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(test_images):
            print("{}/{}        ".format(idx + 1, n_test), end="\r")
            q_test_images.append(self.quanv(img))
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
        np.save(SAVE_PATH + "q_test_images.npy", q_test_images)
    
    
    
    
    

        
