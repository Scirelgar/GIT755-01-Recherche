import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

qml.enable_tape()

class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, input, theta):
        dev = qml.device("default.qubit", wires=len(input[0]))

        @qml.qnode(dev)
        def circuit(inputs, theta):
            for i in range(len(inputs)):
                qml.RX(theta[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

        expectation = circuit(input, theta)
        ctx.save_for_backward(input, theta)
        return torch.tensor(expectation, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        input_list = input.tolist()
        gradient = []
        for i in range(len(theta)):
            input_list_copy1 = input_list.copy()
            input_list_copy2 = input_list.copy()
            input_list_copy1[i] += 0.01
            input_list_copy2[i] -= 0.01
            input_plus = torch.tensor(input_list_copy1, requires_grad=False)
            input_minus = torch.tensor(input_list_copy2, requires_grad=False)
            expectation_plus = QuantumLayer.forward(None, input_plus, theta)
            expectation_minus = QuantumLayer.forward(None, input_minus, theta)
            gradient.append((expectation_plus - expectation_minus) / 0.02)
        grad_input = torch.tensor(gradient, requires_grad=True)
        grad_theta = grad_output.clone().detach()
        return grad_input, grad_theta

class QuantumLayerModule(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumLayerModule, self).__init__()
        self.num_qubits = num_qubits
        self.theta = nn.Parameter(torch.randn(self.num_qubits))

    def forward(self, x):
        return QuantumLayer.apply(x, self.theta)

class HybridQuantumModel(nn.Module):
    def __init__(self, num_qubits, num_classes):
        super(HybridQuantumModel, self).__init__()
        self.conv_layer = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.quantum_layer = QuantumLayerModule(num_qubits)
        self.fc_layer = nn.Linear(num_qubits, 64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.quantum_layer(x)
        x = self.fc_layer(x)
        x = self.softmax(x)
        return x

# Create an instance of the model
model = HybridQuantumModel(num_qubits=4, num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Display model summary
print(model)