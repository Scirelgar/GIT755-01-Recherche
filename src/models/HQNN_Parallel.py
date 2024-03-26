from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from layers.QLinear import QuantumLayer

class HQNN(Module):
	def __init__(self, numChannels, classes):
		# Call the parent constructor
		super(HQNN, self).__init__()

		# Initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=16, kernel_size=(5, 5), padding=2)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# Initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		# Initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=1568, out_features=500)
		self.relu3 = ReLU()
		
		# Initialize the quantum layer
		self.qlayer1 = QuantumLayer(size_in=500, n_qubits=5, n_layers=1)

		# Initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# ==== Convolutional layer
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		x = flatten(x, 1) # flatten the output from the previous layer and pass it

		# ==== FC layer
		x = self.fc1(x)
		x = self.relu3(x)
		
		# ==== Quantum layer
		x = self.qlayer1(x)

		# ==== Softmax classifier
		x = self.fc2(x)
		output = self.logSoftmax(x)

		return output # return the output predictions