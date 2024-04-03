from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten, float32
from layers.QuanvolutionLayer import QuanvolutionLayer

class HQNN_Quanv(Module):
	def __init__(self, in_channels, classes):
		# Call the parent constructor
		super(HQNN_Quanv, self).__init__()

		self.quanv = QuanvolutionLayer()
		
		# Initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=784, out_features=784)
		self.relu3 = ReLU()

		# Initialize our softmax classifier
		self.fc2 = Linear(in_features=784, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)


	def forward(self, x):
		# ==== Convolutional layer
		x = self.quanv(x)

		x = flatten(x.to(float32), 1) # flatten the output from the previous layer and pass it

		# ==== FC layer
		x = self.fc1(x)
		x = self.relu3(x)
		
		# ==== Softmax classifier
		x = self.fc2(x)
		output = self.logSoftmax(x)

		return output # return the output predictions