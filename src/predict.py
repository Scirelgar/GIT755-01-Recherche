# Set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)

# Import the necessary packages
import argparse
import torch

# Construct the argument parser to receive the model and the data to predict
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to the trained PyTorch model")
args = vars(ap.parse_args())

# Configure the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and set it to evaluation mode
model = torch.load(args["model"]).to(device)
model.eval()

# TODO Load the data and predict