# Some of the code taken from
# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

# Set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# Import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from datetime import datetime
import os
import pickle

from models.LeNet import LeNet
from models.HQNN_Quanv import HQNN_Quanv
from models.HQNN_Parallel import HQNN_Parallel

# Define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

print("[INIT] Loading dataset...")

dataset = MNIST(root="../data", train=True, download=True, transform=ToTensor())
testDataset = MNIST(root="../data", train=False, download=True, transform=ToTensor())

print("[INIT] Preparing the datasets...")

# Define the training and validation split
TRAIN_SPLIT = 0.083333334
VAL_SPLIT = 0.008333334
NUM_TEST_SAMPLES = 100

# Calculate the train/validation split
numTrainSamples = int((len(dataset)) * TRAIN_SPLIT)
numValSamples = int((len(dataset)) * VAL_SPLIT)
unusedTrainSamples = len(dataset) - numTrainSamples - numValSamples
unusedTestSamples = len(testDataset) - NUM_TEST_SAMPLES

# Create Subset objects for train, validation, and test
trainData, valData, unusedData = random_split(
    dataset,
    [numTrainSamples, numValSamples, unusedTrainSamples],
)
testData, unusedData = random_split(
    testDataset,
    [NUM_TEST_SAMPLES, unusedTestSamples],
)

# Initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# Calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

print("[INIT] Initializing the model...")

# Configure the device we will be using to train the model
device = torch.device("cpu")

# Initialize the model
# model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(device)
model = HQNN_Quanv(in_channels=1, classes=len(trainData.dataset.classes)).to(device)
# Initialize the optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

# Initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("[TRAIN] Training the model...")

startTime = time.time()  # To measure how long training is going to take


# loop over our epochs
for e in range(0, EPOCHS):
    # Set the model in training mode
    model.train()
    # Initialize the total training/validation loss/correct predictions
    totalTrainLoss, totalValLoss, trainCorrect, valCorrect = 0, 0, 0, 0

    # Training
    for x, y in trainDataLoader:
        # Send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # Perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # Zero out the gradients, perform the backpropagation step, and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Sum the loss to the total training loss so far
        totalTrainLoss += loss
        # Calculate the number of correct predictions
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Validation
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        # Loop over the validation set
        for x, y in valDataLoader:
            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # Make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            # Calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # Calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    # Update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    intervalTime = time.time() - startTime

    # Print the model training and validation information
    print(
        "[TRAIN] EPOCH: {}/{} | Train loss: {:0.6f} | Train acc: {:0.4f} | Val loss: {:0.6f} | Val acc: {:0.4f} "
        "| Time elapsed: {:0.2f} ".format(
            e + 1,
            EPOCHS,
            avgTrainLoss,
            trainCorrect,
            avgValLoss,
            valCorrect,
            intervalTime,
        )
    )

# Finish measuring how long training took
endTime = time.time()
print("[TRAIN] Finished training the model...")
print(
    "[TRAIN] Total time taken to train the model: {:.2f}s".format(endTime - startTime)
)

print("[END] Evaluating the model...")

ys = []
# We can now evaluate the network on the test set
with torch.no_grad():  # Turn off autograd for testing evaluation
    model.eval()  # Set the model in evaluation mode
    preds = []  # Initialize a list to store our predictions
    # Evaluation with test dataset
    for x, y in testDataLoader:
        # Send the input to the device
        x = x.to(device)
        # Make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

print("[END] Generating the results...")

# Generate a classification report
print(
    classification_report(
        [y for _, y in testData], np.array(preds), target_names=testData.dataset.classes
    )
)

# Plot the training loss and accuracy
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
fig.suptitle(f"Training Loss and Accuracy with {model._get_name()}")
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.2)
ax1.plot(H["train_acc"], label="Training")
ax1.plot(H["val_acc"], label="Validation")
ax1.set_ylabel("Accuracy")
ax1.legend(loc="upper right")
ax2.plot(H["train_loss"], label="Training")
ax2.plot(H["val_loss"], label="Validation")
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epoch #")

fig.text(
    0.01,
    0.01,
    "Batch: {}, Epochs: {}, Train num: {}, Val num: {}, Opt: {}, Learning rate: {}, Time: {:.2f}s".format(
        BATCH_SIZE,
        EPOCHS,
        numTrainSamples,
        numValSamples,
        opt.__class__.__name__,
        INIT_LR,
        endTime - startTime,
    ),
    fontsize=10,
)

print("[END] Saving & logging...")

# Create a directory to store the results
dt = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
dirname = os.path.dirname(__file__)
path = "results/{date}".format(date=dt)
os.mkdir(path)

# Saving the results
plt.savefig(path + "/plot.png")

with open(path + "/data.pkl", "wb") as f:
    pickle.dump(H, f)

# TODO NOT WORKING AT THE MOMENT
# torch.save(model, path + "/model.pth") # Serialize the model to disk
