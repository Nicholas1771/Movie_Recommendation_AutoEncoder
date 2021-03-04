# Import the libraries
import torch.nn as nn


class SAE(nn.Module):

    # Method to initialize the architecture of the autoencoder
    def __init__(self, size):
        super(SAE, self).__init__()
        # Create 4 full connection layers in the neural network
        self.fc1 = nn.Linear(size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, size)
        # Create sigmoid activation function
        self.activation = nn.Sigmoid()

    # Forward propagation
    def forward(self, input_tensor):
        # Move x forward through the four full connections layers and use the sigmoid activation
        input_tensor = self.activation(self.fc1(input_tensor))
        input_tensor = self.activation(self.fc2(input_tensor))
        input_tensor = self.activation(self.fc3(input_tensor))
        input_tensor = self.fc4(input_tensor)
        return input_tensor
