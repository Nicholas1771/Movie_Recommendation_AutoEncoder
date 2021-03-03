# Import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable


# Import the dataset
movies = pd.read_csv('data/ml-1m/ml-1m/movies.dat', sep='::', engine='python', header=None, encoding='latin-1')
users = pd.read_csv('data/ml-1m/ml-1m/users.dat', sep='::', engine='python', header=None, encoding='latin-1')
ratings = pd.read_csv('data/ml-1m/ml-1m/ratings.dat', sep=':', header=None, engine='python', encoding='latin-1')

# Prepare the training set and test set
training_set = pd.read_csv('data/ml-100k/ml-100k/u1.base', delimiter='\t', header=None)
test_set = pd.read_csv('data/ml-100k/ml-100k/u1.test', delimiter='\t', header=None)

# Convert training set and test set from dataframe to numpy arrays
training_set = np.array(training_set, dtype='int')
test_set = np.array(test_set, dtype='int')

# Get number of users and movies in the dataset
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))


# Define method to convert to matrix structure
def to_matrix(num_users, num_movies, data):
    """
    This method will take the training set and test set and rearrange the data into a list of lists where each row will
    correspond to a user with all of its ratings for every movie in the movies dataset. If the user has not rated the
    the value will be zero.
    :return:
    """

    # Create a 2D matrix of zeros representing every movie rating from every user
    n_matrix = np.empty((num_users, num_movies), dtype='int')

    # Loop through each user rating of every user
    for user_rating in data:
        # Get the user id
        id_user = user_rating[0]
        # Get the movie id
        id_movie = user_rating[1]
        # Get the rating of the movie by the user
        rating = user_rating[2]

        # In the (id_user)th row in the matrix, in the (id_movie)th column, set the value to the rating of that movie
        n_matrix[id_user - 1, id_movie - 1] = rating

    # Convert ndarray matrix to list matrix by copying and casting each line in n_matrix
    matrix = []
    for line in n_matrix:
        matrix.append(list(line))

    return matrix


# Convert the training set and test set to matrix using the to_matrix method defined above
training_matrix = to_matrix(nb_users, nb_movies, training_set)
test_matrix = to_matrix(nb_users, nb_movies, test_set)

# Convert the data into torch tensors
training_tensor = torch.FloatTensor(training_matrix)
test_tensor = torch.FloatTensor(test_matrix)


# Create the architecture of the neural network
class SAE(nn.Module):

    # Method to initialize the architecture of the autoencoder
    def __init__(self, ):
        super(SAE, self).__init__()
        # Create 4 full connection layers in the neural network
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
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


# Create the stacked auto encoder object
sae = SAE()
# Criterion is the means squared error loss
criterion = nn.MSELoss()
# Uses the RMSProp optimizer from pyTorch
optimizer = opt.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Train the SAE
# Number of epochs to train the AE
nb_epochs = 200

# Loop through the epochs
for epoch in range(1, nb_epochs+1):
    # Tracks the training loss throughout the epochs
    train_loss = 0
    # Tracks number of users who rated at least one movie in the training set
    s = 0.

    # Loop through all users in the training set
    for id_user in range(nb_users):
        # Get the tensor corresponding the user in the loop and create a fake "batch" using the Variable class
        x = Variable(training_tensor[id_user]).unsqueeze(0)
        target = x.clone()
        # Filter out users who have not rated a movie by checking their sum of ratings
        if torch.sum(target.data > 0) > 0:
            # Feed the tensor forward and save the result
            output = sae.forward(x)
            # Make sure gradient is not computed for the target, saving memory
            target.require_grad = False
            # Get rid of generated output values where the corresponding input was 0, loss cant be computed from this
            output[target == 0] = 0

            # Calculate the mean squared loss
            loss = criterion(target, output)
            # Compute the mean using movies with non-zero ratings, adding 1e-10 prevent division by zero
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()

    print(f'epoch: {epoch} train loss: {train_loss/s}')


s_test = 0
test_loss = 0
# Loop through all users in the test set
for id_user in range(nb_users):
    # Get the tensor corresponding the user in the loop and create a fake "batch" using the Variable class
    x = Variable(training_tensor[id_user]).unsqueeze(0)
    target = Variable(test_tensor[id_user]).unsqueeze(0)
    # Filter out users who have not rated a movie by checking their sum of ratings
    if torch.sum(target.data > 0) > 0:
        # Feed the tensor forward and save the result
        output = sae.forward(x)
        # Make sure gradient is not computed for the target, saving memory
        target.require_grad = False
        output[target == 0] = 0
        # Calculate the mean squared loss
        loss = criterion(target, output)
        # Compute the mean using movies with non-zero ratings, adding 1e-10 prevent division by zero
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s_test += 1.

print(f'test loss: {test_loss / s_test}')
