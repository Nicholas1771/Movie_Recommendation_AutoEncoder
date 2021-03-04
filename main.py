# Import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
from SAE import SAE
import tkinter as tk

movies = None
users = None
ratings = None
nb_users = 0
nb_movies = 0
training_tensor = None
test_tensor = None
sae = None


# Define method to convert to matrix structure
def to_matrix(num_users, num_movies, dataset):
    """
    This method will take the training set and test set and rearrange the data into a list of lists where each row will
    correspond to a user with all of its ratings for every movie in the movies dataset. If the user has not rated the
    the value will be zero.
    :return:
    """

    # Create a 2D matrix of zeros representing every movie rating from every user
    n_matrix = np.empty((num_users, num_movies), dtype='int')

    # Loop through each user rating of every user
    for user_rating in dataset:
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


def data_preprocessing():
    """
    This method sets up the global variables for the training to begin. It properly formats the training set into
    a tensor, among other things.
    :return:
    """
    global nb_movies, nb_users, training_tensor, test_tensor, movies, users, ratings

    print('Preprocessing the dataset')

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

    # Convert the training set and test set to matrix using the to_matrix method defined above
    training_matrix = to_matrix(nb_users, nb_movies, training_set)
    test_matrix = to_matrix(nb_users, nb_movies, test_set)

    # Convert the data into torch tensors
    training_tensor = torch.FloatTensor(training_matrix)
    test_tensor = torch.FloatTensor(test_matrix)

    print('Preprocessing Complete')


def create_gui():
    window = tk.Tk(className='Movie Recommender - Stacked Autoencoder')
    window.geometry('500x500')

    epochs_label = tk.Label(window, text='Epochs: ')
    epochs_label.pack()

    epochs_var = tk.IntVar()
    epochs_entry = tk.Entry(window, textvariable=epochs_var)
    epochs_entry.pack()

    data_prep_btn = tk.Button(window, text='Preprocess Data', width=25, command=data_preprocessing)
    data_prep_btn.pack()

    train_btn = tk.Button(window, text='Train SAE', width=25, command=lambda: train(epochs_var.get()))
    train_btn.pack()

    test_btn = tk.Button(window, text='Test SAE', width=25, command=test)
    test_btn.pack()

    window.mainloop()


def train(nb_epochs=10):
    global sae
    # Create the stacked auto encoder object
    sae = SAE(nb_movies)
    # Criterion is the means squared error loss
    criterion = nn.MSELoss()
    # Uses the RMSProp optimizer from pyTorch
    optimizer = opt.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

    # Train the SAE
    # Number of epochs to train the AE
    print(f'Training dataset on {nb_epochs} epochs')

    # Loop through the epochs
    for epoch in range(1, nb_epochs + 1):
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
                # Get rid of generated output values where the corresponding input was 0,
                # loss cant be computed from this
                output[target == 0] = 0

                # Calculate the mean squared loss
                loss = criterion(target, output)
                # Compute the mean using movies with non-zero ratings, adding 1e-10 prevent division by zero
                mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
                loss.backward()
                train_loss += np.sqrt(loss.data * mean_corrector)
                s += 1.
                optimizer.step()
        print(f'epoch: {epoch} train loss: {train_loss / s}')
    print('Training complete')


def test():
    s_test = 0
    test_loss = 0
    criterion = nn.MSELoss()
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
    print('Testing complete')


if __name__ == '__main__':
    create_gui()
