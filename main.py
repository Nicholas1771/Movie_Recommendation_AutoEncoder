# Import the libraries
import numpy as np
import pandas as pd
import torch

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
    matrix = np.empty((num_users, num_movies), dtype='int')

    # Loop through each user rating of every user
    for id_user in data[:, 0]:
        # Get the movie id
        id_movie = data[id_user, 1]
        # Get the rating of the movie by the user
        rating = data[id_user, 2]

        # In the (id_user)th row in the matrix, in the (id_movie)th column, set the value to the rating of that movie
        matrix[id_user - 1][id_movie - 1] = rating

    return matrix


# Convert the training set and test set to matrix using the to_matrix method defined above
training_matrix = to_matrix(nb_users, nb_movies, training_set)
test_matrix = to_matrix(nb_users, nb_movies, test_set)
