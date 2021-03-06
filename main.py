# Import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
from SAE import SAE
import tkinter as tk
from tkinter import Frame, Canvas, Scrollbar, Listbox, Tk, Button, Label, IntVar, Entry

movies = None
users = None
ratings = None
nb_users = 0
nb_movies = 0
training_tensor = None
test_tensor = None
sae = None
user_ratings = []
selected_text = None
selected_movie_name = None
pred_matrix = None
movie_list = None


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


def get_movie_list(training_set):
    global movie_list
    movie_list = []
    for movie_id in training_set.iloc[:, 1].unique():
        try:
            movie_list.append(movies[movie_id == movies.iloc[:, 0]].iloc[0, 1])
        except:
            pass


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

    get_movie_list(training_set)

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


def submit_rating(rating):
    rating = int(rating)
    user_ratings.append((selected_movie_name, rating))
    print(user_ratings)


def cur_select(event):
    global selected_movie_name
    widget = event.widget
    selection = widget.curselection()
    picked = widget.get(selection[0])
    selected_movie_name = picked[0:-1]
    selected_text.delete(1.0, tk.END)
    selected_text.insert(1.0, selected_movie_name)


def create_gui():
    global selected_text
    window = Tk()
    frame = Frame(window, width=500, height=500)
    frame.pack(padx=10, pady=10)

    container = Frame(frame)
    container.pack()

    epochs_label = Label(container, text='Epochs: ')
    epochs_label.pack(side=tk.LEFT)

    epochs_var = IntVar()
    epochs_var.set(200)
    epochs_entry = Entry(container, textvariable=epochs_var, width=4)
    epochs_entry.pack(side=tk.LEFT)

    train_btn = Button(container, text='Train SAE', width=15, command=lambda: train(epochs_var.get()))
    test_btn = Button(container, text='Test SAE', width=15, command=test)

    train_btn.pack(side=tk.LEFT)
    test_btn.pack(side=tk.LEFT)

    canvas = Canvas(frame, bg='#FFFFFF')
    canvas.pack(fill=tk.BOTH, expand=True)

    selected_container = Frame(canvas)
    selected_container.pack(side=tk.TOP, pady=5)

    selected_lbl = Label(selected_container, text='Selected movie: ', height=1)
    selected_lbl.pack(side=tk.LEFT, fill=tk.X)

    selected_text = tk.Text(selected_container, height=1, width=25)
    # selected_text.config(state=tk.DISABLED)
    selected_text.pack(side=tk.LEFT, fill=tk.X)

    rating_lbl = Label(selected_container, text='Rating (1 - 5): ', height=1)
    rating_lbl.pack(side=tk.LEFT, fill=tk.X)

    rating_var = IntVar()
    rating_var.set(0)
    epochs_entry = Entry(selected_container, textvariable=rating_var, width=4)
    epochs_entry.pack(side=tk.LEFT)

    rating_btn = Button(selected_container, text='Rate', width=5, command=lambda: submit_rating(rating_var.get()))
    rating_btn.pack()

    pred_btn = Button(selected_container, text='Predict', width=5, command=lambda: predict())
    pred_btn.pack()

    movie_scroll = Scrollbar(canvas, orient=tk.VERTICAL)
    movie_scroll.pack(side='right', fill='y')
    movie_listbox = Listbox(canvas, yscrollcommand=movie_scroll.set)
    movie_listbox.bind('<<ListboxSelect>>', cur_select)

    for movie in movie_list:
        movie_listbox.insert(tk.END, movie + '\n')

    movie_listbox.pack(side='left', fill='both', expand=True)
    movie_scroll.config(command=movie_listbox.yview)

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


def predict():
    import pandas as pd

    pred_dataset = zip(*user_ratings)
    pred_movie_names, pred_ratings = list(pred_dataset)
    pred_movie_names = list(pred_movie_names)
    pred_ratings = list(pred_ratings)

    pred_movie_id = []
    for name in pred_movie_names:
        pred_movie_id.append(int(movies[name == movies[1]].iloc[:, 0]))

    pred_dict = {'user id': list(np.ones(len(pred_ratings))),
                 'movie id': pred_movie_id,
                 'rating': pred_ratings,
                 'time': list(np.zeros(len(pred_ratings)))}

    pred_df = pd.DataFrame(pred_dict, columns=['user id', 'movie id', 'rating', 'time'])

    pred_np = np.array(pred_df, dtype='int')

    global pred_matrix
    pred_matrix = to_matrix(1, nb_movies, pred_np)

    pred_tensor = torch.FloatTensor(pred_matrix)

    x = Variable(pred_tensor).unsqueeze(0)
    output = sae.forward(x)
    output = output.squeeze().tolist()
    print(type(output))
    print(len(output))
    print(output)
    string = 'Truman Show, The (1998)'
    suggestions = []
    for i, pred in enumerate(output):
        if pred > 4.:
            try:
                suggestions.append((movies[movies.iloc[:, 0] == i].iloc[0, 1], pred))
            except:
                pass
    suggestions.sort(key=lambda tup: tup[1])
    suggestions = suggestions[::-1]
    print(len(suggestions), suggestions)


if __name__ == '__main__':
    data_preprocessing()
    create_gui()
