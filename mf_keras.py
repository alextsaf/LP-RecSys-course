# implement simple matrix factorization algorithm using keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

K = 15
epochs = 1
sample = 1000
debug = False

def print_debug(*args, **kwargs):
    if debug:
        print(*args, **kwargs)
        
class MatrixFactorization:
    def __init__(self, n_users, n_movies, K, reg=0.1):
        self.n_users = n_users
        self.n_movies = n_movies
        self.K = K
        self.reg = reg
        
    def fit(self, X_train, X_test, epochs=10, lr=0.01):
        # Create the model
        user = Input(shape=(1,))
        movie = Input(shape=(1,))
        P = Embedding(self.n_users, self.K)(user)
        Q = Embedding(self.n_movies, self.K)(movie)
        user_bias = Embedding(self.n_users, 1)(user)
        movie_bias = Embedding(self.n_movies, 1)(movie)
        R = Dot(axes=2)([P, Q])
        R = Add()([R, user_bias, movie_bias])
        R = Flatten()(R)
        
        model = Model(inputs=[user, movie], outputs=R)
        model.compile(
            loss='mse',
            optimizer='SGD',
            metrics=['mse']
        )
        
        # Train the model   
        self.hist = model.fit(
            [X_train['userId'], X_train['movieId']],
            X_train['rating'],
            epochs=epochs,
            batch_size=256,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
            validation_data=([X_test['userId'], X_test['movieId']], X_test['rating'])
        )
        
        # Get the weights
        self.P = model.get_layer('embedding').get_weights()[0]
        self.Q = model.get_layer('embedding_1').get_weights()[0]
        self.user_bias = model.get_layer('embedding_2').get_weights()[0]
        self.movie_bias = model.get_layer('embedding_3').get_weights()[0]
        
    def predict(self, X):
        r = np.sum(self.P[X['userId']] * self.Q[X['movieId']], axis=1)
        r += self.user_bias[X['userId']].flatten()
        r += self.movie_bias[X['movieId']].flatten()
        return r
    
    def evaluate(self, X):
        return np.mean(np.power(X['rating'] - self.predict(X), 2))
    
    def plot(self):
        plt.plot(self.hist.history['loss'], label='loss')
        plt.plot(self.hist.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()
        

# Load the data

# read and drop the timestamp column, ids are int, ratings are float
print_debug("Reading data")
df = pd.read_csv('../Data/rating.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': int, 'movieId': int, 'rating': float})
R = df.copy()

# keep the top 10000 most rated movies if sample is set
if sample > 0:
    print_debug(f"Sampling {sample} movies")
    top_movies = df['movieId'].value_counts().head(10000).sample(sample).index
    R = df[df['movieId'].isin(top_movies)]

    # exclude users who have rated less than 5 movies
    top_users = R['userId'].value_counts()
    top_users = top_users[top_users >= 4].index
    R = R[R['userId'].isin(top_users)]

user_ids = sorted(R['userId'].unique().tolist())
movie_ids = sorted(R['movieId'].unique().tolist())

user_to_index = {old: new for new, old in enumerate(user_ids)}
movie_to_index = {old: new for new, old in enumerate(movie_ids)}

n_users = len(user_ids)
n_movies = len(movie_ids)

R['userId'] = R['userId'].apply(lambda x: user_to_index[x])
R['movieId'] = R['movieId'].apply(lambda x: movie_to_index[x])

# Split the data into training and test sets, stratified by user and movie
print_debug("Splitting data")
# train, test = train_test_split(R, test_size=0.2, stratify=R[['movieId']])
train, test = train_test_split(R, test_size=0.2)

# print occurences of each userID and movieID in the training set
print_debug(f"Movies of the user with the least ratings: {min(train['userId'].value_counts())}")
print_debug(f"Ratings of the movie with the least ratings: {min(train['movieId'].value_counts())}")

# Create the model
print_debug("Creating model")
model = MatrixFactorization(n_users, n_movies, K)
print_debug("Fitting model")
model.fit(train, test, epochs=epochs)

mse = model.evaluate(test)
print(f"Mean Squared Error: {mse}")

model.plot()

        