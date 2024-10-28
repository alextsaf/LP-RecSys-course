import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm

def create_sparse_matrix(train, test):
    """
    Create a sparse matrix from the data frame
    :param df: data frame
    :return: sparse matrix
    """
    
    # Make ids start at 0 for both sets
    user_ids = sorted(df['userId'].unique().tolist() + test['userId'].unique().tolist())
    movie_ids = sorted(df['movieId'].unique().tolist() + test['movieId'].unique().tolist())
    
    # Create a mapping
    user_to_index = {old: new for new, old in enumerate(user_ids)}
    index_to_user = {new: old for new, old in enumerate(user_ids)}
    movie_to_index = {old: new for new, old in enumerate(movie_ids)}
    index_to_movie = {new: old for new, old in enumerate(movie_ids)}
    
    # Apply the mapping
    df['userId'] = df['userId'].apply(lambda x: user_to_index[x])
    df['movieId'] = df['movieId'].apply(lambda x: movie_to_index[x])
    

    # Create a sparse matrix
    sparse_matrix = csr_matrix((df['rating'], (df['userId'], df['movieId'])))
    return sparse_matrix, user_to_index, index_to_user, movie_to_index, index_to_movie

def compute_user_similarity(userID, sparse_matrix):
    """
    Compute the user similarity
    :param userID: user ID
    :param sparse_matrix: sparse matrix
    :return: user similarity
    """
    # Get the user ratings
    user_ratings = sparse_matrix[userID]
    
    # Compute the similarity
    similarity = cosine_similarity(user_ratings, sparse_matrix).flatten()

    
    return similarity

def compute_user_similarity_matrix(sparse_matrix):
    """
    Compute the user similarity matrix
    :param sparse_matrix: sparse matrix
    :return: user similarity matrix
    """
    
    user_similarity_matrix = np.array([compute_user_similarity(userID, sparse_matrix) for userID in tqdm(range(sparse_matrix.shape[0]))])
    
    return user_similarity_matrix

def score(userID, itemID, user_to_index, movie_to_index, sparse_matrix):
    """
    Compute the score for a user and an item
    :param userID: user ID
    :param itemID: item ID
    :param user_to_index: user to index mapping
    :param movie_to_index: movie to index mapping
    :param sparse_matrix: sparse matrix
    :param user_similarity: user similarity
    :return: score
    """
    # Get the user index
    user_index = user_to_index[userID]
    
    # Get the item index
    item_index = movie_to_index[itemID]
    
    # Get the user similarity
    user_sim = user_similarity_matrix[user_index]
    
    # Get the ratings
    ratings = sparse_matrix[:, item_index].toarray().flatten()

    # Compute the score
    sum_user_sim = np.sum(user_sim)   # sum of user similarities
    dot_product = np.dot(user_sim, ratings) # dot product of user similarity and ratings
    score = dot_product / sum_user_sim

    return score

def predict(df, user_to_index, movie_to_index, sparse_matrix):
    """
    Predict the ratings
    :param df: data frame
    :param user_to_index: user to index mapping
    :param movie_to_index: movie to index mapping
    :param sparse_matrix: sparse matrix
    :param user_similarity: user similarity
    :return: data frame with predictions
    """
    #tqdm 
    
    # df['prediction'] = df.apply(lambda x: score(x['userId'], x['movieId'], user_to_index, movie_to_index, sparse_matrix), axis=1)
    df['prediction'] = [score(x['userId'], x['movieId'], user_to_index, movie_to_index, sparse_matrix) 
                        for index, x in tqdm(df.iterrows(), total=len(df))]

    return df
    


# https://www.kaggle.com/grouplens/movielens-20m-dataset/download
df = pd.read_csv('../Data/rating.csv')

# Implement and test the accuracy of the collaborative filtering algorithm. Use a train and test set.

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)
print(train.shape, test.shape, df.shape)

starttime = time.time()
sparse_matrix, user_to_index, index_to_user, movie_to_index, index_to_movie = create_sparse_matrix(train, test)
print("Time to create sparse matrix: ", time.time() - starttime)

starttime = time.time()
user_similarity_matrix = compute_user_similarity_matrix(sparse_matrix)
print("Time to compute user similarity matrix: ", time.time() - starttime)

starttime = time.time()
predictions = predict(test, user_to_index, movie_to_index, sparse_matrix)
print("Time to predict ", len(predictions)," values: ", time.time() - starttime)

print(predictions.head())




