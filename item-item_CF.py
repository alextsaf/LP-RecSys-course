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
    
    # make movies be the rows
    sparse_matrix = sparse_matrix.T
    
    return sparse_matrix, user_to_index, index_to_user, movie_to_index, index_to_movie

def compute_item_similarity(itemID, sparse_matrix):
    """
    Compute the item similarity
    :param itemID: item ID
    :param sparse_matrix: sparse matrix
    :return: item similarity
    """
    # Get the user ratings
    item_ratings = sparse_matrix[itemID]
    
    # Compute the similarity
    similarity = cosine_similarity(item_ratings, sparse_matrix).flatten()

    
    return similarity

def compute_item_similarity_matrix(sparse_matrix):
    """
    Compute the user similarity matrix
    :param sparse_matrix: sparse matrix
    :return: user similarity matrix
    """
    
    user_similarity_matrix = np.array([compute_item_similarity(itemID, sparse_matrix) for itemID in tqdm(range(sparse_matrix.shape[0]))])
    
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
    item_sim = item_similarity_matrix[item_index]
    
    # Get the ratings
    ratings = sparse_matrix[:, user_index].toarray().flatten()

    # Compute the score
    numerator_factors = (item_sim * (ratings - average_movie_ratings))[ratings > 0]
    numerator = np.sum(numerator_factors)

    denominator = np.sum(np.abs(item_sim)[ratings > 0])

    score = average_movie_ratings[item_index] + numerator / denominator if denominator != 0 else average_movie_ratings[item_index]
    
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

def mse(predictions):
    """
    Compute the mean squared error
    :param predictions: predictions
    :return: mean squared error
    """
    return np.mean((predictions['rating'] - predictions['prediction']) ** 2)
    


# https://www.kaggle.com/grouplens/movielens-20m-dataset/download
df = pd.read_csv('../Data/rating.csv')

# keep the top 10000 most rated movies
top_movies = df['movieId'].value_counts().head(500).index
df = df[df['movieId'].isin(top_movies)]

# Implement and test the accuracy of the collaborative filtering algorithm. Use a train and test set.
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)
print(train.shape, test.shape, df.shape)

starttime = time.time()
sparse_matrix, user_to_index, index_to_user, movie_to_index, index_to_movie = create_sparse_matrix(train, test)
print("Time to create sparse matrix: ", time.time() - starttime)

average_movie_ratings = np.mean(sparse_matrix[sparse_matrix > 0], axis=1).flatten()

starttime = time.time()
item_similarity_matrix = compute_item_similarity_matrix(sparse_matrix)
print("Time to compute user similarity matrix: ", time.time() - starttime)

starttime = time.time()
predictions = predict(test, user_to_index, movie_to_index, sparse_matrix)
print("Time to predict", len(predictions),"values: ", time.time() - starttime)

print("Mean squared error: ", mse(predictions))




