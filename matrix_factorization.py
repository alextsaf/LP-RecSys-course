import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm

K = 2
steps = 1000
sample = 1000
debug = False

def print_debug(*args, **kwargs):
    if debug:
        print(*args, **kwargs)
        
def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def loss(X, U, V, b, c, mu, reg):
    loss = 0
    
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > 0:
                pred = np.dot(U[i, :], V[j, :]) + b[i] + c[j] + mu
                loss += (X[i, j] - pred) ** 2
                count += 1
    
    regularization = reg * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2 + np.linalg.norm(b) ** 2 + np.linalg.norm(c) ** 2)
    return loss / count + regularization

def Wi_bi(X, V, U, b , c, mu, i, reg):    
    print_debug(f"X: {X.toarray()}")
    print_debug(f"V: {V}")
    
    movieIds = np.where(X[i, :].toarray() > 0)[1]
    print_debug(f"movieIds: {movieIds}")
    
    A = np.dot(V[movieIds, :].T, V[movieIds, :]) + np.eye(K) * reg # (K, K)
    r = X[i, movieIds].toarray().T[0]
    B = np.dot((r - b[i] - c[movieIds] - mu), V[movieIds, :]) # (K, )

    print_debug(f"A: {A}")
    print_debug(f"B: {B}")
    
    Wi = np.linalg.solve(A, B)
    bi = np.sum(r - np.dot(U[i, :], V[movieIds, :].T) - c[movieIds] - mu) / (len(movieIds) + reg)
    print_debug(f"Wi: {Wi}")
    print_debug(f"Wi * A: {np.dot(Wi, A)}")
    
    return Wi, bi

def Uj_cj(X, V, U, b, c, mu, j, reg):
    print_debug(f"X: {X.toarray()}")
    print_debug(f"U: {U}")   

    userids = np.where(X[:, 0].toarray() > 0)[1]
    print_debug(f"userids: {userids}")
    
    A = np.dot(U[userids].T, U[userids]) + np.eye(K) * reg # (K, K)
    r = X[userids, j].toarray().T[0]
    B = np.dot((r - c[j] - mu), U[userids]) # (K, )
    
    print_debug(f"A: {A}")
    print_debug(f"B: {B}")

    Uj = np.linalg.solve(A, B)
    cj = np.sum(r - np.dot(U[userids, :], V[j, :].T) - b[userids] - mu) / (len(userids) + reg)
    print_debug(f"Uj: {Uj}")
    print_debug(f"Uj * A: {np.dot(Uj, A)}")
    
    return Uj, cj

def matrix_factorization(df, U, V, b, c, reg = 0.1, steps=1000):
    N, M = len(U), len(V)
    mu = np.mean(df['rating'])
    
    X = csr_matrix((df['rating'], (df['userId'], df['movieId'])), shape=(N, M))
    
    history = []
    for step in tqdm(range(steps), leave=False):
        for i in range(X.shape[0]):
            print_debug(f"Step {step}, i={i}")
            U[i, :], b[i] = Wi_bi(X, V, U, b, c, mu, i, reg)
            print_debug(f"U[i, :]: {U[i, :]}")
            print_debug(f"U: {U}")
        for j in range(X.shape[1]):
            print_debug(f"Step {step}, j={j}")
            V[j, :], c[j] = Uj_cj(X, V, U, b, c, mu, j, reg)
            print_debug(f"V[j, :]: {V[j, :]}")
            print_debug(f"V: {V}")
        if step % 100 == 0:
            step_loss = loss(X, U, V, b, c, mu, reg)
            print(f"Step {step}, loss={step_loss}")
            history.append(step_loss)
    return U, V, c, j, mu, history

def score(userId, movieId, P, Q, b, c, mu, user_to_index, movie_to_index):
    user_index = user_to_index[int(userId)]
    movie_index = movie_to_index[int(movieId)]
    pred = np.dot(P[user_index, :], Q[movie_index, :]) + mu + b[user_index] + c[movie_index]
    pred = np.clip(pred, 0, 5)
    return pred

def predict(df, user_to_index, movie_to_index, P, Q, b, c, mu):
    df['prediction'] = [score(x['userId'], x['movieId'], P, Q, b, c, mu, user_to_index, movie_to_index) 
                        for index, x in tqdm(df.iterrows(), total=len(df))]
    return df

print_debug("Debug mode on")

# read and drop the timestamp column, ids are int, ratings are float
df = pd.read_csv('../Data/rating.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': int, 'movieId': int, 'rating': float})
R = df.copy()

# keep the top 10000 most rated movies if sample is set
if sample > 0:
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

# Split the data into training and test sets, stratified by user and movie
train, test = train_test_split(R, test_size=0.2, stratify=R[['movieId']])

# Apply the mapping to the training set
train['userId'] = train['userId'].apply(lambda x: user_to_index[x])
train['movieId'] = train['movieId'].apply(lambda x: movie_to_index[x])

# print occurences of each userID and movieID in the training set
print_debug(f"Movies of the user with the least ratings: {min(train['userId'].value_counts())}")
print_debug(f"Ratings of the movie with the least ratings: {min(train['movieId'].value_counts())}")

# Generate martrices P and Q
P = np.random.randn(R["userId"].nunique(), K)
Q = np.random.randn(R["movieId"].nunique(), K)
b = np.zeros(R["userId"].nunique())
c = np.zeros(R["movieId"].nunique())

P, Q, c, j, mu, history = matrix_factorization(train, P, Q, b, c, steps)


print_debug(csr_matrix((train['rating'], (train['userId'], train['movieId']))).toarray())
matrix = np.dot(P, Q.T) + mu + b[:, np.newaxis] + c[np.newaxis, :]
# trim the matrix to the range of the original ratings
matrix = np.clip(matrix, 0, 5)
print_debug(matrix)

plt.plot(history)

test_preds = predict(test, user_to_index, movie_to_index, P, Q, b, c, mu)
mse = mean_squared_error(test_preds['rating'], test_preds['prediction'])
print(f"Mean Squared Error: {mse}")

print(test_preds.head())



