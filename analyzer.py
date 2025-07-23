
import pandas as pd

# Analisi di movies.dat
movies_df = pd.read_csv('c:/Users/vitop/Desktop/Tesi_py/data/raw/movies.dat', sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

num_movies = len(movies_df)
num_genres = movies_df['Genres'].str.split('|').explode().nunique()

print(f"Analisi di movies.dat:")
print(f"Numero di film: {num_movies}")
print(f"Numero di generi unici: {num_genres}")

# Analisi di ratings.dat
ratings_df = pd.read_csv('c:/Users/vitop/Desktop/Tesi_py/data/raw/ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')

num_users = ratings_df['UserID'].nunique()
num_reviews = len(ratings_df)
num_metadata_ratings = len(ratings_df.columns)

print(f"\nAnalisi di ratings.dat:")
print(f"Numero di utenti unici: {num_users}")
print(f"Numero di recensioni: {num_reviews}")
print(f"Numero di metadati per recensione: {num_metadata_ratings}")
