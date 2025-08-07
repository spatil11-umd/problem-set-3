'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")
    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = sorted(genres_df["genre"].unique())

    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}


    model_pred_df["actual genres"] = model_pred_df["actual genres"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )


    for _, row in model_pred_df.iterrows():
        predicted_genre = row["predicted"]
        true_genres = set(row["actual genres"])


        for genre in true_genres:
            if genre in genre_true_counts:
                genre_true_counts[genre] += 1


        if predicted_genre in true_genres:
            genre_tp_counts[predicted_genre] += 1
        else:

            if predicted_genre in genre_fp_counts:
                genre_fp_counts[predicted_genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
