'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''


 
    total_tp = sum(genre_tp_counts.values())
    total_fp = sum(genre_fp_counts.values())
    total_fn = sum(genre_true_counts[genre] - genre_tp_counts.get(genre, 0) for genre in genre_list)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0
    )

    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts.get(genre, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''



    model_pred_df["actual genres"] = model_pred_df["actual genres"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


    pred_rows = []
    true_rows = []

    for _, row in model_pred_df.iterrows():
        predicted = row["predicted"]
        actual = set(row["actual genres"])

        pred_row = [1 if genre == predicted else 0 for genre in genre_list]
        true_row = [1 if genre in actual else 0 for genre in genre_list]

        pred_rows.append(pred_row)
        true_rows.append(true_row)

    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_matrix, pred_matrix, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_matrix, pred_matrix, average='micro', zero_division=0
    )

    return precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro