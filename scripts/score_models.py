from pathlib import Path
import pandas as pd

from what_to_wear_outdoors import RunningOutfitPredictor, get_test_data_path, get_model_path


def save_model_scores(drop_known, model_id):
    rop = RunningOutfitPredictor()
    # Need to load up a dataset with known values
    mdl = rop.get_model_by_strategy(model_id)
    df = rop.ingest_data(get_test_data_path())
    col_scores, overall_score = mdl.score(df, drop_known)

    results_df = None
    if Path.exists(get_model_path('model_results.csv')):
        results_df = pd.read_csv('model_results.csv')
    else:
        results_df = pd.DataFrame(columns=['dt', 'model_id', ''])
    print(f'\n\nScore (Drop 100% = {drop_known}) -- {overall_score}\nColumn Scores:\n{col_scores}')


save_model_scores(True, 'ddt')
