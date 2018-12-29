from typing import Dict

import numpy as np
import logging

logger = logging.getLogger(__name__)


class NoWeightScoringStrategy:
    """ A scoring strategy that doesn't weight any particular classification more than any other.  Therefore it just
        returns a mean score of whether the labels match.
    """

    def __init__(self):
        pass

    def score(self, df_predicted, df_actual, drop_perfect_scores=False) -> (Dict[str, float], float):
        column_score = np.mean((df_predicted == df_actual), axis=0)
        if drop_perfect_scores:
            column_score = column_score[column_score != 1]
        logger.info(f'column scores for the model:\n {column_score}')
        overall_score = np.mean(column_score)
        return column_score, overall_score


class WeightedScoringStrategy(NoWeightScoringStrategy):
    """ A scoring strategy that gives higher precedence to certain labels """

    def __init__(self, weights):
        """ Creates and instance of this class
        :type weights: Dict[str, float]
        :param weights: Dictionary of label names and their associated weights

        """
        super().__init__()
        self.weights = weights

    def score(self, df_predicted, df_actual, drop_perfect_scores=False) -> (Dict[str, float], float):
        col_scores, overall_score = super().score(df_predicted, df_actual, drop_perfect_scores)
        weighted_scores = [col_scores[x] * w for x, w in self.weights.items()]
        return col_scores, np.mean(weighted_scores)
