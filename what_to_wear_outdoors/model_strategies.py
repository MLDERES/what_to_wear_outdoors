import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import logging

from sklearn.utils.validation import check_is_fitted

from what_to_wear_outdoors.scoring_strategies import NoWeightScoringStrategy, WeightedScoringStrategy
from what_to_wear_outdoors.utility import get_model_path, get_model_name

logger = logging.getLogger(__name__)


class IOutfitPredictorStrategy(ABC):

    @abstractmethod
    def save_model(self, athlete='default', version=None):
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict_outfit(self, **kwargs) -> Dict[str, Any]:
        raise NotImplemented

    @abstractmethod
    def predict_outfits(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    @abstractmethod
    def get_model_filename(self, athlete_name='default'):
        raise NotImplemented

    @abstractmethod
    def features(self) -> [str]:
        """
        The names of the parameters used to predict the outcomes (labels)
        :return:
        """
        pass

    @abstractmethod
    def labels(self) -> [str]:
        """
        The names of the columns that are predicted
        :return:
        """
        pass

    @abstractmethod
    def strategy_id(self):
        pass

def load_model(file_path, athlete='default', activity='run', version='Any') -> IOutfitPredictorStrategy:
    """
    Get the pickled OutfitModel from the filename specified.  If no filename specified then athlete_activity.mdl
    """
    # TODO: Implement the version parameter
    fp = f'{athlete}_{activity}' if file_path == '' else file_path
    fp = get_model_path(fp)
    logger.info(f'Loading model from {fp}')
    if not (Path(fp).exists()):
        raise ValueError(f'No such file exists: {fp}')
    with warnings.catch_warnings(), open(fp, 'rb') as mf:
        warnings.simplefilter('ignore')
        this_model = pickle.load(mf)
    return this_model


class BaseOutfitStrategy(IOutfitPredictorStrategy):
    """
    This class encapsulates all the aspects of the prediction model for the particular kind of activity
    """

    def __init__(self, activity, features, labels):
        """
        Create a new
        :param activity:
        :param features:
        """
        self._activity = activity
        self._strategy_id = 'IMPLEMENTED IN BASE CLASS'
        self._features = features
        self._labels = labels
        self._is_fit = False

    def save_model(self, athlete='default', version=None) -> Path:
        """
        Save the current model to the 'models' folder
        :param athlete: unique identifier for the athlete
        :param version: in case there are multiple training datasets used to be this model type multiple times,
            the version identifier can be used to track the changes
        :return:
        """
        # TODO: Implement the version parameter
        # In essense, if version is not None, then use it as the cookie otherwise don't
        model_file_path = get_model_path(self.get_model_filename(athlete))
        pickle.dump(self, open(model_file_path, 'wb'))
        return model_file_path

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict_outfit(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_outfits(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_model_filename(self, athlete='default'):
        return get_model_name(self._activity, cookie=self.strategy_id, athlete=athlete)

    @property
    def strategy_id(self):
        """
        This strategy's identifier
        :return:
        """
        return self._strategy_id

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels


class DualDecisionTreeStrategy(BaseOutfitStrategy):
    """
    This outfit decision making strategy uses two separate models to determine an outfit.  One for boolean outfit
    components and one for categorical components.  Both models are calculated using DecisionTrees
    """
    # _scoring_strategy = NoWeightScoringStrategy()
    _scoring_strategy = WeightedScoringStrategy({'outer_layer': 10, 'base_layer': 10, 'jacket': 5, 'lower_body': 10,
                                                 'ears_hat': 5, 'gloves': 5, 'heavy_socks': 5, 'arm_warmers': 1})

    def __init__(self, activity: str, features: List[str], categorical_targets: Dict[str, List[str]],
                 boolean_labels: List[str]):
        """
        :type activity: str
        :param activity: The name of the activity which this strategy is to be trained
        :type: features: List[str]
        :param features: List of features which should be used to fit this model
        :type categorical_targets: Dict[str, List[str]]
        :param categorical_targets: Dictionary of outcomes that are categorical (i.e. have a fixed number of options).
        The column name is the dictionary key, the values are the possible factors for this label (order assumed)
        :param boolean_labels: List of items that either are (True) or are not (False) appropriate
        given the weather conditions provided


        """
        super(DualDecisionTreeStrategy, self).__init__(activity, features,
                                                       list(categorical_targets.keys()) + boolean_labels)
        self.categorical_labels = list(categorical_targets.keys())
        self.boolean_labels = boolean_labels
        self._categorical_model = self._boolean_model = None
        self._strategy_id = 'ddt'

    def fit(self, df) -> Tuple[MultiOutputClassifier, MultiOutputClassifier]:
        """
        Using this strategy fit the dataframe, df, using the strategy's approach

        :param df: The dataframe with the training data used to train/fit the model

        """
        train_X = self._preprocess_pipeline(df[self.features])

        y_class = df[self.categorical_labels]
        y_bools = df[self.boolean_labels]

        # https://scikit-learn.org/stable/modules/multiclass.html
        # We have to do two models, one for booleans and one for classifiers (which seems dumb)
        # Starting with the booleans
        bool_forest = DecisionTreeClassifier(max_depth=4)
        bool_forest_mo = MultiOutputClassifier(bool_forest)
        bool_forest_mo.fit(train_X, y_bools)
        self._boolean_model = bool_forest_mo

        # Now for the classifiers
        forest = DecisionTreeClassifier(max_depth=4)
        mt_forest = MultiOutputClassifier(forest, n_jobs=-1)
        mt_forest.fit(train_X, y_class)
        self._categorical_model = mt_forest
        self._is_fit = True
        return mt_forest, bool_forest_mo

    # TODO: Handle Imperial and Metric measures (Celsius and kph wind speed)
    def predict_outfit(self, df) -> Dict[str, Union[str, bool]]:
        """ Predict the clothing options that would be appropriate for an outdoor activity.

        :type kwargs: additional forecasting features supported by the particular activity class see
        _supported_features contains this list of useful arguments
        :param duration: length of activity in minutes
        :return: dictionary of outfit components, keys are defined by output components
        """

        self._check_has_been_fit()

        # Now we need to predict using the categorical model and then the boolean model
        #  The categorical model will predict items that can be of more than one type
        #  (i.e. base layer can be sleeveless, short-sleeved or long-sleeve)
        #  boolean is used for categories of 2 or True/False
        #   (i.e. heavy socks, if true then use heavy socks, otherwise we can assume regular socks)
        #   (i.e. arm warmers, if needed True if not then False

        # Filter this list based on what is used to predict the outcomes
        # Run the predictor
        # Put the results into a dictionary that can be returned to the caller
        df = df[self.features]
        logger.debug(f'Peeling off the factors from the supplied prediction parameters ({self.features})')
        cat_labels = self._categorical_model.predict(df).reshape(-1)
        logger.debug(f'Predictions from the category model {cat_labels}')
        bool_labels = self._boolean_model.predict(df).reshape(-1)
        logger.debug(f'Predictions from the boolean model {bool_labels}')

        predictions = np.concatenate((cat_labels, bool_labels), axis=None)
        results = dict(zip(self.labels, predictions))
        logger.debug(f'All the results for the prediction: {results}')

        return results


    def predict_outfits(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Predict multiple outfits given the dataframe of prediction factors


        :param df:
        :return:
        """
        self._check_has_been_fit()
        df = df[self.features]
        cat_labels = self._categorical_model.predict(df)
        bool_labels = self._boolean_model.predict(df)
        predictions = np.hstack((cat_labels, bool_labels))
        results = pd.DataFrame(data=predictions, columns=self.labels)

        return results

    def score(self, df, drop_perfect_scores=False) -> (Dict[str, float], float):
        """ Provide score for the model by outfit component

        :param drop_perfect_scores: If False, then don't consider the labels that are 100% accurate in the scoring.
        This keeps the model from looking better than it is if there are a few labels that aren't actually used (like
        shoe covers for running).
        :param df: Dataframe with the the known good data
        :return: tuple with series of scores (0 - 1.0) for each predicted label
        """
        if self._categorical_model is None or self._boolean_model is None:
            NotFittedError('No instance of this model has been fit yet.')

        predicted = self.predict_outfits(df)
        actual = df[self.categorical_labels + self.boolean_labels]

        return self._scoring_strategy.score(df_predicted=predicted, df_actual=actual,
                                            drop_perfect_scores=drop_perfect_scores)

    def _check_has_been_fit(self) -> None:
        """ Raise NotFittedError if the model has not yet been fit

        :return: None
        """
        if not self._is_fit or self._categorical_model is None or self._boolean_model is None:
            err_msg = f'Attempting to predict prior to training. _is_fit:{self._is_fit} cat_model is built:' \
                '{self._categorical_model is not None} boolean model is built: {self._boolean_model is not None}'
            logger.error(err_msg)
            raise NotFittedError('This model has yet to be trained. Execute the fit function with training'
                                 'data prior to attempting to predict. ')


class SingleDecisionTreeStrategy(BaseOutfitStrategy):
    """
    This outfit decision making strategy uses two separate models to determine an outfit.  One for boolean outfit
    components and one for categorical components.
    """

    def __init__(self, activity, features, labels):
        """

        :param activity:
        """
        super(SingleDecisionTreeStrategy, self).__init__(activity, features, labels)
        self._baseModel = None
        self._strategy_id = 'sdt'
        self._features = features
        self._labels = labels

    def fit(self, df: pd.DataFrame) -> None:
        # Change all of the columns that were of type 'boolean' to 'category'
        bool_cols = df.select_dtypes(['bool']).columns
        df.replace({True: 'Yes', False: 'No'}, inplace=True)
        for x in bool_cols:
            df[x] = df[x].astype('category')
        train_X = df[self.features]
        train_y = df[self.labels]

        fst = DecisionTreeClassifier(max_depth=4)
        fst_mo = MultiOutputClassifier(fst)
        fst_mo.fit(train_X, train_y)
        self._is_fit = True
        self._baseModel = fst_mo

    def predict_outfits(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict_outfit(self, **kwargs):
        pass

    def score(self, df) -> (float):
        """ Provide the scores for the two inner models of this strategy

        :param df:
        :return:
        """
        if self._baseModel is None:
            NotFittedError('No instance of this model has been fit yet.')

        df_factors = df[self.features]
        df_results = df[self.labels]

        model_score = self._baseModel(df_factors, df_results)
        return model_score


class RemoveConstantColumns(TransformerMixin):
    """
    Identify constant columns and enable their removal
    Attributes
    ----------
    const_cols : list
        The list of column names which are constant.
        List may be of length 1.
    """

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` where the constant columns (as identified)
        during the ``fit`` are removed
        Parameters
        ----------
        df : DataFrame
            Should have the same columns as those of the DataFrame used at
            fitting.
        """
        check_is_fitted(self, 'const_cols')
        return df.drop(self.const_cols, axis=1)

    def fit(self, df, y=None, **fit_params):
        """
        Identify the constant columns
        Parameters
        ----------
        df : DataFrame
            Data from which constant features are identified
        """
        self.const_cols = df.loc[:, df.apply(pd.Series.nunique) == 1].columns
        return self


class LabelEncodingColumns(BaseEstimator, TransformerMixin):
    """Label encoding selected columns
    Apply sklearn.preprocessing.LabelEncoder_ to `cols`
    .. _sklearn.preprocessing.LabelEncoder : https://is.gd/Vx2njl
    Attributes
    ----------
    cols : list
        List of columns in the data to be scaled
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.les = {col: LabelEncoder() for col in cols}
        self._is_fitted = False

    def transform(self, df, **transform_params):
        """
        Label encoding ``cols`` of ``df`` using the fitting
        Parameters
        ----------
        df : DataFrame
            DataFrame to be preprocessed
        """
        if not self._is_fitted:
            raise NotFittedError("Fitting was not preformed")
        is_cols_subset_of_df(self.cols, df)

        df = df.copy()

        label_enc_dict = {}
        for col in self.cols:
            label_enc_dict[col] = self.les[col].transform(df[col])

        labelenc_cols = pd.DataFrame(label_enc_dict,
                                     # The index of the resulting DataFrame should be assigned and
                                     # equal to the one of the original DataFrame. Otherwise, upon
                                     # concatenation NaNs will be introduced.
                                     index=df.index
                                     )

        for col in self.cols:
            df[col] = labelenc_cols[col]
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Fitting the preprocessing
        Parameters
        ----------
        df : DataFrame
            Data to use for fitting.
            In many cases, should be ``X_train``.
        """
        is_cols_subset_of_df(self.cols, df)
        for col in self.cols:
            self.les[col].fit(df[col])
        self._is_fitted = True
        return self


def is_cols_subset_of_df(cols, df):
    if not set(cols).issubset(set(df.columns)):
        raise ValueError('Class instantiated with columns not in the dataframe provided.')
