import logging
import pickle
import typing

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import iqr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

class LRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init LRTransformer')
        self.lr = LogisticRegressionCV(Cs=100, class_weight=None, max_iter=1000, random_state=42)

    def fit(self, X, y=None):
        print('fit LRTransformer')
        iqr_ = iqr(y)
        q25 = np.quantile(y, 0.25)
        q50 = np.quantile(y, 0.50)
        q75 = np.quantile(y, 0.75)
        target = np.array(list(map(lambda x: 1 if x < q25 else 2 if x < q50 else 3 if x < q75 else 4, y)))
        self.lr.fit(X, target)
        return self

    def transform(self, X, y=None):
        print('transform LRTransformer')
        X_ = X.copy()
        pred = self.lr.predict_proba(X)[:, 1]
        X_ = np.hstack((X_, np.atleast_2d(pred).T))
        return X_

class CityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init CityTransformer')
        self.city_dict = None

    def fit(self, X, y=None):
        print('fit CityTransformer')
        df = X.join(y)
        self.city_dict = np.round(df.groupby(['city'])['per_square_meter_price'].median()).astype(int).to_dict()
        return self

    def transform(self, X, y=None):
        print('transform CityTransformer')
        X_ = X.copy()
        X_['city'] = X_['city'].map(self.city_dict)
        return X_

class KBestSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init KBestSelector')
        self.k_best = SelectKBest(score_func=mutual_info_regression, k=50)

    def fit(self, X, y=None):
        print('fit KBestSelector')
        self.k_best.fit(X, y)
        return self

    def transform(self, X, y=None):
        print('transform KBestSelector')
        X_ = self.k_best.transform(X)
        return X_

class BenchmarkModel():
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 binary_features: typing.List[str],
                 model_params: typing.Dict[str, typing.Union[str,int,float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features
        self.bin_features = binary_features

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', MinMaxScaler(), self.num_features),
            ('bin', 'passthrough', self.bin_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), self.ohe_cat_features),
            ('ste', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.ste_cat_features),
        ])

        self.model = LGBMRegressor(**model_params)
        # models = [
        #     ('cat_boost', CatBoostRegressor(depth=6, learning_rate=0.1, l2_leaf_reg=9, iterations=1000,
        #                                     loss_function='MAPE', verbose=0, random_state=42)),
        #     ('RidgeCV', RidgeCV())
        # ]
        # final_model =  LGBMRegressor(**model_params)
        # self.model = StackingRegressor(estimators=models, final_estimator=final_model,
        #                                cv=5, passthrough=True, verbose=2)

        self.lr = LRTransformer()
        # self.k_best = KBestSelector()

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
#            ('lr', self.lr),
            ('model', self.model)])

        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.pipeline.predict(X_manual)**2
        deviation = ((y_manual - predictions) / predictions).median()
        self.corr_coef = deviation

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series,
            X_manual: pd.DataFrame, y_manual: pd.Series):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info(f'Fit lightgbm {X_offer.shape}')
        self.pipeline.fit(pd.concat([X_offer, X_manual]), pd.concat([y_offer**0.5, y_manual**0.5]))
        logger.info('Find corr coefficient')
        self._find_corr_coefficient(X_manual, y_manual)
        logger.info(f'Corr coef: {self.corr_coef:.2f}')
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            predictions = np.floor(self.pipeline.predict(X)**2 / 100) * 100
            corrected_price = predictions * (1 + self.corr_coef)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model