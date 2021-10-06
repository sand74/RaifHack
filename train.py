import argparse
import logging.config
from traceback import format_exc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from raifhack_ds.features import prepare_categorical, generate_features
from raifhack_ds.metrics import metrics_stat, deviation_metric_one_sample
from raifhack_ds.model import BenchmarkModel
from raifhack_ds.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES, TARGET, BIN_FEATURES
from raifhack_ds.utils import PriceTypeEnum

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        columns = np.delete(train_df.columns.values, np.where(train_df.columns.values == 'id'))
        train_df.drop_duplicates(subset=columns, inplace=True)
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)
        train_df = generate_features(train_df)

        train_m, test_m = train_test_split(train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE], test_size=0.2, random_state=42)

        X_offer = train_m[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES+BIN_FEATURES].copy()
        y_offer = train_m[TARGET].copy()
        X_manual = test_m[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES+BIN_FEATURES].copy()
        y_manual = test_m[TARGET].copy()
        # X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES+BIN_FEATURES]
        # y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        # X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES+BIN_FEATURES]
        # y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                               ste_categorical_features=CATEGORICAL_STE_FEATURES,  binary_features=BIN_FEATURES,
                               model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(args['mp'])

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')
        X_offer['y_pred'] = predictions_offer/(1+model.corr_coef)
        X_offer['y_true'] = y_offer
        X_offer['metric'] = X_offer.apply(lambda row: deviation_metric_one_sample(row['y_true'], row['y_pred']), axis=1)
        X_offer.to_csv('X_offer.csv')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')
        X_manual['y_pred'] = predictions_manual
        X_manual['y_true'] = y_manual
        X_manual['metric'] = X_manual.apply(lambda row: deviation_metric_one_sample(row['y_true'], row['y_pred']), axis=1)
        X_manual.to_csv('X_manual.csv')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')