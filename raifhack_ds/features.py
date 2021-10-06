import logging.config
import re
import string

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from raifhack_ds.settings import CATEGORICAL_OHE_FEATURES, LOGGING_CONFIG, CATEGORICAL_STE_FEATURES, NUM_FEATURES, \
    BIN_FEATURES


def correct_floor(floor) -> str:
    '''
    Преобразование этажа в номинатив
    '''
    floors = str(floor).lower().split(',')
    if len(floors) > 1:
        return 'manylevels'
    floor_0 = floors[0].strip()
    if floor_0.endswith('.0'):
        if int(floor_0[0:-2]) < 0:
            return 'underground'
        elif int(floor_0[0:-2]) == 0:
            return 'ground'
        elif int(floor_0[0:-2]) == 1:
            return 'first'
        elif int(floor_0[0:-2]) == 2:
            return 'second'
        elif int(floor_0[0:-2]) == 3:
            return 'third'
        else:
            return 'high'
    elif 'тех' in floor_0:
        return 'tech'
    elif 'подва' in floor_0:
        return 'underground'
    elif 'мансард' in floor_0:
        return 'high'
    elif 'цоколь' in floor_0:
        return 'ground'
    elif 'антресоль' in floor_0:
        return 'high'
    elif 'мезонин' in floor_0:
        return 'high'
    elif '-' in floor_0:
        return 'manylevels'
    elif '.' in floor_0:
        return 'manylevels'
    elif floor_0 == 'nan':
        return 'unknown'
    else:
        return 'other'

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()

    # TF-IDF + K-Means
    def preprocessing(line):
        line = line.lower()
        line = re.sub(r"[{}]".format(string.punctuation), " ", line)
        return line

    # tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 5))
    # tfidf = tfidf_vectorizer.fit_transform(df_new['floor'].astype(str))
    # for i in range(2, 3):
    #     clustering = kmeans = KMeans(n_clusters=i).fit_predict(tfidf)
    #     unique, counts = np.unique(clustering, return_counts=True)
    #     logger.info(f'clusters = {i}, counts = {counts}')
    #     df_new[f'floor_cluster_{i}'] = clustering
    #     CATEGORICAL_OHE_FEATURES.append(f'floor_cluster_{i}')

    # tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf).astype(int)
    # tfidf_df = tfidf_df.set_index(df_new.index)
    # tfidf_df.columns = [f'tfidf_{c}' for c in tfidf_df.columns]
    # tfidf_df.fillna(0, inplace=True)
    # df_new = df_new.join(tfidf_df)
    # NUM_FEATURES.extend(tfidf_df.columns)

    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # X = StandardScaler().fit_transform(df_new[['lat', 'lng']])
    # dbscan.fit(X)
    # y_pred = dbscan.labels_.astype(np.int)
    # uniq = np.unique(y_pred, return_counts=True)
    # logger.info(f'Add latlon_cluster, counts = {uniq}')
    # df_new[f'latlon_cluster'] = y_pred
    # CATEGORICAL_OHE_FEATURES.append(f'latlon_cluster')

    # X = StandardScaler().fit_transform(df_new[['lat', 'lng']])
    # connectivity = kneighbors_graph(X, 10, include_self=False)
    # # делаем матрицу смежности симметричной
    # connectivity = 0.5 * (connectivity + connectivity.T)
    # ac = AgglomerativeClustering(linkage='average', n_clusters=35, connectivity=connectivity)
    # ac.fit(X)
    # y_pred = ac.labels_.astype(np.int)
    # uniq = np.unique(y_pred, return_counts=True)
    # logger.info(f'Add latlon_cluster, counts = {uniq}')
    # df_new[f'latlon_cluster'] = y_pred
    # CATEGORICAL_OHE_FEATURES.append(f'latlon_cluster')

    df_new['floor'] = df_new['floor'].apply(lambda x: correct_floor(x))
    df_new['loc'] = (df_new['lng'] // 10) * 10 + (df_new['lat'] // 10)
    CATEGORICAL_OHE_FEATURES.append('loc')
    df_new['latlng'] = df_new['lat'] * df_new['lng']
    NUM_FEATURES.append('latlng')
    df_new['ts_eq_10'] = (df_new['total_square'] % 10) == 0
    BIN_FEATURES.append('ts_eq_10')
    df_new['ts_eq_1000'] = (df_new['total_square'] % 1000) == 0
    BIN_FEATURES.append('ts_eq_1000')

    # num_features_ = NUM_FEATURES.copy()
    # for feature in NUM_FEATURES:
    #     if feature.startswith('osm_'):
    #         df_new[feature] = df_new[feature] > 0
    #         num_features_.remove(feature)
    #         BIN_FEATURES.append(feature)
    # NUM_FEATURES.clear()
    # NUM_FEATURES.extend(num_features_)

    return df_new


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    df_new['osm_city_nearest_population'].fillna(df_new['osm_city_nearest_population'].median(), inplace=True)
    df_new['reform_house_population_1000'].fillna(df_new['reform_house_population_1000'].median(), inplace=True)
    df_new['reform_house_population_500'].fillna(df_new['reform_house_population_500'].median(), inplace=True)
    df_new['reform_mean_floor_count_1000'].fillna(df_new['reform_mean_floor_count_1000'].median(), inplace=True)
    df_new['reform_mean_floor_count_500'].fillna(df_new['reform_mean_floor_count_500'].median(), inplace=True)
    df_new['reform_mean_year_building_1000'].fillna(df_new['reform_mean_year_building_1000'].median(), inplace=True)
    df_new['reform_mean_year_building_500'].fillna(df_new['reform_mean_year_building_500'].median(), inplace=True)
    df_new['street'].fillna('S12711', inplace=True)

    # CATEGORICAL_STE_FEATURES = ['city']
    # CATEGORICAL_OHE_FEATURES = ['region', 'realty_type', 'floor', 'street']
    for feature in CATEGORICAL_STE_FEATURES + CATEGORICAL_OHE_FEATURES:
        df_new[feature] = df_new[feature].apply(lambda x: str(x).lower().strip())

    # city_dict = np.round(df_new.groupby(['city'])['per_square_meter_price'].median()).astype(int).to_dict()
    # df_new['city'] = df_new['city'].map(city_dict)

    # for feature in CATEGORICAL_STE_FEATURES + CATEGORICAL_OHE_FEATURES:
    #     vals = df_new[feature].value_counts()[df_new[feature].value_counts() < 2].index.values
    #     df_new.loc[df_new[feature].isin(vals), feature] = 'other'
    # df_new['total_square'] = np.log(df_new['total_square'])

    return df_new