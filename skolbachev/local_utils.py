import os, sys, gc, re, datetime, unidecode, pickle, tqdm, math, random, itertools, multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from IPython.lib.display import FileLink
cpu_cores = multiprocessing.cpu_count()

import math, random
import pickle
import scipy
import bcolz
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import scale, minmax_scale, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
pd.set_option('float_format', '{:f}'.format)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='ticks', context='talk')

import tensorflow as tf
import keras
from keras.layers import *
from keras.layers.core import Activation
from keras.models import *
from keras.callbacks import *
from keras.constraints import *
from keras.regularizers import *
from keras.utils import Sequence

from keras import backend as K

cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

data_dir = 'data/'
models_dir = 'models/'
results_dir = 'results/'

# Seed
seed = 7961730
print("Seed: {}".format(seed))

def fit_transform(df_data, minmax=False, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range) if minmax else StandardScaler()
    return scaler.fit_transform(df_data.values.reshape(-1, 1)).reshape(-1)

def fit_transform_cols(df, cols, new_cols, groupby=None, minmax=False, feature_range=(0,1)):
    for col, new_col in zip(cols, new_cols):
        if groupby == None:
            df[new_col] = fit_transform(df[col], minmax, feature_range)
        else:
            df[new_col] = df.groupby(groupby)[col].transform(lambda group: fit_transform(group, minmax, feature_range))
    return df

def factorize_cols(df, cols, new_cols):
    for col, new_col in zip(cols, new_cols):
        df[new_col] = df[col].factorize()[0]
    return df

def binarize_cols(df, cols, new_cols):
    for col, new_col in zip(cols, new_cols):
        df[new_col] = pd.get_dummies(df[col], drop_first=True).values
    return df

def fit_scalers(df, col, forecast_scalers, site_scalers, minmax=False, feature_range=(0,1)):
    for group_name, group in df.groupby("forecast_id")[col]:
        scaler = MinMaxScaler(feature_range=feature_range) if minmax else StandardScaler()
        scaler.fit(group.values.reshape(-1, 1))
        forecast_scalers[group_name] = scaler
        
    for group_name, group in df.groupby("site_id")[col]:
        scaler = MinMaxScaler(feature_range=feature_range) if minmax else StandardScaler()
        scaler.fit(group.values.reshape(-1, 1))
        site_scalers[group_name] = scaler

    return df

def transform_group(group, forecast_scalers, site_scalers):
    try: 
        return forecast_scalers[group.name[0]].transform(group.values.reshape(-1, 1)).reshape(-1)
    except KeyError: 
        return site_scalers[group.name[1]].transform(group.values.reshape(-1, 1)).reshape(-1)

def transform(df, col, new_col, forecast_scalers, site_scalers):
    df[new_col] = df.groupby(["forecast_id", "site_id"])[col].transform(lambda group: transform_group(group, forecast_scalers, site_scalers))
    return df

def inverse_group(group, forecast_scalers, site_scalers):
    try: 
        return forecast_scalers[group.name[0]].inverse_transform(group.values.reshape(-1, 1)).reshape(-1)
    except KeyError: 
        return site_scalers[group.name[1]].inverse_transform(group.values.reshape(-1, 1)).reshape(-1)

def inverse(df, col, new_col, forecast_scalers, site_scalers):
    df[new_col] = df.groupby(["forecast_id", "site_id"])[col].transform(lambda group: inverse_group(group, forecast_scalers, site_scalers))
    return df

def fit_maxlog_scalers(df, col, forecast_scalers, site_scalers):
    for group_name, group in df.groupby("forecast_id")[col]:
        maxlog = np.max(np.log(np.maximum(group, 1)))
        forecast_scalers[group_name] = maxlog
     
    for group_name, group in df.groupby("site_id")[col]:
        maxlog = np.max(np.log(np.maximum(group, 1)))
        site_scalers[group_name] = maxlog
    
    return df

def transform_maxlog_group(group, forecast_scalers, site_scalers):
    try: 
        return np.log(np.maximum(group.values, 1))/(forecast_scalers[group.name[0]]+1e-15)
    except KeyError: 
        return np.log(np.maximum(group.values, 1))/(site_scalers[group.name[1]]+1e-15)

def transform_maxlog(df, col, new_col, forecast_scalers, site_scalers):
    df[new_col] = df.groupby(["forecast_id", "site_id"])[col].transform(lambda group: transform_maxlog_group(group, forecast_scalers, site_scalers))
    return df

def inverse_maxlog_group(group, forecast_scalers, site_scalers):
    try: 
        return np.exp(group.values*forecast_scalers[group.name[0]])
    except KeyError: 
        return np.exp(group.values*site_scalers[group.name[1]])

def inverse_maxlog(df, col, new_col, forecast_scalers, site_scalers):
    df[new_col] = df.groupby(["forecast_id", "site_id"])[col].transform(lambda group: inverse_maxlog_group(group, forecast_scalers, site_scalers))
    return df

def keras_rmse(y_true, y_pred):
    return K.sqrt(keras.losses.mean_squared_error(y_true, y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

class MSEEvaluation(Callback):
    def __init__(self, X, y, name, batch_size, interval=1):
        super(Callback, self).__init__()

        self.X, self.y = X, y
        self.name = name
        self.batch_size = batch_size
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X, batch_size=self.batch_size, verbose=0)
            mse = metrics.mean_squared_error(self.y, y_pred)
            rmse = np.sqrt(mse)
            logs[self.name+"_mse"] = mse
            logs[self.name+"_rmse"] = rmse
            print((self.name+"_mse: {:.8f}; "+self.name+"_rmse: {:.8f};").format(mse, rmse))

def WRMSE(y, pred_y):
    sq_err = np.square(y - pred_y)
    Tn = len(y)
    t = np.arange(1, Tn+1)
    W = (3*Tn - 2*t + 1)/(2*Tn**2)
    mu = y.mean()+1e-15
    return np.sqrt(np.sum(W*sq_err))/mu

def metric_by_group(df, metric_fun, target_col, pred_col, groupby="forecast_id"):
    return df.groupby(groupby).apply(lambda group: metric_fun(group[target_col], group[pred_col]))

def print_report(df, target_col, pred_col, groupby="forecast_id"):
    print("\nNWRMSE: {}; \nRMSE: {}; \nMSE: {}; \nMAE: {}; \nR2: {};"
          .format(metric_by_group(df, WRMSE, target_col, pred_col, groupby).mean(),
                  metric_by_group(df, rmse, target_col, pred_col, groupby).mean(),
                  metric_by_group(df, metrics.mean_squared_error, target_col, pred_col, groupby).mean(), 
                  metric_by_group(df, metrics.mean_absolute_error, target_col, pred_col, groupby).mean(), 
                  metric_by_group(df, metrics.r2_score, target_col, pred_col, groupby).mean()))

def plot_graph(df, cond_id, cond_col="forecast_id", target="value"):
    df[df[cond_col] == cond_id].plot(x="timestamp", y=target, figsize=(18, 6), label=cond_id)
    plt.legend(bbox_to_anchor=(1.0, .5))
    plt.xlabel("timestamp")
    plt.ylabel(target)
    sns.despine()

def split_inputs(X):
    return np.split(X, X.shape[-1], axis=-1)

class FeatureSequence(Sequence):
    
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X, self.y = X, y
        self.batch_size = batch_size
        
        self.inx = np.arange(0, self.X.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]

        return split_inputs(self.X[batch_inx]), self.y[batch_inx]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)