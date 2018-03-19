import pandas as pd
from holidayList import hol_list
from sklearn import  metrics
from math import sqrt
import tqdm
import time
import numpy as np

target = 'Value'

def parse_date(df,
               date_col,
               new_col,
               year_col=None,
               month_col=None,
               day_col=None,
               hour_col=None,
               minute_col=None,
               second_col=None,
               dow=None):
    df[new_col] = pd.to_datetime(df[date_col])
    if year_col is not None:
        df[year_col] = df[new_col].dt.year
    if month_col is not None:
        df[month_col] = df[new_col].dt.month        
    if day_col is not None:
        df[day_col] = df[new_col].dt.day
    if hour_col is not None:
        df[hour_col] = df[new_col].dt.hour
    if minute_col is not None:
        df[minute_col] = df[new_col].dt.minute        
    if second_col is not None:    
        df[second_col] = df[new_col].dt.second
    if dow is not None:    
        df[dow] = df[new_col].dt.dayofweek           
    return df
def date_to_int(df,
               date_col,
               new_col):
    df[new_col] = pd.to_timedelta(df[date_col]).dt.total_seconds().astype(int)
    return df
def clean_ids(df,column_name):
    df[column_name].fillna(-1, inplace=True)
    df[column_name] = df[column_name].astype(str)
    return df
def process_categorical_features(df,feature_list):
    for feature in feature_list:
        df[feature].fillna(-1, inplace=True) 
        le, u = df[feature].factorize()
        df[feature] = le
    return df
def add_forecast_id_type(df):
    f_ids = list(df.ForecastId.unique())

    # understand which forecast_is has which time periodicity
    f_id_dict = {}

    for forecast_id in f_ids:
        if len(df[df.ForecastId == forecast_id].hour.unique()) == 1:
            f_id_dict[forecast_id] = '1_day'
        elif len(df[df.ForecastId == forecast_id].minute.unique()) == 4:
            f_id_dict[forecast_id] = '15_mins'
        else:
            f_id_dict[forecast_id] = '1_hour'

    # keys = [(item[0]) for item in list(f_id_dict.items())]
    # values = [(item[1]) for item in list(f_id_dict.items())]
    # f_id_df = pd.DataFrame()
    # f_id_df['ForecastId'] = keys
    # f_id_df['ForecastId_type'] = values    

    df['ForecastId_type'] = df['ForecastId'].apply(lambda x: f_id_dict[x])

    return df
def process_days_off(df):
    days = ['MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']
    df['is_day_off'] = 0
    # add all days off to a dummy variable
    for i,day in enumerate(days):
        df.loc[(df[day] == True)&(df.dow == i),'is_day_off'] = 1
        del df[day]
    return df
def convert_to_dummies(df,col):
    dummy_df = pd.get_dummies(df[col], prefix=['hol'])
    for column in dummy_df.columns:
        df[column] = dummy_df[column]
        
    df = df.groupby(by = ['Date','SiteId', 'ts', 'year', 'month', 'day'])[hol_list].sum()
    return df
def ETL():
    df_train = (pd.read_csv('../data/forecast/train.csv')
                .pipe(parse_date,'Timestamp','ts','year','month','day','hour','minute','second','dow')
                .pipe(add_forecast_id_type)
                .pipe(date_to_int,'ts','ts_trend')
                )

    df_hol = (pd.read_csv('../data/forecast/holidays.csv')
            .drop('Unnamed: 0', axis=1)
            .drop_duplicates()          
            .pipe(parse_date,'Date','ts',
                            'year','month','day')
            .drop_duplicates()
            # .pipe(process_categorical_features,['Holiday']) sometimes there are 2 holidays per day ...
            .pipe(convert_to_dummies,['Holiday'])
            .reset_index()
            )

    df_weather = (pd.read_csv('../data/forecast/weather.csv')
                  .drop('Unnamed: 0', axis=1)
                  .drop('Distance', axis=1)
                  .drop_duplicates()
                  .pipe(parse_date,'Timestamp','ts',
                        'year','month','day','hour')
                  .groupby(['year','month','day','hour','SiteId'])['Temperature'].mean()
                  .reset_index()
                 )

    df_meta = (pd.read_csv('../data/forecast/metadata.csv')
            .drop_duplicates()          
            )

    # add meta-data
    cols = ['SiteId','Surface','BaseTemperature','MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']
    join_keys = ['SiteId']
    df_train = df_train.merge(df_meta[cols], on=join_keys)
    # process days off
    df_train = process_days_off(df_train)

    # add weather by composite key
    cols = ['year','month','day','hour','Temperature','SiteId']
    join_keys = ['year','month','day','hour','SiteId']
    df_train = df_train.merge(df_weather[cols], on=join_keys, how='left')
    # fill missing values with -1
    df_train['Temperature'].fillna(-1, inplace=True)


    # join holidays by composite key
    cols = ['SiteId', 'year', 'month', 'day'] + hol_list
    join_keys = ['year','month','day','SiteId']
    df_train = df_train.merge(df_hol[cols], on=join_keys, how='left')


    # ETL pipeline for test data
    df_sub = (pd.read_csv('../data/forecast/submission_format.csv')
                .pipe(parse_date,'Timestamp','ts','year','month','day','hour','minute','second','dow')
                .pipe(add_forecast_id_type)
                .pipe(date_to_int,'ts','ts_trend')
                )

    # add meta-data
    cols = ['SiteId','Surface','BaseTemperature','MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']
    join_keys = ['SiteId']
    df_sub = df_sub.merge(df_meta[cols], on=join_keys)
    # process days off
    df_sub = process_days_off(df_sub)

    # add weather by composite key
    cols = ['year','month','day','hour','Temperature','SiteId']
    join_keys = ['year','month','day','hour','SiteId']
    df_sub = df_sub.merge(df_weather[cols], on=join_keys, how='left')
    # fill missing values with -1
    df_sub['Temperature'].fillna(-1, inplace=True)

    # join holidays by composite key
    cols = ['SiteId', 'year', 'month', 'day'] + hol_list
    join_keys = ['year','month','day','SiteId']
    df_sub = df_sub.merge(df_hol[cols], on=join_keys, how='left')

    del df_hol,df_weather,df_meta

    # fill all blank holidays with zeroes
    for holiday in hol_list:
        df_train[holiday] = df_train[holiday].fillna(value=0)
        df_sub[holiday] = df_sub[holiday].fillna(value=0)

    df_train['temp_diff'] = df_train['BaseTemperature'] - df_train['Temperature']
    df_sub['temp_diff'] = df_sub['BaseTemperature'] - df_sub['Temperature']
    
    return df_train,df_sub
def count_test_period(df):
    f_ids = list(df.ForecastId.unique())

    # calculate length of test set for each forecast_id
    f_id_dict = {}

    for forecast_id in f_ids:
        f_id_dict[forecast_id] = df[df.ForecastId == forecast_id].shape[0]
        
    return f_id_dict
def xgb_fit_log(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    rmse_train = sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions))
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.get_booster().get_fscore()
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores
def produce_ar_features(df):
    df['Value1'] = df['Value'].shift(+1)
    df['Value2'] = df['Value'].shift(+2)
    df['Value3'] = df['Value'].shift(+3)
    df['Value4'] = df['Value'].shift(+4)    
    df['Value5'] = df['Value'].shift(+5)    
    df['Value10'] = df['Value'].shift(+10)
    df['dValue1'] = df['Value'] - df['Value'].shift(+1)
    
    # df['dValue2'] = df['dValue1'] - df['dValue1'].shift(+1)

    df['T1'] = df['Temperature'].shift(+1)
    df['T5'] = df['Temperature'].shift(+5)
    df['T10'] = df['Temperature'].shift(+10)

    # df['dT1'] = df['Temperature'] - df['Temperature'].shift(+1)
    # df['dT2'] = df['dT1'] - df['dT1'].shift(+1)
    # df['T1_diff'] = df['temp_diff'].shift(+1)
    # df['T10_diff'] = df['temp_diff'].shift(+10)
    # df['dT1_diff'] = df['temp_diff'] - df['temp_diff'].shift(+1)
    # df['dT2_diff'] = df['dT1_diff'] - df['dT1_diff'].shift(+1) 
    return df
def produce_weather_future(df):
    # fill all the -1 with NA and then interpolate
    df.loc[df['Temperature']==-1,'Temperature'] = np.nan
    df['Temperature'] = df['Temperature'].fillna(method='ffill')
    df['Temperature'] = df['Temperature'].fillna(method='bfill')
    
    df['TF1'] = df['Temperature'].shift(-1)
    df['TF5'] = df['Temperature'].shift(-5)
    df['TF10'] = df['Temperature'].shift(-10)
    df['TF50'] = df['Temperature'].shift(-50)
    
    df['TF1'] = df['TF1'].fillna(df['TF1'].mean())
    df['TF5'] = df['TF5'].fillna(df['TF5'].mean())
    df['TF10'] = df['TF10'].fillna(df['TF10'].mean())
    df['TF50'] = df['TF50'].fillna(df['TF50'].mean())
    
    df['Temperature'] = df['Temperature'].fillna(value = -100)
    df['TF1'] = df['Temperature'].fillna(value = -100)
    df['TF5'] = df['Temperature'].fillna(value = -100)
    df['TF10'] = df['Temperature'].fillna(value = -100)
    df['TF50'] = df['Temperature'].fillna(value = -100)

    return df
def cb_fit_log(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    print(dtrain_predictions.shape,dtest_predictions.shape)
    
    rmse_train = sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions))
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.get_feature_importance(dtrain[predictors], dtrain[target])
    scores = dict(zip(predictors, scores))
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores
def cb_fit_log_alg(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    rmse_train = sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions))
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.get_feature_importance(dtrain[predictors], dtrain[target])
    scores = dict(zip(predictors, scores))
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def lgb_fit_log_alg(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    rmse_train = sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions))
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.booster_.feature_importance()
    scores = dict(zip(predictors, scores))
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def produce_ar_step(df, current_index):
    df.loc[current_index,'Value1'] = df.loc[current_index - 1,'Value']
    return df
def cb_predict_test(alg,
             df,
             predictors,
             test_ind,
             train_ind):
    
    df.loc[test_ind,'Value'] = 0
    
    for i,current_index in enumerate(test_ind):
        # calculate AR features  forward
        # (except temperature, which is already calculated)
        df = produce_ar_step(df, current_index)

        df.loc[current_index,'Value'] = alg.predict(df.filter(items=[current_index],axis=0)[predictors])
    
    return df
def lgb_predict_test(alg,
             df,
             predictors,
             test_ind,
             train_ind):
    
    df.loc[test_ind,'Value'] = 0
    
    for i,current_index in enumerate(test_ind):
        # calculate AR features  forward
        # (except temperature, which is already calculated)
        df = produce_ar_step(df, current_index)
        df.loc[current_index,'Value'] = alg.predict(df.filter(items=[current_index],axis=0)[predictors])
    return df
def mlp_fit_log_alg(alg,
             X_train,y_train,
             X_val,y_val,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtest_predictions = alg.predict(X_val)
    
    rmse_train = sqrt(metrics.mean_squared_error(y_train, dtrain_predictions))
    r2_score_train = metrics.r2_score(y_train, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(y_val, dtest_predictions))
    r2_score_val = metrics.r2_score(y_val, dtest_predictions)
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,alg
def slice_train_data_rf(df_train,
                        df_sub,
                        forecast_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min 
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        val_ind = list(temp_df.index)[:test_lengths[forecast_id]//3]
        train_ind = list(temp_df.index)[test_lengths[forecast_id]//3:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        val_ind = list(temp_df.index)[:test_lengths[forecast_id]//2]
        train_ind = list(temp_df.index)[test_lengths[forecast_id]//2:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]            
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors
def slice_train_data_rf_weather(df_train,
                                df_sub,
                                forecast_id,
                                test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    weather_future_features = ['TF1','TF5','TF10','TF50']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = produce_weather_future(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day + weather_future_features           
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour + weather_future_features
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min + weather_future_features
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        val_ind = list(temp_df.index)[:test_lengths[forecast_id]//3]
        train_ind = list(temp_df.index)[test_lengths[forecast_id]//3:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        val_ind = list(temp_df.index)[:test_lengths[forecast_id]//2]
        train_ind = list(temp_df.index)[test_lengths[forecast_id]//2:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]            
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors  
def slice_train_data_rf_reverse(df_train,
                        df_sub,
                        forecast_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min 
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]-test_lengths[forecast_id]//3]
        val_ind = list(temp_df.index)[-test_lengths[forecast_id]-test_lengths[forecast_id]//3:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]-test_lengths[forecast_id]//2]
        val_ind = list(temp_df.index)[-test_lengths[forecast_id]-test_lengths[forecast_id]//2:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]            
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors
def slice_train_data_rf_weather_reverse(df_train,
                                df_sub,
                                forecast_id,
                                test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    weather_future_features = ['TF1','TF5','TF10','TF50']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = produce_weather_future(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day + weather_future_features           
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour + weather_future_features
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min + weather_future_features
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]-test_lengths[forecast_id]//3]
        val_ind = list(temp_df.index)[-test_lengths[forecast_id]-test_lengths[forecast_id]//3:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]-test_lengths[forecast_id]//2]
        val_ind = list(temp_df.index)[-test_lengths[forecast_id]-test_lengths[forecast_id]//2:-test_lengths[forecast_id]]
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]        
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors  
def sk_fit_alg(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
   
    rmse_train = sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions))
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = []
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def slice_train_data_rf_full(df_train,
                        df_sub,
                        forecast_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min 
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]]
        val_ind = []
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]]
        val_ind = []
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]       
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors
def slice_train_data_rf_weather_full(df_train,
                                df_sub,
                                forecast_id,
                                test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    weather_future_features = ['TF1','TF5','TF10','TF50']
    time_features_day = ['dow']
    time_features_hour = ['dow','hour']
    time_features_15min = ['dow','hour','minute'] 
    
    temp_df = df_train[(df_train.ForecastId == forecast_id)
                        &(pd.notnull(df_train.Value))]
    
    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])
    temp_df = produce_ar_features(temp_df)
    temp_df = produce_weather_future(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])
    
    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day + weather_future_features           
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour + weather_future_features
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min + weather_future_features
    
    # shorter val dataset for short time series
    if temp_df.shape[0]<500:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]]
        val_ind = []
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]                
    else:
        train_ind = list(temp_df.index)[:-test_lengths[forecast_id]]
        val_ind = []
        test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]       
    
    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors  
def WRMSE(y, pred_y):
    sq_err = np.square(y - pred_y)
    Tn = len(y)
    t = np.arange(1, Tn+1)
    W = (3*Tn - 2*t + 1)/(2*Tn**2)
    mu = y.mean()+1e-15
    return np.sqrt(np.sum(W*sq_err))/mu
def sk_fit_alg_wrmse(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
   
    rmse_train = WRMSE(dtrain[target].values, dtrain_predictions)
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = WRMSE(dtest[target].values, dtest_predictions)
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = []
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def cb_fit_log_alg_wrmse(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    rmse_train = WRMSE(dtrain[target].values, dtrain_predictions)
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = WRMSE(dtest[target].values, dtest_predictions)
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.get_feature_importance(dtrain[predictors], dtrain[target])
    scores = dict(zip(predictors, scores))
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def lgb_fit_log_alg_wrmse(alg,
             dtrain,
             dtest,
             predictors):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    
    rmse_train = WRMSE(dtrain[target].values, dtrain_predictions)
    r2_score_train = metrics.r2_score(dtrain[target].values, dtrain_predictions)
    rmse_val = WRMSE(dtest[target].values, dtest_predictions)
    r2_score_val = metrics.r2_score(dtest[target].values, dtest_predictions)
    scores = alg.booster_.feature_importance()
    scores = dict(zip(predictors, scores))
    
    return rmse_train,r2_score_train,rmse_val,r2_score_val,scores,alg
def slice_train_data_rf_reverse_site(df_train,
                        df_sub,
                        site_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow','month','day']
    time_features_hour = ['dow','hour','month','day']
    time_features_15min = ['dow','hour','minute','month','day'] 

    temp_df = df_train[(df_train.SiteId == site_id)
                        &(pd.notnull(df_train.Value))]

    temp_df['is_train'] = 1

    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.SiteId == site_id)])
    # sort data to ensure consistency for producing weather time features 
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'] )         
    temp_df = temp_df.sort_values(by = 'Timestamp')
    local_forecast_ids = temp_df.ForecastId.unique()

    temp_df = produce_ar_features(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])

    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min 


    train_ind = []
    val_ind = []

    for local_forcast_id in local_forecast_ids:
        local_temp_df = temp_df[temp_df.ForecastId == local_forcast_id]
        train_ind.extend(list(local_temp_df.index)[:-test_lengths[local_forcast_id]-test_lengths[local_forcast_id]//2])
        val_ind.extend(list(local_temp_df.index)[-test_lengths[local_forcast_id]-test_lengths[local_forcast_id]//2 :-test_lengths[local_forcast_id]])

    test_ind = list(temp_df[pd.isnull(temp_df.is_train)].index)              

    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors    
def slice_train_data_rf_reverse_site_weather(df_train,
                        df_sub,
                        site_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow','month','day']
    time_features_hour = ['dow','hour','month','day']
    time_features_15min = ['dow','hour','minute','month','day'] 
    weather_future_features = ['TF1','TF5','TF10','TF50']    

    temp_df = df_train[(df_train.SiteId == site_id)
                        &(pd.notnull(df_train.Value))]

    temp_df['is_train'] = 1

    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.SiteId == site_id)])
    # sort data to ensure consistency for producing weather time features 
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'] )         
    temp_df = temp_df.sort_values(by = 'Timestamp')
    local_forecast_ids = temp_df.ForecastId.unique()

    temp_df = produce_ar_features(temp_df)
    temp_df = produce_weather_future(temp_df)    
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])

    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day + weather_future_features            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour + weather_future_features
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min + weather_future_features

    train_ind = []
    val_ind = []

    for local_forcast_id in local_forecast_ids:
        local_temp_df = temp_df[temp_df.ForecastId == local_forcast_id]
        train_ind.extend(list(local_temp_df.index)[:-test_lengths[local_forcast_id]-test_lengths[local_forcast_id]//2])
        val_ind.extend(list(local_temp_df.index)[-test_lengths[local_forcast_id]-test_lengths[local_forcast_id]//2 :-test_lengths[local_forcast_id]])

    test_ind = list(temp_df[pd.isnull(temp_df.is_train)].index)              

    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors
def slice_train_data_rf_reverse_site_full(df_train,
                        df_sub,
                        site_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow','month','day']
    time_features_hour = ['dow','hour','month','day']
    time_features_15min = ['dow','hour','minute','month','day'] 

    temp_df = df_train[(df_train.SiteId == site_id)
                        &(pd.notnull(df_train.Value))]

    temp_df['is_train'] = 1

    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.SiteId == site_id)])
    # sort data to ensure consistency for producing weather time features 
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'] )         
    temp_df = temp_df.sort_values(by = 'Timestamp')
    local_forecast_ids = temp_df.ForecastId.unique()

    temp_df = produce_ar_features(temp_df)
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])

    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min 


    train_ind = []
    val_ind = []
    
    train_ind = list(temp_df[pd.notnull(temp_df.is_train)].index)         
    test_ind = list(temp_df[pd.isnull(temp_df.is_train)].index)              

    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors    
def slice_train_data_rf_reverse_site_weather_full(df_train,
                        df_sub,
                        site_id,
                        test_lengths):
    
    numeric_features = ['temp_diff','Temperature','is_day_off']
    time_features_day = ['dow','month','day']
    time_features_hour = ['dow','hour','month','day']
    time_features_15min = ['dow','hour','minute','month','day'] 
    weather_future_features = ['TF1','TF5','TF10','TF50']    

    temp_df = df_train[(df_train.SiteId == site_id)
                        &(pd.notnull(df_train.Value))]

    temp_df['is_train'] = 1

    # add test set data
    temp_df = temp_df.append(df_sub[(df_sub.SiteId == site_id)])
    # sort data to ensure consistency for producing weather time features 
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'] )         
    temp_df = temp_df.sort_values(by = 'Timestamp')
    local_forecast_ids = temp_df.ForecastId.unique()

    temp_df = produce_ar_features(temp_df)
    temp_df = produce_weather_future(temp_df)    
    temp_df = temp_df.reset_index()

    # produce variable sets
    # only non-zero dummy variables
    all_hols = temp_df[hol_list].sum()>0
    non_zero_hols = list(all_hols.index[all_hols == True])

    prediction_freq = temp_df.ForecastId_type.values[0]
    if  prediction_freq == '1_day':
        predictors = numeric_features + non_zero_hols + time_features_day + weather_future_features            
    elif prediction_freq == '1_hour':
        predictors = numeric_features + non_zero_hols + time_features_hour + weather_future_features
    else: # 15_mins
        predictors = numeric_features + non_zero_hols + time_features_15min + weather_future_features

    train_ind = []
    val_ind = []
    
    train_ind = list(temp_df[pd.notnull(temp_df.is_train)].index)         
    test_ind = list(temp_df[pd.isnull(temp_df.is_train)].index)            

    return temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors
def ETL_emb():
    df_train = (pd.read_csv('../data/forecast/train.csv')
                .pipe(parse_date,'Timestamp','ts','year','month','day','hour','minute','second','dow')
                .pipe(add_forecast_id_type)
                .pipe(date_to_int,'ts','ts_trend')
                )

    df_hol = (pd.read_csv('../data/forecast/holidays.csv')
            .drop('Unnamed: 0', axis=1)
            .drop_duplicates()          
            .pipe(parse_date,'Date','ts',
                            'year','month','day')
            .drop_duplicates()
            # .pipe(process_categorical_features,['Holiday']) sometimes there are 2 holidays per day ...
            .pipe(process_categorical_features,['Holiday'])
            .reset_index()
            )

    df_weather = (pd.read_csv('../data/forecast/weather.csv')
                  .drop('Unnamed: 0', axis=1)
                  .drop('Distance', axis=1)
                  .drop_duplicates()
                  .pipe(parse_date,'Timestamp','ts',
                        'year','month','day','hour')
                  .groupby(['year','month','day','hour','SiteId'])['Temperature'].mean()
                  .reset_index()
                 )

    df_meta = (pd.read_csv('../data/forecast/metadata.csv')
            .drop_duplicates()          
            )

    # add meta-data
    cols = ['SiteId','Surface','BaseTemperature','MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']
    join_keys = ['SiteId']
    df_train = df_train.merge(df_meta[cols], on=join_keys)
    # process days off
    df_train = process_days_off(df_train)

    # add weather by composite key
    cols = ['year','month','day','hour','Temperature','SiteId']
    join_keys = ['year','month','day','hour','SiteId']
    df_train = df_train.merge(df_weather[cols], on=join_keys, how='left')
    # fill missing values with -1
    df_train['Temperature'].fillna(-1, inplace=True)


    # join holidays by composite key
    cols = ['SiteId', 'year', 'month', 'day'] + ['Holiday']
    join_keys = ['year','month','day','SiteId']
    df_train = df_train.merge(df_hol[cols], on=join_keys, how='left')


    # ETL pipeline for test data
    df_sub = (pd.read_csv('../data/forecast/submission_format.csv')
                .pipe(parse_date,'Timestamp','ts','year','month','day','hour','minute','second','dow')
                .pipe(add_forecast_id_type)
                .pipe(date_to_int,'ts','ts_trend')
                )

    # add meta-data
    cols = ['SiteId','Surface','BaseTemperature','MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']
    join_keys = ['SiteId']
    df_sub = df_sub.merge(df_meta[cols], on=join_keys)
    # process days off
    df_sub = process_days_off(df_sub)

    # add weather by composite key
    cols = ['year','month','day','hour','Temperature','SiteId']
    join_keys = ['year','month','day','hour','SiteId']
    df_sub = df_sub.merge(df_weather[cols], on=join_keys, how='left')
    # fill missing values with -1
    df_sub['Temperature'].fillna(-1, inplace=True)

    # join holidays by composite key
    cols = ['SiteId', 'year', 'month', 'day'] + ['Holiday']
    join_keys = ['year','month','day','SiteId']
    df_sub = df_sub.merge(df_hol[cols], on=join_keys, how='left')

    del df_hol,df_weather,df_meta

    # fill all blank holidays with zeroes
    # for holiday in hol_list:
    #     df_train[holiday] = df_train[holiday].fillna(value=0)
    #     df_sub[holiday] = df_sub[holiday].fillna(value=0)

    df_train['temp_diff'] = df_train['BaseTemperature'] - df_train['Temperature']
    df_sub['temp_diff'] = df_sub['BaseTemperature'] - df_sub['Temperature']
    
    return df_train,df_sub
def preprocess_seq2seq(df_train,df_sub):
    # add train / test marker
    df_train['is_train'] = 1
    df_sub['is_train'] = 0

    # merge dataframes together
    data_df = df_train.append(df_sub)
    del df_train,df_sub
    
    # sort dataframe by site_id and timestamp 
    # so that we could easily use ar features 
    # and do indexing
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'] )
    data_df = data_df.sort_values(by = ['SiteId','Timestamp'])
    
    # replace non-existent temperature with base temperature
    data_df.loc[(data_df.Temperature == -1),'Temperature'] = data_df.loc[(data_df.Temperature == -1),'BaseTemperature']    
    
    
    # produce future features for temperature
    """
    print('Producing shifted future temperature features ...')
    with tqdm.tqdm(total=len(site_ids)) as pbar:
        for site_id in site_ids:
            data_df.loc[data_df.SiteId == site_id, 'FT1'] = data_df.loc[data_df.SiteId == site_id, 'Temperature'].shift(-1)
            data_df.loc[data_df.SiteId == site_id, 'FT5'] = data_df.loc[data_df.SiteId == site_id, 'Temperature'].shift(-5)
            data_df.loc[data_df.SiteId == site_id, 'FT10'] = data_df.loc[data_df.SiteId == site_id, 'Temperature'].shift(-10)
            data_df.loc[data_df.SiteId == site_id, 'FT50'] = data_df.loc[data_df.SiteId == site_id, 'Temperature'].shift(-50)

            data_df.loc[data_df.SiteId == site_id, 'FT1'] = data_df.loc[data_df.SiteId == site_id, 'FT1'].fillna(method='ffill')
            data_df.loc[data_df.SiteId == site_id, 'FT5'] = data_df.loc[data_df.SiteId == site_id, 'FT5'].fillna(method='ffill')
            data_df.loc[data_df.SiteId == site_id, 'FT10'] = data_df.loc[data_df.SiteId == site_id, 'FT10'].fillna(method='ffill')
            data_df.loc[data_df.SiteId == site_id, 'FT50'] = data_df.loc[data_df.SiteId == site_id, 'FT50'].fillna(method='ffill')

            pbar.update(1)
    """
    
    # remove very drastic outliers 
    print('Removing drastic outliers ... ')
    with tqdm.tqdm(total=len([1964, 1968, 1969, 4325, 5276, 5536])) as pbar:
        for forecast_id in [1964, 1968, 1969, 4325, 5276, 5536]:
            data_df.loc[(data_df.ForecastId == forecast_id)
                        &(data_df.Value>10**8), 'Value'] = np.nan

            data_df.loc[data_df.ForecastId == forecast_id, 'Value'] = data_df.loc[data_df.ForecastId == forecast_id, 'Value'].fillna(method='ffill')
            data_df.loc[data_df.ForecastId == forecast_id, 'Value'] = data_df.loc[data_df.ForecastId == forecast_id, 'Value'].fillna(method='bfill')

            pbar.update(1)
            
    return data_df
def interpolate(data_df):
    # interpolation pipeline
    # (1) find normal series
    # (2) find series with all NaNs => mark as submit zeroes
    # (3) use linear interpolation for series with < 0.2 NaNs
    # (4) series with > 0.5 NaNs => submit averages
    # (5) series with at least one test set worth of data immediately before test set => truncate train data
    # (6) submit averages for the remaining series
    # in the end we have
    # (1) normal or interpolated series
    # (2) submit zeroes
    # (3) submit averages


    data_df['is_null'] = 0
    data_df.loc[pd.isnull(data_df.Value),'is_null'] = 1

    # find normal time series
    null_df = data_df[data_df.is_train == 1].groupby(['ForecastId'])['is_null'].sum()
    count_df = data_df[data_df.is_train == 1].groupby(['ForecastId'])['is_null'].count()
    stat_df = pd.DataFrame()
    stat_df['null_count'] = null_df
    stat_df['count'] = count_df
    stat_df['percent_null'] = stat_df['null_count']/stat_df['count']
    # stat_df[stat_df.percent_null>0].percent_null.plot(kind='hist')
    normal_forecast_ids = list(stat_df[(stat_df.percent_null==0)].index.values)

    # all non-normal series
    # that have all the train dataset values equal to NAs
    submit_zeroes = list(stat_df[(stat_df.percent_null==1)].index)
    # that have the majority of values equal to NAs
    submit_averages = list(stat_df[(stat_df.percent_null>0.5)
                                   &(stat_df.percent_null<1)].index)
    # use linear interpolation for values between 0 and 0.2
    linear_interpolation = list(stat_df[(stat_df.percent_null>0)
                                   &(stat_df.percent_null<0.2)].index)

    data_df.loc[data_df.ForecastId.isin(linear_interpolation),'Value'] = data_df.loc[data_df.ForecastId.isin(linear_interpolation),'Value'].interpolate(method='linear')

    try_truncating = list(stat_df[(stat_df.percent_null>0.2)
                                   &(stat_df.percent_null<0.5)].index)

    use_last_window = []

    for forecast_id in try_truncating:
        test_len = data_df[(data_df.ForecastId==forecast_id)
                            &(data_df.is_train==0)].shape[0]

        last_window = data_df[(data_df.ForecastId==forecast_id)
                            &(data_df.is_train==1)].iloc[-test_len * 2 - 1:]

        non_blank_last_window = last_window[pd.notnull(last_window.Value)].shape[0] / last_window.shape[0]

        if (non_blank_last_window == 1):
            use_last_window.append(forecast_id)
        else:
            submit_averages.append(forecast_id)

    # drop non-last values for these forecast_ids
    for forecast_id in use_last_window:

        test_len = data_df[(data_df.ForecastId==forecast_id)
                            &(data_df.is_train==0)].shape[0]

        drop_index = data_df[data_df.ForecastId == forecast_id].index[:-test_len * 3 - 1]

        data_df.drop(index=drop_index,inplace=True,axis=0)         

    train_forecast_ids = normal_forecast_ids + linear_interpolation + use_last_window    
    
    return data_df,train_forecast_ids,normal_forecast_ids,linear_interpolation,use_last_window,submit_zeroes,submit_averages
