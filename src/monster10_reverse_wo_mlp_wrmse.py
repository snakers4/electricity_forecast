import tqdm
import time
import numpy as np
import pandas as pd
from math import sqrt

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import  metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  

# custom utils and random forest scripts
from Utils import parse_date,date_to_int,clean_ids,process_categorical_features,add_forecast_id_type,\
                  process_days_off,convert_to_dummies,ETL,count_test_period,xgb_fit_log,produce_ar_features,\
                  cb_fit_log,cb_fit_log_alg,lgb_fit_log_alg,produce_ar_step,cb_predict_test,lgb_predict_test,\
                  slice_train_data_rf,slice_train_data_rf_weather,sk_fit_alg,\
                  slice_train_data_rf_reverse,slice_train_data_rf_weather_reverse,\
                  slice_train_data_rf_full,slice_train_data_rf_weather_full,\
                  sk_fit_alg_wrmse,cb_fit_log_alg_wrmse,lgb_fit_log_alg_wrmse                    

# dummy variable list            
from holidayList import hol_list

# params for random forest baselines
from params import xgb_params,cbr_params,lgb_params,s_gbr_params,s_rfr_params

# windowed MLP baseline
from MLPPipeline import train,validate,predict_train,\
                        predict_test,slice_train_data_mlp,train_one_cnn,\
                        slice_train_data_mlp_weather

LOGNUMBER = '8_models_wrmse_train_val_add_time'

print('Starting the ETL process')

start_time = time.time()
df_train,df_sub = ETL()
test_lengths = count_test_period(df_sub)
elapsed_time = time.time() - start_time 

print('Time taken to complete the ETL process {}'.format(elapsed_time))

# use different set of time related features for each time span
target = 'Value'
numeric_features = ['temp_diff','Temperature','is_day_off']
time_features_day = ['dow']
time_features_hour = ['dow','hour']
time_features_15min = ['dow','hour','minute']

forecast_ids = list(test_lengths.keys())

print('LGB params : {}'.format(lgb_params))
print('CB  params : {}'.format(cbr_params))
print('SGBR params : {}'.format(s_gbr_params))
print('SRFR  params : {}'.format(s_rfr_params))

# suppress pandas warnings
# do not do this in production!
pd.set_option('mode.chained_assignment', None)

# store basic evaluation results
stat_cols = ['forecast_id',
             'cb_rmse_train','cb_r2_score_train','cb_rmse_val','cb_r2_score_val','cb_scores',
             'lgb_rmse_train','lgb_r2_score_train','lgb_rmse_val','lgb_r2_score_val','lgb_scores',
             'cb_w_rmse_train','cb_w_r2_score_train','cb_w_rmse_val','cb_w_r2_score_val','cb_w_scores',
             'lgb_w_rmse_train','lgb_w_r2_score_train','lgb_w_rmse_val','lgb_w_r2_score_val','lgb_w_scores',    
             
             's_gbr_rmse_train','s_gbr_r2_score_train','s_gbr_rmse_val','s_gbr_r2_score_val','s_gbr_scores',
             's_rfr_rmse_train','s_rfr_r2_score_train','s_rfr_rmse_val','s_rfr_r2_score_val','s_rfr_scores',
             's_gbr_w_rmse_train','s_gbr_w_r2_score_train','s_gbr_w_rmse_val','s_gbr_w_r2_score_val','s_gbr_w_scores',
             's_rfr_w_rmse_train','s_rfr_w_r2_score_train','s_rfr_w_rmse_val','s_rfr_w_r2_score_val','s_rfr_w_scores',

             'model_chosen'               
            ]
stat_df = pd.DataFrame(columns=stat_cols)

submission_df = pd.read_csv('../data/forecast/submission_format.csv')
submission_df = submission_df.set_index('obs_id')

submission_df_2 = pd.read_csv('../data/forecast/submission_format.csv')
submission_df_2 = submission_df_2.set_index('obs_id')

submission_df_3 = pd.read_csv('../data/forecast/submission_format.csv')
submission_df_3 = submission_df_3.set_index('obs_id')

with tqdm.tqdm(total=len(forecast_ids)) as pbar:
    for i,forecast_id in enumerate(forecast_ids):
        try:
            #============ RF models ============#
            cbr = CatBoostRegressor(**cbr_params)
            lgb = LGBMRegressor(**lgb_params)
            s_gbr = GradientBoostingRegressor(**s_gbr_params)
            s_rfr = RandomForestRegressor(**s_rfr_params)
            
            # first fit all models on reverse validation (train - val) to collect the metrics
            temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors = slice_train_data_rf_reverse(df_train,
                                                                                                         df_sub,
                                                                                                         forecast_id,
                                                                                                         test_lengths)
            
            # then fit the model on the whole dataset and use the best one
            _,_,train_ind_full,_,_,_,_ = slice_train_data_rf_full(df_train,
                                                             df_sub,
                                                             forecast_id,
                                                             test_lengths)
            
            start_time = time.time()

            #============ Fit CatBoost ============#
            try:
                # first collect the metrics
                cb_rmse_train,cb_r2_score_train,cb_rmse_val,cb_r2_score_val,cb_scores,cbr = cb_fit_log_alg_wrmse(cbr,
                         temp_df.filter(items=train_ind,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                # then fit the model on the whole dataset
                _,_,_,_,_,cbr = cb_fit_log_alg_wrmse(cbr,
                         temp_df.filter(items=train_ind_full,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_cb'] = cbr.predict(temp_df[predictors])
                cb_preds = list(temp_df.loc[test_ind].preds_cb)            
            except Exception as e:
                # in case of exception - record poor scores
                print('CatBoost fitting failed due to error : {}'.format(print(str(e))))

                cb_rmse_train = 1e9
                cb_r2_score_train = -1
                cb_rmse_val = 1e9
                cb_r2_score_val = -1
                cb_scores = []
                cb_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))   

            #============ Fit LightGBM ============#                
            try:
                # first collect the metrics
                lgb_rmse_train,lgb_r2_score_train,lgb_rmse_val,lgb_r2_score_val,lgb_scores,lgb = lgb_fit_log_alg_wrmse(lgb,
                         temp_df.filter(items=train_ind,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                # then fit the model on the whole dataset
                _,_,_,_,_,lgb = lgb_fit_log_alg(lgb,
                         temp_df.filter(items=train_ind_full,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)                
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_lgb'] = lgb.predict(temp_df[predictors])
                lgb_preds = list(temp_df.loc[test_ind].preds_lgb)            
            except Exception as e:
                # in case of exception - record poor scores
                print('LightGBM fitting failed due to error : {}'.format(print(str(e))))
                lgb_rmse_train = 1e9
                lgb_r2_score_train = -1
                lgb_rmse_val = 1e9
                lgb_r2_score_val = -1
                lgb_scores = []
                lgb_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))  

            #============ Fit Sklearn Boosted Trees ============#
            try:
                # first collect the metrics
                s_gbr_rmse_train,s_gbr_r2_score_train,s_gbr_rmse_val,s_gbr_r2_score_val,s_gbr_scores,s_gbr = sk_fit_alg_wrmse(s_gbr,
                                                                                                             temp_df.filter(items=train_ind,axis=0),
                                                                                                             temp_df.filter(items=val_ind,axis=0),                
                                                                                                             predictors)
                # then fit the model on the whole dataset
                _,_,_,_,_,s_gbr = sk_fit_alg_wrmse(s_gbr,
                                             temp_df.filter(items=train_ind_full,axis=0),
                                             temp_df.filter(items=val_ind,axis=0),                
                                             predictors)                
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_s_gbr'] = s_gbr.predict(temp_df[predictors])
                s_gbr_preds = list(temp_df.loc[test_ind].preds_s_gbr)
            except Exception as e:
                # in case of exception - record poor scores
                print('Sklearn boosted trees fitting failed due to error : {}'.format(print(str(e))))

                s_gbr_rmse_train = 1e9
                s_gbr_r2_score_train = -1
                s_gbr_rmse_val = 1e9
                s_gbr_r2_score_val = -1
                s_gbr_scores = []
                s_gbr_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))   
                
            #============ Fit Sklearn Random Forest ============#
            try:
                # first collect the metrics
                s_rfr_rmse_train,s_rfr_r2_score_train,s_rfr_rmse_val,s_rfr_r2_score_val,s_rfr_scores,s_rfr = sk_fit_alg_wrmse(s_rfr,
                                                                                                             temp_df.filter(items=train_ind,axis=0),
                                                                                                             temp_df.filter(items=val_ind,axis=0),                
                                                                                                             predictors)
                # then fit the model on the whole dataset
                _,_,_,_,_,s_rfr = sk_fit_alg_wrmse(s_rfr,
                                             temp_df.filter(items=train_ind_full,axis=0),
                                             temp_df.filter(items=val_ind,axis=0),                
                                             predictors)
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_s_rfr'] = s_rfr.predict(temp_df[predictors])
                s_rfr_preds = list(temp_df.loc[test_ind].preds_s_rfr)  
            except Exception as e:
                # in case of exception - record poor scores
                print('Sklearn Random forest fitting failed due to error : {}'.format(print(str(e))))

                s_rfr_rmse_train = 1e9
                s_rfr_r2_score_train = -1
                s_rfr_rmse_val = 1e9
                s_rfr_r2_score_val = -1
                s_rfr_scores = []
                s_rfr_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))                   
                   
            #============ RF models + future weather ============#

            cbr = CatBoostRegressor(**cbr_params)
            lgb = LGBMRegressor(**lgb_params)
            s_gbr = GradientBoostingRegressor(**s_gbr_params)
            s_rfr = RandomForestRegressor(**s_rfr_params)            

            temp_df,all_hols,train_ind,val_ind,test_ind,prediction_freq,predictors = slice_train_data_rf_weather_reverse(df_train,
                                                                                                                 df_sub,
                                                                                                                 forecast_id,
                                                                                                                 test_lengths)
            
            _,_,train_ind_full,_,_,_,_ = slice_train_data_rf_weather_full(df_train,
                                                                 df_sub,
                                                                 forecast_id,
                                                                 test_lengths)            
            
            #============ Fit CatBoost + weather ============#
            try:
                # first collect the metrics
                cb_w_rmse_train,cb_w_r2_score_train,cb_w_rmse_val,cb_w_r2_score_val,cb_w_scores,cbr = cb_fit_log_alg_wrmse(cbr,
                         temp_df.filter(items=train_ind,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                # then fit the model on the whole dataset    
                _,_,_,_,_,cbr = cb_fit_log_alg(cbr,
                         temp_df.filter(items=train_ind_full,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_cb_w'] = cbr.predict(temp_df[predictors])
                cb_w_preds = list(temp_df.loc[test_ind].preds_cb_w)            
            except Exception as e:
                # in case of exception - record poor scores
                print('CatBoost with weather fitting failed due to error : {}'.format(print(str(e))))

                cb_w_rmse_train = 1e9
                cb_w_r2_score_train = -1
                cb_w_rmse_val = 1e9
                cb_w_r2_score_val = -1
                cb_w_scores = []
                cb_w_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))

            #============ Fit LightGBM + weather ============#                
            try:
                # first collect the metrics
                lgb_w_rmse_train,lgb_w_r2_score_train,lgb_w_rmse_val,lgb_w_r2_score_val,lgb_w_scores,lgb = lgb_fit_log_alg_wrmse(lgb,
                         temp_df.filter(items=train_ind,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)
                # then fit the model on the whole dataset  
                _,_,_,_,_,lgb = lgb_fit_log_alg(lgb,
                         temp_df.filter(items=train_ind_full,axis=0),
                         temp_df.filter(items=val_ind,axis=0),                
                         predictors)                
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_lgb_w'] = lgb.predict(temp_df[predictors])
                lgb_w_preds = list(temp_df.loc[test_ind].preds_lgb_w)
            except Exception as e:
                # in case of exception - record poor scores
                print('LightGBM with weather fitting failed due to error : {}'.format(print(str(e))))
                lgb_w_rmse_train = 1e9
                lgb_w_r2_score_train = -1
                lgb_w_rmse_val = 1e9
                lgb_w_r2_score_val = -1
                lgb_w_scores = []
                lgb_w_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))            

            #============ Fit Sklearn Boosted Trees ============#
            try:
                # first collect the metrics
                s_gbr_w_rmse_train,s_gbr_w_r2_score_train,s_gbr_w_rmse_val,s_gbr_w_r2_score_val,s_gbr_w_scores,s_gbr = sk_fit_alg_wrmse(s_gbr,
                                                                                                             temp_df.filter(items=train_ind,axis=0),
                                                                                                             temp_df.filter(items=val_ind,axis=0),                
                                                                                                             predictors)
                # then fit the model on the whole dataset 
                _,_,_,_,_,s_gbr = sk_fit_alg_wrmse(s_gbr,
                                             temp_df.filter(items=train_ind_full,axis=0),
                                             temp_df.filter(items=val_ind,axis=0),                
                                             predictors)                
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_s_gbr_w'] = s_gbr.predict(temp_df[predictors])
                s_gbr_w_preds = list(temp_df.loc[test_ind].preds_s_gbr_w)
            except Exception as e:
                # in case of exception - record poor scores
                print('Sklearn boosted trees fitting failed due to error : {}'.format(print(str(e))))

                s_gbr_w_rmse_train = 1e9
                s_gbr_w_r2_score_train = -1
                s_gbr_w_rmse_val = 1e9
                s_gbr_w_r2_score_val = -1
                s_gbr_w_scores = []
                s_gbr_w_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))   
                
            #============ Fit Sklearn Random Forest ============#
            try:
                # first collect the metrics
                s_rfr_w_rmse_train,s_rfr_w_r2_score_train,s_rfr_w_rmse_val,s_rfr_w_r2_score_val,s_rfr_w_scores,s_rfr = sk_fit_alg_wrmse(s_rfr,
                                                                                                             temp_df.filter(items=train_ind,axis=0),
                                                                                                             temp_df.filter(items=val_ind,axis=0),                
                                                                                                             predictors)
                # then fit the model on the whole dataset 
                _,_,_,_,_,s_rfr = sk_fit_alg_wrmse(s_rfr,
                                             temp_df.filter(items=train_ind_full,axis=0),
                                             temp_df.filter(items=val_ind,axis=0),                
                                             predictors)
                
                obs_ids = list(temp_df.loc[test_ind].obs_id)
                temp_df['preds_s_rfr_w'] = s_rfr.predict(temp_df[predictors])
                s_rfr_w_preds = list(temp_df.loc[test_ind].preds_s_rfr_w)  
            except Exception as e:
                # in case of exception - record poor scores
                print('Sklearn Random forest fitting failed due to error : {}'.format(print(str(e))))

                s_rfr_w_rmse_train = 1e9
                s_rfr_w_r2_score_train = -1
                s_rfr_w_rmse_val = 1e9
                s_rfr_w_r2_score_val = -1
                s_rfr_w_scores = []
                s_rfr_w_preds  = [0] * len(list(temp_df.loc[test_ind].obs_id))                                   
               

            #============ Record scores ============#  
            elapsed_time = time.time() - start_time 

            stat_df_temp = pd.DataFrame(columns=stat_cols)
            stat_df_temp['forecast_id'] = [forecast_id]

            stat_df_temp['cb_rmse_train'] = [cb_rmse_train]
            stat_df_temp['cb_r2_score_train'] = [cb_r2_score_train]
            stat_df_temp['cb_rmse_val'] = [cb_rmse_val]
            stat_df_temp['cb_r2_score_val'] = [cb_r2_score_val]
            stat_df_temp['cb_scores'] = [cb_scores]

            stat_df_temp['lgb_rmse_train'] = [lgb_rmse_train]
            stat_df_temp['lgb_r2_score_train'] = [lgb_r2_score_train]
            stat_df_temp['lgb_rmse_val'] = [lgb_rmse_val]
            stat_df_temp['lgb_r2_score_val'] = [lgb_r2_score_val]
            stat_df_temp['lgb_scores'] = [lgb_scores]

            stat_df_temp['cb_w_rmse_train'] = [cb_w_rmse_train]
            stat_df_temp['cb_w_r2_score_train'] = [cb_w_r2_score_train]
            stat_df_temp['cb_w_rmse_val'] = [cb_w_rmse_val]
            stat_df_temp['cb_w_r2_score_val'] = [cb_w_r2_score_val]
            stat_df_temp['cb_w_scores'] = [cb_w_scores]

            stat_df_temp['lgb_w_rmse_train'] = [lgb_w_rmse_train]
            stat_df_temp['lgb_w_r2_score_train'] = [lgb_w_r2_score_train]
            stat_df_temp['lgb_w_rmse_val'] = [lgb_w_rmse_val]
            stat_df_temp['lgb_w_r2_score_val'] = [lgb_w_r2_score_val]
            stat_df_temp['lgb_w_scores'] = [lgb_w_scores]
            
           
            stat_df_temp['s_gbr_rmse_train'] = [s_gbr_rmse_train]
            stat_df_temp['s_gbr_r2_score_train'] = [s_gbr_r2_score_train]
            stat_df_temp['s_gbr_rmse_val'] = [s_gbr_rmse_val]
            stat_df_temp['s_gbr_r2_score_val'] = [s_gbr_r2_score_val]
            stat_df_temp['s_gbr_scores'] = [s_gbr_scores]

            stat_df_temp['s_rfr_rmse_train'] = [s_rfr_rmse_train]
            stat_df_temp['s_rfr_r2_score_train'] = [s_rfr_r2_score_train]
            stat_df_temp['s_rfr_rmse_val'] = [s_rfr_rmse_val]
            stat_df_temp['s_rfr_r2_score_val'] = [s_rfr_r2_score_val]
            stat_df_temp['s_rfr_scores'] = [s_rfr_scores]

            stat_df_temp['s_gbr_w_rmse_train'] = [s_gbr_w_rmse_train]
            stat_df_temp['s_gbr_w_r2_score_train'] = [s_gbr_w_r2_score_train]
            stat_df_temp['s_gbr_w_rmse_val'] = [s_gbr_w_rmse_val]
            stat_df_temp['s_gbr_w_r2_score_val'] = [s_gbr_w_r2_score_val]
            stat_df_temp['s_gbr_w_scores'] = [s_gbr_w_scores]
            
            stat_df_temp['s_rfr_w_rmse_train'] = [s_rfr_w_rmse_train]
            stat_df_temp['s_rfr_w_r2_score_train'] = [s_rfr_w_r2_score_train]
            stat_df_temp['s_rfr_w_rmse_val'] = [s_rfr_w_rmse_val]
            stat_df_temp['s_rfr_w_r2_score_val'] = [s_rfr_w_r2_score_val]
            stat_df_temp['s_rfr_w_scores'] = [s_rfr_w_scores] 
           
            best_model_idx = np.asarray([cb_rmse_val,lgb_rmse_val,cb_w_rmse_val,lgb_w_rmse_val,s_gbr_rmse_val,s_rfr_rmse_val,s_gbr_w_rmse_val,s_rfr_w_rmse_val]).argmin() 
            if best_model_idx == 0:
                stat_df_temp['model_chosen'] = ['cb']  
            elif best_model_idx == 1:
                stat_df_temp['model_chosen'] = ['lgb'] 
            elif best_model_idx == 2:
                stat_df_temp['model_chosen'] = ['cb_w']
            elif best_model_idx == 3:
                stat_df_temp['model_chosen'] = ['lgb_w']
            elif best_model_idx == 4:
                stat_df_temp['model_chosen'] = ['s_gbr']
            elif best_model_idx == 5:
                stat_df_temp['model_chosen'] = ['s_rfr']
            elif best_model_idx == 6:
                stat_df_temp['model_chosen'] = ['s_gbr_w']
            elif best_model_idx == 7:
                stat_df_temp['model_chosen'] = ['s_rfr_w']             
            

            #============ Finish recording scores ============#  
            # record scores even if they are bad
            stat_df_temp['time_taken'] = [elapsed_time]              
            stat_df = stat_df.append(stat_df_temp)     

            # log every 500 iterations
            if (i+1)%500 == 0:
                stat_df.to_csv('forest_stats_{}.csv'.format(LOGNUMBER))
                submission_df.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_always_best_model'))
                submission_df_2.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_last_if_bad'))
                submission_df_3.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_average_or_last'))
                print('Iteration {}, logs saved'.format(i))
            pbar.update(1)            

            #============ Choose the best model ============# 

            best_model_r2 = np.asarray([cb_r2_score_val,lgb_r2_score_val,cb_w_r2_score_val,lgb_w_r2_score_val,s_gbr_r2_score_val,s_rfr_r2_score_val,s_gbr_w_r2_score_val,s_rfr_w_r2_score_val]).max()
            
            if sum([cb_r2_score_val,lgb_r2_score_val,cb_w_r2_score_val,lgb_w_r2_score_val,s_gbr_r2_score_val,s_rfr_r2_score_val,s_gbr_w_r2_score_val,s_rfr_w_r2_score_val]) == -8:
                # all the models failed / produce crap
                raise ValueError('All the models failed - submit last value to both dataframes')
            else:
                # save best result in the first dataframe anyway
                if best_model_idx == 0:
                    submission_df.loc[obs_ids,'Value'] = cb_preds
                elif best_model_idx == 1:
                    submission_df.loc[obs_ids,'Value'] = lgb_preds
                elif best_model_idx == 2:
                    submission_df.loc[obs_ids,'Value'] = cb_w_preds
                elif best_model_idx == 3:
                    submission_df.loc[obs_ids,'Value'] = lgb_w_preds

                elif best_model_idx == 4:
                    submission_df.loc[obs_ids,'Value'] = s_gbr_preds
                elif best_model_idx == 5:
                    submission_df.loc[obs_ids,'Value'] = s_rfr_preds
                elif best_model_idx == 6:
                    submission_df.loc[obs_ids,'Value'] = s_gbr_w_preds
                elif best_model_idx == 7:
                    submission_df.loc[obs_ids,'Value'] = s_rfr_w_preds

                # if no model failed
                if -1 not in [cb_r2_score_val,lgb_r2_score_val,cb_w_r2_score_val,lgb_w_r2_score_val,s_gbr_r2_score_val,s_rfr_r2_score_val,s_gbr_w_r2_score_val,s_rfr_w_r2_score_val]:
                    # save averages to the third dataframe                
                    arr = np.vstack((cb_preds,lgb_preds,cb_w_preds,lgb_w_preds,s_gbr_preds,s_rfr_preds,s_gbr_w_preds,s_rfr_w_preds))
                    average_preds = list(arr.mean(axis=0))
                    submission_df_3.loc[obs_ids,'Value'] = average_preds
                else:
                    # if any of the models failed - just submit the best model instead of average
                    if best_model_idx == 0:
                        submission_df_3.loc[obs_ids,'Value'] = cb_preds
                    elif best_model_idx == 1:
                        submission_df_3.loc[obs_ids,'Value'] = lgb_preds
                    elif best_model_idx == 2:
                        submission_df_3.loc[obs_ids,'Value'] = cb_w_preds
                    elif best_model_idx == 3:
                        submission_df_3.loc[obs_ids,'Value'] = lgb_w_preds

                    elif best_model_idx == 4:
                        submission_df_3.loc[obs_ids,'Value'] = s_gbr_preds
                    elif best_model_idx == 5:
                        submission_df_3.loc[obs_ids,'Value'] = s_rfr_preds
                    elif best_model_idx == 6:
                        submission_df_3.loc[obs_ids,'Value'] = s_gbr_w_preds
                    elif best_model_idx == 7:
                        submission_df_3.loc[obs_ids,'Value'] = s_rfr_w_preds

                    
                # if there is no proper model, submit the last ts value to the second dataframe
                # otherwise save proper values to the second dataframe
                if best_model_r2 < 0:
                    try:
                        train_mean = temp_df.Value.iloc[-1]
                    except:
                        print('Bad train data sample, submitting 0')
                        train_mean = 0
                    submission_df_2.loc[obs_ids,'Value'] = [train_mean] * len(obs_ids)
                    
                    with open('last_value_submits_{}.csv'.format(LOGNUMBER), 'a') as the_file:
                        the_file.write(str(forecast_id)+'\n')
                
                else:
                    # save the best model to the second dataframe
                    if best_model_idx == 0:
                        submission_df_2.loc[obs_ids,'Value'] = cb_preds
                    elif best_model_idx == 1:
                        submission_df_2.loc[obs_ids,'Value'] = lgb_preds
                    elif best_model_idx == 2:
                        submission_df_2.loc[obs_ids,'Value'] = cb_w_preds
                    elif best_model_idx == 3:
                        submission_df_2.loc[obs_ids,'Value'] = lgb_w_preds

                    elif best_model_idx == 4:
                        submission_df_2.loc[obs_ids,'Value'] = s_gbr_preds
                    elif best_model_idx == 5:
                        submission_df_2.loc[obs_ids,'Value'] = s_rfr_preds
                    elif best_model_idx == 6:
                        submission_df_2.loc[obs_ids,'Value'] = s_gbr_w_preds
                    elif best_model_idx == 7:
                        submission_df_2.loc[obs_ids,'Value'] = s_rfr_w_preds
        
              
        except Exception as e:
            
            with open('last_value_submits_{}.csv'.format(LOGNUMBER), 'a') as the_file:
                the_file.write(str(forecast_id)+'\n')
                        
                        
            # if model fitting and prediction fails, then just submit average values            
            print('Error {}. Forecast_id {}, submitting mean'.format(str(e),forecast_id))
            
            temp_df = df_train[(df_train.ForecastId == forecast_id)
                                &(pd.notnull(df_train.Value))]
            
            try:
                train_mean = temp_df.Value.iloc[-1]
            except:
                print('Bad train data sample, submitting 0')
                train_mean = 0
             
            temp_df = temp_df.append(df_sub[(df_sub.ForecastId == forecast_id)])            
            temp_df = temp_df.reset_index()
            
            temp_df['pred'] = train_mean
            
            test_ind = list(temp_df.index)[-test_lengths[forecast_id]:]            

            # set the predictions in the prediction df
            obs_ids = list(temp_df.loc[test_ind].obs_id)
            predictions = list(temp_df.loc[test_ind].pred)
            
            # save last value or 0 to both dataframes
            submission_df.loc[obs_ids,'Value'] = predictions
            submission_df_2.loc[obs_ids,'Value'] = predictions
            submission_df_3.loc[obs_ids,'Value'] = predictions

stat_df.to_csv('forest_stats_{}.csv'.format(LOGNUMBER))
submission_df['Value'] = submission_df['Value'].fillna(value=0)
submission_df_2['Value'] = submission_df_2['Value'].fillna(value=0)
submission_df_3['Value'] = submission_df_3['Value'].fillna(value=0)
submission_df.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_always_best_model'))
submission_df_2.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_last_if_bad'))
submission_df_3.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER+'_average_or_last'))   