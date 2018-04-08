xgb_params = {
     'learning_rate':0.1,
     'n_estimators':100,
     'max_depth':5,
     'min_child_weight':1,
     'booster':'gbtree',
     'gamma':0,
     'subsample':0.8,
     'colsample_bytree':0.8,
     'objective':'reg:linear',
     'n_jobs':6,
     'scale_pos_weight':1,
     'silent':True,
     'seed':27    
}
cbr_params = {
    'random_state':27, 
    'has_time':True, 
    'learning_rate':0.1, 
    'l2_leaf_reg':500, 
    'n_estimators':700, 
    'loss_function':'RMSE', 
    'approx_on_full_history':True,
    'thread_count':10,
    'verbose':False
}
lgb_params = {
    'boosting_type':'gbdt',
    'random_state':27, 
    'learning_rate':0.1, 
    'num_leaves':512,
    'n_estimators':700, 
    'objective':'regression', 
    'n_jobs':10,
    "verbose": -1 
}
mlp_params = {
    'activation':'relu',
    'alpha':1e-05,
    'batch_size':'auto',
    'early_stopping':False,
    'hidden_layer_sizes':(256, 256),
    'learning_rate':'constant',
    'learning_rate_init':0.001,
    'max_iter':200,
    'random_state':1,
    'shuffle':True,
    'solver':'adam',
    'validation_fraction':0.1,
    'verbose':False
    }
s_gbr_params = {
    'learning_rate' : 0.1,
    'n_estimators' : 100,
    'max_depth' : 7,
    'criterion':'friedman_mse',
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'subsample' : 0.8,
    'verbose' : 0,    
}
s_rfr_params = {
    'n_estimators' : 50,
    'criterion':'mse',
    'max_depth' : 100,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'verbose' : 0,    
}