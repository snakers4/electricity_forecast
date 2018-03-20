# Custom tensorboard logging
from TbLogger import Logger

import os
import time
import tqdm
import shutil
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import  metrics
 
# custom utils and random forest scripts
from Utils import ETL_emb, count_test_period,preprocess_seq2seq,interpolate

# torch imports
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

# custom classes
from SDataset import S2SDataset
from ELstm import E2ELSTM,WMSELoss,E2ELSTM_day,EncoderDecoderGRU,E2EGRU

import pandas as pd
import numpy as np
from math import sqrt

from Utils import WRMSE

import pickle

LOGNUMBER = 'encoder_decoder_grus_ar_features2'

def preprocess_data():
    print('Starting the ETL process')

    start_time = time.time()
    df_train,df_sub = ETL_emb()
    test_lengths = count_test_period(df_sub)
    elapsed_time = time.time() - start_time 

    print('Time taken to complete the ETL process {}'.format(elapsed_time))

    forecast_ids = list(test_lengths.keys())
    site_ids = list(df_sub.SiteId.unique())

    # suppress pandas warnings
    # do not do this in production!
    pd.set_option('mode.chained_assignment', None)

    data_df = preprocess_seq2seq(df_train,df_sub)
    # reset index to make sure we do not have double indexes and for easier indexing
    data_df = data_df.reset_index()
    del data_df['index']

    # leave only the first holiday
    data_df = data_df[(data_df['obs_id'].shift(+1) != data_df['obs_id'])]
    data_df = data_df.reset_index()
    del data_df['index']
    # fill days wo holidays with -1
    data_df['Holiday'] = data_df['Holiday'].fillna(value=-1)

    data_df,train_forecast_ids,normal_forecast_ids,linear_interpolation,last_window,submit_zeroes,submit_averages = interpolate(data_df)
    
    return data_df,train_forecast_ids,normal_forecast_ids,linear_interpolation,last_window,submit_zeroes,submit_averages

def main():
    
    # suppress pandas warnings
    # do not do this in production!
    pd.set_option('mode.chained_assignment', None)
    
    # read all pre-calculated objects
    data_df = pd.read_feather('../data/forecast/data_df_feather_ar_values')
    with open('train_forecast_ids.pkl', 'rb') as input:
        train_forecast_ids = pickle.load(input)
    with open('normal_forecast_ids.pkl', 'rb') as input:
        normal_forecast_ids = pickle.load(input)
    with open('linear_interpolation.pkl', 'rb') as input:
        linear_interpolation = pickle.load(input)
    with open('use_last_window.pkl', 'rb') as input:
        use_last_window = pickle.load(input)
    with open('submit_zeroes.pkl', 'rb') as input:
        submit_zeroes = pickle.load(input)
    with open('submit_averages.pkl', 'rb') as input:
        submit_averages = pickle.load(input)    

    # override - exclude last window series
    train_forecast_ids = normal_forecast_ids + linear_interpolation
    
    # take last window from previous submit
    
    best_1d_model = 'weights/bl_1d_1e2_best.pth.tar'
    best_1h_model = 'weights/15m1d_encoder_decoder_w192_hid192_3lyr_5ar_best.pth.tar'
    best_15m_model = 'weights/15m1d_encoder_decoder_w192_hid192_3lyr_5ar_best.pth.tar'
    
    model_1d = E2ELSTM_day(in_sequence_len = 30,
                     out_sequence_len = 30,
                     features_meta_total = 43,
                     features_ar_total = 1,
                     meta_hidden_layer_length = 30,
                     ar_hidden_layer_length = 30,
                     meta_hidden_layers = 2,
                     ar_hidden_layers = 2,
                     lstm_dropout = 0,
                     classifier_hidden_length = 256)
    
    model_1h = EncoderDecoderGRU(in_sequence_len = 192,
                     out_sequence_len = 192,
                     features_meta_total = 76,
                     features_ar_total = 1,
                     meta_hidden_layer_length = 192,
                     ar_hidden_layer_length = 192,
                     meta_hidden_layers = 3,
                     ar_hidden_layers = 3,
                     lstm_dropout = 0,
                     classifier_hidden_length = 512)
    
    model_15m = EncoderDecoderGRU(in_sequence_len = 192,
                     out_sequence_len = 192,
                     features_meta_total = 76,
                     features_ar_total = 1,
                     meta_hidden_layer_length = 192,
                     ar_hidden_layer_length = 192,
                     meta_hidden_layers = 3,
                     ar_hidden_layers = 3,
                     lstm_dropout = 0,
                     classifier_hidden_length = 512)

    model_1h = torch.nn.DataParallel(model_1h)
    model_15m = torch.nn.DataParallel(model_15m)
    
    checkpoint = torch.load(best_1d_model)
    model_1d.load_state_dict(checkpoint['state_dict'])
    print("model_1d => loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    checkpoint = torch.load(best_1h_model)
    model_1h.load_state_dict(checkpoint['state_dict'])
    print("model_1h => loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    
    checkpoint = torch.load(best_15m_model)
    model_15m.load_state_dict(checkpoint['state_dict'])
    print("model_15m => loaded checkpoint (epoch {})".format(checkpoint['epoch']))    
    
    submission_df = pd.read_csv('../data/forecast/submission_format.csv')
    submission_df = submission_df.set_index('obs_id')
    
    model_1d.cuda()
    model_1h.cuda()
    model_15m.cuda()
    
    model_1d.eval()
    model_1h.eval()
    model_15m.eval()    

    stat_cols = ['forecast_id','wrmse_val','r2_val']
    stat_df = pd.DataFrame(columns = stat_cols)

    # select only series we marked as trainable
    # negation is for speed only
    trainable_df = data_df[(~data_df.ForecastId.isin(list(set(data_df.ForecastId.unique()) - set(train_forecast_ids))))]    
    
    print('Predicting for 1 day series ...')
    
    temp_features = ['Temperature']
    hol_emb_features = ['Holiday']
    time_emb_features = ['month','day','dow']
    target = ['Value']
    predictors = temp_features + hol_emb_features + time_emb_features
    
    predict_dataset = S2SDataset(df = trainable_df,
                         series_type = '1_day',
                         in_sequence_len = 30,
                         out_sequence_len = 30,
                         target = 'Value',
                         mode = 'test',
                         split_mode = 'random',
                         predictors = predictors)

    predict_dataset_wrmse = S2SDataset(df = trainable_df,
                         series_type = '1_day',
                         in_sequence_len = 30,
                         out_sequence_len = 30,
                         target = 'Value',
                         mode = 'evaluate_wrmse',
                         split_mode = 'random',
                         predictors = predictors)    
    
    print('Dataset length is {}'.format(len(predict_dataset.forecast_ids)))
    
    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar:    
        for i,forecast_id in enumerate(predict_dataset.forecast_ids):
            i = predict_dataset.forecast_ids.index(forecast_id)

            test_X_sequences_meta,test_X_sequences_ar,len_diff = predict_dataset.__getitem__(i)

            # into PyTorch format
            test_X_sequences_meta = torch.from_numpy(test_X_sequences_meta).view(1,-1,5)
            test_X_sequences_ar = torch.from_numpy(test_X_sequences_ar).view(1,-1,1)    

            # transform data from Batch x Window x Etc into Batch x Etc format
            test_X_sequences_ar = test_X_sequences_ar.float()
            test_X_sequences_temp = test_X_sequences_meta[:,:,0:1].float()
            test_X_sequences_meta = test_X_sequences_meta[:,:,1:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_1d(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()
            # predict first 30 time points
            output1 = output

            # then predict the remaining points using data we have
            predict_len = predict_dataset.df[(predict_dataset.df.ForecastId == forecast_id)
                               & (predict_dataset.df.is_train == 0)].shape[0]
            remaining_len = predict_len - len(output1)

            # use our preds as AR values for final prediction
            # predict more values
            test_X_sequences_ar = output1
            test_X_sequences_meta = predict_dataset.df[(predict_dataset.df.ForecastId == forecast_id)
                                                       &(predict_dataset.df.is_train == 0)].iloc[-len(output1) * 2:][predictors].values
            test_X_sequences_meta = test_X_sequences_meta.copy()
            # into PyTorch format
            test_X_sequences_meta = torch.from_numpy(test_X_sequences_meta).view(1,-1,5)
            test_X_sequences_ar = torch.from_numpy(test_X_sequences_ar).view(1,-1,1)    

            # transform data from Batch x Window x Etc into Batch x Etc format
            test_X_sequences_ar = test_X_sequences_ar.float()
            test_X_sequences_temp = test_X_sequences_meta[:,:,0:1].float()
            test_X_sequences_meta = test_X_sequences_meta[:,:,1:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)    

            # compute output
            output = model_1d(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()
            # predict first 30 time points
            output2 = output    

            truncate_len = predict_len - len(output1) - len(output2)
            final_output = np.hstack((output1,output2[-truncate_len:]))

            final_output = final_output * predict_dataset.std_dict[forecast_id] + predict_dataset.mean_dict[forecast_id]
            submission_df.loc[submission_df.ForecastId == forecast_id, 'Value'] =  final_output    

            pbar.update(1)

    # forecast loop - evaluate on the last sequence on trainval dataset
    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar: 
        for i,forecast_id in enumerate(predict_dataset_wrmse.forecast_ids):
            i = predict_dataset_wrmse.forecast_ids.index(forecast_id)

            X_sequences_ar,X_sequences_meta,y_sequences = predict_dataset_wrmse.__getitem__(i)
            X_sequences_meta = torch.from_numpy(X_sequences_meta).view(1,-1,5)
            X_sequences_ar = torch.from_numpy(X_sequences_ar).view(1,-1,1)

            y_true = y_sequences.reshape(-1) * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            # transform data from Batch x Window x Etc into Batch x Etc format
            X_sequences_ar = X_sequences_ar.float()
            X_sequences_temp = X_sequences_meta[:,:,0:1].float()
            X_sequences_meta = X_sequences_meta[:,:,1:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_1d(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()

            output = output * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            wrmse_val = WRMSE(y_true, output)
            r2_score_val = metrics.r2_score(y_true, output)
            stat_df = stat_df.append(pd.DataFrame([dict(zip(stat_cols,[forecast_id,r2_score_val,wrmse_val]))]))

            pbar.update(1)    

    print('Predicting for 1 hour series ...')   
    
    temp_features = ['Temperature']
    ar_features = ['Value1','Value4','Value12','Value96']
    hol_emb_features = ['Holiday']
    time_emb_features = ['year', 'month', 'day', 'hour', 'minute','dow']
    target = ['Value']
    predictors = temp_features + ar_features + hol_emb_features + time_emb_features

    predict_dataset = S2SDataset(df = trainable_df,
                         series_type = '1_hour',
                         in_sequence_len = 192,
                         out_sequence_len = 192,
                         target = 'Value',
                         mode = 'test',
                         split_mode = 'random',
                         predictors = predictors)

    predict_dataset_wrmse = S2SDataset(df = trainable_df,
                         series_type = '1_hour',
                         in_sequence_len = 192,
                         out_sequence_len = 192,
                         target = 'Value',
                         mode = 'evaluate_wrmse',
                         split_mode = 'random',
                         predictors = predictors)                      
       
    print('Dataset length is {}'.format(len(predict_dataset.forecast_ids)))

    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar:
        for i,forecast_id in enumerate(predict_dataset.forecast_ids):
            i = predict_dataset.forecast_ids.index(forecast_id)

            test_X_sequences_meta,test_X_sequences_ar,len_diff = predict_dataset.__getitem__(i)

            # into PyTorch format
            test_X_sequences_meta = torch.from_numpy(test_X_sequences_meta).view(1,-1,12)
            test_X_sequences_ar = torch.from_numpy(test_X_sequences_ar).view(1,-1,1)    

            # transform data from Batch x Window x Etc into Batch x Etc format
            test_X_sequences_ar = test_X_sequences_ar.float()
            test_X_sequences_temp = test_X_sequences_meta[:,:,0:5].float()
            test_X_sequences_meta = test_X_sequences_meta[:,:,5:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_1h(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()
            # only the necessary length
            output = output[-len_diff:]

            output = output * predict_dataset.std_dict[forecast_id] + predict_dataset.mean_dict[forecast_id]
            submission_df.loc[submission_df.ForecastId == forecast_id, 'Value'] =  output 
            pbar.update(1)

    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar:
        for i,forecast_id in enumerate(predict_dataset_wrmse.forecast_ids):
            i = predict_dataset_wrmse.forecast_ids.index(forecast_id)

            X_sequences_ar,X_sequences_meta,y_sequences = predict_dataset_wrmse.__getitem__(i)        
            X_sequences_meta = torch.from_numpy(X_sequences_meta).view(1,-1,12)
            X_sequences_ar = torch.from_numpy(X_sequences_ar).view(1,-1,1)

            y_true = y_sequences.reshape(-1) * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            # transform data from Batch x Window x Etc into Batch x Etc format
            X_sequences_ar = X_sequences_ar.float()
            X_sequences_temp = X_sequences_meta[:,:,0:5].float()
            X_sequences_meta = X_sequences_meta[:,:,5:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_1h(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()

            output = output * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            wrmse_val = WRMSE(y_true, output)
            r2_score_val = metrics.r2_score(y_true, output)
            stat_df = stat_df.append(pd.DataFrame([dict(zip(stat_cols,[forecast_id,r2_score_val,wrmse_val]))]))
            pbar.update(1)
                       
    print('Predicting for 15 min series ...')                       

    predict_dataset = S2SDataset(df = trainable_df,
                         series_type = '15_mins',
                         in_sequence_len = 192,
                         out_sequence_len = 192,
                         target = 'Value',
                         mode = 'test',
                         split_mode = 'random',
                         predictors = predictors)

    predict_dataset_wrmse = S2SDataset(df = trainable_df,
                         series_type = '15_mins',
                         in_sequence_len = 192,
                         out_sequence_len = 192,
                         target = 'Value',
                         mode = 'evaluate_wrmse',
                         split_mode = 'random',
                         predictors = predictors)                      
               
    print('Dataset length is {}'.format(len(predict_dataset.forecast_ids)))

    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar:
        for i,forecast_id in enumerate(predict_dataset.forecast_ids):
            i = predict_dataset.forecast_ids.index(forecast_id)

            test_X_sequences_meta,test_X_sequences_ar,len_diff = predict_dataset.__getitem__(i)

            # into PyTorch format
            test_X_sequences_meta = torch.from_numpy(test_X_sequences_meta).view(1,-1,12)
            test_X_sequences_ar = torch.from_numpy(test_X_sequences_ar).view(1,-1,1)    

            # transform data from Batch x Window x Etc into Batch x Etc format
            test_X_sequences_ar = test_X_sequences_ar.float()
            test_X_sequences_temp = test_X_sequences_meta[:,:,0:5].float()
            test_X_sequences_meta = test_X_sequences_meta[:,:,5:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_15m(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()
            # only the necessary length
            output = output[-len_diff:]

            output = output * predict_dataset.std_dict[forecast_id] + predict_dataset.mean_dict[forecast_id]
            submission_df.loc[submission_df.ForecastId == forecast_id, 'Value'] =  output 
            pbar.update(1)

    with tqdm.tqdm(total=len(predict_dataset.forecast_ids)) as pbar:
        for i,forecast_id in enumerate(predict_dataset_wrmse.forecast_ids):
            i = predict_dataset_wrmse.forecast_ids.index(forecast_id)

            X_sequences_ar,X_sequences_meta,y_sequences = predict_dataset_wrmse.__getitem__(i)        
            X_sequences_meta = torch.from_numpy(X_sequences_meta).view(1,-1,12)
            X_sequences_ar = torch.from_numpy(X_sequences_ar).view(1,-1,1)

            y_true = y_sequences.reshape(-1) * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            # transform data from Batch x Window x Etc into Batch x Etc format
            X_sequences_ar = X_sequences_ar.float()
            X_sequences_temp = X_sequences_meta[:,:,0:5].float()
            X_sequences_meta = X_sequences_meta[:,:,5:].long()

            x_temp_var = torch.autograd.Variable(test_X_sequences_temp).cuda(async=True)
            x_meta_var = torch.autograd.Variable(test_X_sequences_meta).cuda(async=True)
            x_ar_var = torch.autograd.Variable(test_X_sequences_ar).cuda(async=True)

            # compute output
            output = model_15m(x_temp_var,x_meta_var,x_ar_var)    
            output = output[0,:].data.cpu().numpy()

            output = output * predict_dataset_wrmse.std_dict[forecast_id] + predict_dataset_wrmse.mean_dict[forecast_id]

            wrmse_val = WRMSE(y_true, output)
            r2_score_val = metrics.r2_score(y_true, output)
            stat_df = stat_df.append(pd.DataFrame([dict(zip(stat_cols,[forecast_id,r2_score_val,wrmse_val]))]))
            pbar.update(1)                   

    # submit zeroes and averages
    print('Submitting averages ... ')
    with tqdm.tqdm(total=len(submit_averages)) as pbar:
        for forecast_id in submit_averages:
            submission_df.loc[submission_df.ForecastId == forecast_id, 'Value'] = data_df[data_df.ForecastId == forecast_id].mean()
            pbar.update(1)

    print('Submitting zeroes ... ')
    with tqdm.tqdm(total=len(submit_zeroes)) as pbar:
        for forecast_id in submit_zeroes:
            submission_df.loc[submission_df.ForecastId == forecast_id, 'Value'] = 0
            pbar.update(1)
            
    print('Using short sequence data from other model for {} series'.format(len(use_last_window)))
    previous_preds = pd.read_csv('../submissions/blended_lstm_forests.csv')
    with tqdm.tqdm(total=len(use_last_window)) as pbar:   
        for forecast_id in use_last_window:
            submission_df.loc[submission_df.ForecastId == forecast_id,'Value'] = previous_preds.loc[previous_preds.ForecastId == forecast_id,'Value']
            pbar.update(1)

    
    stat_df.to_csv('forest_stats_{}.csv'.format(LOGNUMBER))
    submission_df['Value'] = submission_df['Value'].fillna(value=0)
    submission_df.to_csv('../submissions/forecast_{}.csv'.format(LOGNUMBER))
                   
if __name__ == '__main__':
    main()