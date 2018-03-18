import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from multiprocessing import Pool
from Utils import process_categorical_features
import numpy as np

class S2VDataset(data.Dataset):
    def __init__(self,
                 ):
        pass

class S2SDataset(data.Dataset):
    def __init__(self,
                 df = None,
                 series_type =  '15_mins', # '1_hour' '1_day'
                 in_sequence_len = 192,
                 out_sequence_len = 192,
                 target = 'Value',
                 mode = 'train', # train, val or test
                 split_mode = 'random', # random or left or right
                 predictors = []
                 ):
        
        
        print('Creating dataset object ...')
        self.mode = mode
        # select only one type of series
        self.df = df[df.ForecastId_type == series_type]
        
        # factorize features even further
        self.df = process_categorical_features(self.df,['Holiday','year', 'month', 'day', 'hour', 'minute','dow'])
        
        
        # order of predictors is important for CNNs
        self.predictors = predictors
        self.in_sequence_len = in_sequence_len
        self.out_sequence_len = out_sequence_len
        self.target = target
        self.mode = mode
        self.split_mode = split_mode
        
        # store the selected time series' site_ids and forecast_ids
        self.forecast_ids = list(self.df.ForecastId.unique())
        
        # do simple train test split based on forecast_ids
        self.train_f_ids, self.val_f_ids = train_test_split(self.forecast_ids,
                                                 test_size = 0.25,
                                                 random_state = 42,
                                                 )
        
        self.site_ids = list(self.df.SiteId.unique())
        
        # store the averages and std of series in dictionaries
        self.mean_dict = {}
        self.std_dict = {}

        self.__normalize__()
        
    def __produce_idx_one__(self,
                            forecast_id):

        slice_df = self.df[self.df.ForecastId == forecast_id]
        slice_df = slice_df.reset_index()

        # how many training data we have
        trainval_len = slice_df[slice_df.is_train == 1].shape[0]
        # calculate how many rolling windows we have
        trainval_window_count =  trainval_len - self.in_sequence_len - self.out_sequence_len

        slice_df_predictors = slice_df[self.predictors].values
        slice_df_targets = slice_df[self.target].values

        # idx of the trainval subset
        trainval_idx = list(slice_df[(slice_df.is_train == 1)].index)
        # idx of the test subset
        test_idx = list(slice_df[(slice_df.is_train == 0)].index)

        # we always have enough data for several rolling windows for trainval
        trainval_X_sequences_ar = np.asarray( [(slice_df_targets[trainval_idx[window:window+self.in_sequence_len]]) for window in range(0,trainval_window_count)] )
        trainval_X_sequences_meta = np.asarray( [(slice_df_predictors[trainval_idx[window : window+self.in_sequence_len+self.out_sequence_len]]) for window in range(0,trainval_window_count)] ) 
        trainval_y_sequences = np.asarray( [(slice_df_targets[trainval_idx[window+self.in_sequence_len : window+self.in_sequence_len+self.out_sequence_len]]) for window in range(0,trainval_window_count)] )

        len_diff = len(test_idx) - self.out_sequence_len
        # if the test set has standard length
        if len(test_idx) == self.out_sequence_len:
            test_X_sequences_meta = slice_df_predictors[trainval_idx[-self.in_sequence_len:] + test_idx]
            test_X_sequences_ar = slice_df_targets[trainval_idx[-self.in_sequence_len:]]
        # otherwise add several points from the train dataset
        else:
            test_X_sequences_meta = slice_df_predictors[trainval_idx[- self.out_sequence_len - len_diff:] + test_idx]
            test_X_sequences_ar = slice_df_targets[trainval_idx[- self.out_sequence_len - len_diff:]]         

        return trainval_X_sequences_ar,trainval_X_sequences_meta,trainval_y_sequences,test_X_sequences_meta,test_X_sequences_ar,len_diff
    
    def __normalize__(self):
        # normalize all series, store their means and stds
        print('Normalizing features ...')
        
        self.mean_dict = self.df[self.df.is_train==1].groupby(by=['ForecastId'])['Value'].mean().to_dict()
        self.std_dict = self.df[self.df.is_train==1].groupby(by=['ForecastId'])['Value'].std().to_dict()
        
        # isolate series where all values are equal to one value
        # preprocess them separately
        flat_forecast_ids = [(k) for k, v in self.std_dict.items() if v == 0]
        
        # insert means for all series
        self.df['mean'] = self.df.ForecastId.apply(lambda x: self.mean_dict[x])
        # std for zero-std series is 1 by default
        self.df['std'] = 1
        # insert stds only for other series with non-zero std
        self.df.loc[~self.df['ForecastId'].isin(flat_forecast_ids),'std'] = self.df[~self.df['ForecastId'].isin(flat_forecast_ids)].ForecastId.apply(lambda x: self.std_dict[x])

        
        self.df['Value'] = (self.df['Value'] - self.df['mean']) / self.df['std']
        
        # set test set values back to 0
        self.df.loc[self.df.is_train==0,'Value'] = 0
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_f_ids)
        elif self.mode == 'val':
            return len(self.val_f_ids)
        elif self.mode == 'test':
            # test set length is equal to the number of forecast ids
            return len(self.forecast_ids)
        elif self.mode == 'evaluate_wrmse':
            # test set length is equal to the number of forecast ids
            return len(self.forecast_ids)        

    def __getitem__(self, idx):
        if self.split_mode in ['random']:
            if self.mode == 'train':
                trainval_X_sequences_ar,trainval_X_sequences_meta,trainval_y_sequences,_,_,_ = self.__produce_idx_one__(self.train_f_ids[idx])
                # sample a random index of data, e.g. 16 sequences
                # if we have a truncated sample, then pull the same sample 16 times
                if trainval_X_sequences_ar.shape[0] < 16:
                    idx = list(range(0,16))
                    trainval_X_sequences_ar = np.vstack([trainval_X_sequences_ar] * 16)
                    trainval_X_sequences_meta = np.vstack([trainval_X_sequences_meta] * 16)
                    trainval_y_sequences = np.vstack([trainval_y_sequences] * 16)
                else:
                    idx = np.random.randint(trainval_X_sequences_ar.shape[0], size=16)                
                
                return trainval_X_sequences_ar[idx],trainval_X_sequences_meta[idx],trainval_y_sequences[idx]
            elif self.mode == 'evaluate_wrmse':
                trainval_X_sequences_ar,trainval_X_sequences_meta,trainval_y_sequences,_,_,_ = self.__produce_idx_one__(self.forecast_ids[idx])
                idx = -1
                # just use the last available window
                return trainval_X_sequences_ar[idx],trainval_X_sequences_meta[idx],trainval_y_sequences[idx] 
            elif self.mode == 'val':
                trainval_X_sequences_ar,trainval_X_sequences_meta,trainval_y_sequences,_,_,_ = self.__produce_idx_one__(self.val_f_ids[idx])
                # sample a random index of data, e.g. 16 sequences
                # if we have a truncated sample, then pull the same sample 16 times
                if trainval_X_sequences_ar.shape[0] < 16:
                    idx = list(range(0,16))
                    trainval_X_sequences_ar = np.vstack([trainval_X_sequences_ar] * 16)
                    trainval_X_sequences_meta = np.vstack([trainval_X_sequences_meta] * 16)
                    trainval_y_sequences = np.vstack([trainval_y_sequences] * 16)
                else:
                    idx = np.random.randint(trainval_X_sequences_ar.shape[0], size=16)                
                
                return trainval_X_sequences_ar[idx],trainval_X_sequences_meta[idx],trainval_y_sequences[idx]
            elif self.mode == 'test':
                _,_,_,test_X_sequences_meta,test_X_sequences_ar,len_diff = self.__produce_idx_one__(self.forecast_ids[idx])
                return test_X_sequences_meta,test_X_sequences_ar,len_diff
        else:
            raise ValueError('getitem method not implemented for this split mode {}'.format(self.split_mode))