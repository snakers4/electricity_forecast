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
import pickle 

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
from ELstm import E2ELSTM,WMSELoss,E2ELSTM_day,E2EGRU,DilatedConvModel


import pandas as pd
import numpy as np
from math import sqrt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# general params
parser = argparse.ArgumentParser(description='PyTorch LSTM training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=30, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    help='model optimizer')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lognumber', '-log', default='test_model', type=str,
                    metavar='LN', help='text id for saving logs')
parser.add_argument('--tensorboard', default=False, type=str2bool,
                    help='Use tensorboard to for loss visualization')
# add later
parser.add_argument('-pr', '--predict', dest='predict', action='store_true',
                    help='generate prediction masks')
parser.add_argument('-pr_train', '--predict_train', dest='predict_train', action='store_true',
                    help='generate prediction masks')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')



# embedding / ar / lstm specific params
# sequence
parser.add_argument('-iseq', '--inp_seq', default=192, type=int,
                    metavar='INPSEQ', help='input sequence length')
parser.add_argument('-oseq', '--out_seq', default=192, type=int,
                    metavar='OUTSEQ', help='output sequence length')
# embeddings
# exact counts hardcoded into the model
parser.add_argument('-fmeta', '--features_meta', default=72, type=int,
                    metavar='FMETA', help='total number of meta-data features')
parser.add_argument('-far', '--features_ar', default=1, type=int,
                    metavar='FMETA', help='total number of ar features')
# series type '15_mins' or '1_hour' or '1_day'
parser.add_argument('--series_type', '-st', default='15_mins', type=str,
                    metavar='ST', help='on which type of time series to train the model')
parser.add_argument('-val_size', '--val_size', default=0.25, type=float,
                    metavar='VS', help='validation set size')

best_val_loss = 100
train_minib_counter = 0
valid_minib_counter = 0

args = parser.parse_args()
print(args)

# remove the log file if it exists if we run the script in the training mode
if not (args.predict or args.predict_train):
    print('Folder {} delete triggered'.format(args.lognumber))
    try:
        shutil.rmtree('tb_logs/{}/'.format(args.lognumber))
    except:
        pass

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    logger = Logger('./tb_logs/{}'.format(args.lognumber))

def main():
    global args, best_val_loss
    global logger
    
    # preprocess data / ETL / interpolation / data curation
    # data_df,train_forecast_ids,normal_forecast_ids,linear_interpolation,last_window,submit_zeroes,submit_averages = preprocess_data()
    
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

    # drop values wo AR features
    data_df = data_df.dropna()

    # override - exclude last window series
    train_forecast_ids = normal_forecast_ids + linear_interpolation

    savva_df = pd.read_feather('../data/forecast/savva_features')
    savva_df['ForecastId'] = savva_df['forecast_id'] 
    savva_df['SiteId'] = savva_df['site_id']
    savva_df['Timestamp'] = savva_df['timestamp']

    cols_2_merge = ['ForecastId', 'Timestamp', # merge columns
                    'closest_distance','closest_temperature', 'closest_temperature_base_diff', 'avg_distance',
                    'avg_temperature', 'avg_temperature_base_diff', 'avg_day_distance','avg_day_temperature', 'avg_day_temperature_base_diff',
                    'avg_month_distance', 'avg_month_temperature','avg_month_temperature_base_diff',
                    'is_holiday','has_weather','cat_site_id', 'time_diff']

    data_df = data_df.merge(savva_df[cols_2_merge], on=['Timestamp','ForecastId']) 


    additional_embeddings = ['has_weather','is_holiday','is_day_off','time_diff','cat_site_id']
    additional_numeric_features = ['closest_distance','closest_temperature', 'closest_temperature_base_diff',
                                'avg_distance','avg_temperature', 'avg_temperature_base_diff',
                                'avg_day_distance','avg_day_temperature', 'avg_day_temperature_base_diff',
                                'avg_month_distance', 'avg_month_temperature','avg_month_temperature_base_diff']
    
    
    # features we use
    if args.series_type == '1_day':
        temp_features = ['Temperature']
        hol_emb_features = ['Holiday']
        time_emb_features = ['month','day','dow']
        target = ['Value']
        predictors = temp_features + hol_emb_features + time_emb_features
        
        model = E2ELSTM_day(in_sequence_len = args.inp_seq,
                         out_sequence_len = args.out_seq,
                         features_meta_total = args.features_meta,
                         features_ar_total = args.features_ar,
                         meta_hidden_layer_length = args.lstm_meta_hid_feat,
                         ar_hidden_layer_length = args.lstm_ar_hid_feat,
                         meta_hidden_layers = args.lstm_meta_hid_lyr,
                         ar_hidden_layers = args.lstm_ar_hid_lyr,
                         lstm_dropout = args.lstm_dropout,
                         classifier_hidden_length = args.mlp_hid_lyr)        
        
    else:
        temp_features = ['Temperature']
        ar_features = ['Value','Value1','Value4','Value12','Value24','Value96','Value168']
        hol_emb_features = ['Holiday']
        time_emb_features = ['year', 'month', 'day', 'hour', 'minute','dow']
        target = ['Value']
        predictors = temp_features + additional_numeric_features + hol_emb_features + time_emb_features + additional_embeddings
        
        # E2EGRU or E2ELSTM
        model = DilatedConvModel(in_sequence_len = args.inp_seq,
                                 out_sequence_len = args.out_seq,
                                 features_meta_total = args.features_meta,
                                 features_ar_total = args.features_ar)

    # model.cuda()
    model = torch.nn.DataParallel(model).cuda()
        
    # select only series we marked as trainable
    # negation is for speed only
    trainable_df = data_df[(~data_df.ForecastId.isin(list(set(data_df.ForecastId.unique()) - set(train_forecast_ids))))]
    
    train_dataset = S2SDataset(df = trainable_df,
                         series_type = args.series_type,
                         in_sequence_len = args.inp_seq,
                         out_sequence_len = args.out_seq,
                         target = 'Value',
                         mode = 'train',
                         split_mode = 'random',
                         predictors = predictors,
                         val_size = args.val_size,
                         ar_features = ar_features)

    val_dataset = S2SDataset(df = trainable_df,
                         series_type = args.series_type,
                         in_sequence_len = args.inp_seq,
                         out_sequence_len = args.out_seq,
                         target = 'Value',
                         mode = 'val',
                         split_mode = 'random',
                         predictors = predictors,
                         val_size = args.val_size,
                         ar_features = ar_features)  
    
    print('Train dataset length is {}'.format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr = args.lr)

    scheduler = ReduceLROnPlateau(optimizer = optimizer,
                                              mode = 'min',
                                              factor = 0.1,
                                              patience = 3,
                                              verbose = True,
                                              threshold = 1e-3,
                                              min_lr = 1e-6
                                              )   

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion)
        
        scheduler.step(val_loss)

        # add code for early stopping here 
        # 
        #
  

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_epoch_loss': train_loss,
                'valid_epoch_loss': val_loss
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)                     
        
        # remember best prec@1 and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            },
            is_best,
            'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
            'weights/{}_best.pth.tar'.format(str(args.lognumber))
        )
   
def train(train_loader, model, criterion, optimizer, epoch):
    global train_minib_counter
    global logger
        
    # for cyclic LR
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (X_sequences_ar,X_sequences_meta,y_sequences) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)        
       
        # transform data from Batch x Window x Etc into Batch x Etc format
        X_sequences_meta = X_sequences_meta.view(-1,X_sequences_meta.size(2),X_sequences_meta.size(3))
        X_sequences_ar = X_sequences_ar.view(-1,X_sequences_ar.size(2),X_sequences_ar.size(3)).float()
        y_sequences = y_sequences.view(-1,y_sequences.size(2)).float()
        # modify here
        X_sequences_temp = X_sequences_meta[:,:,0:13].float()
        X_sequences_meta = X_sequences_meta[:,:,13:].long()

        x_temp_var = torch.autograd.Variable(X_sequences_temp.cuda(async=True))
        x_meta_var = torch.autograd.Variable(X_sequences_meta.cuda(async=True))
        x_ar_var = torch.autograd.Variable(X_sequences_ar.cuda(async=True))
        y_var = torch.autograd.Variable(y_sequences.cuda(async=True))

        # compute output
        output = model(x_temp_var,x_meta_var,x_ar_var)
        loss = criterion(output, y_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], X_sequences_meta.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter) 
            info = {
                'train_lr': current_lr,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter)                     
        
        train_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    print(' * Avg Train Loss {loss.avg:.4f}'.format(loss=losses))         
            
    return losses.avg

def validate(val_loader, model, criterion):
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    losses = AverageMeter()
           
    for i, (X_sequences_ar,X_sequences_meta,y_sequences) in enumerate(val_loader):
        
        # transform data from Batch x Window x Etc into Batch x Etc format
        X_sequences_meta = X_sequences_meta.view(-1,X_sequences_meta.size(2),X_sequences_meta.size(3))
        X_sequences_ar = X_sequences_ar.view(-1,X_sequences_ar.size(2),X_sequences_ar.size(3)).float()
        y_sequences = y_sequences.view(-1,y_sequences.size(2)).float()
        # modify here
        X_sequences_temp = X_sequences_meta[:,:,0:13].float()
        X_sequences_meta = X_sequences_meta[:,:,13:].long()

        x_temp_var = torch.autograd.Variable(X_sequences_temp.cuda(async=True))
        x_meta_var = torch.autograd.Variable(X_sequences_meta.cuda(async=True))
        x_ar_var = torch.autograd.Variable(X_sequences_ar.cuda(async=True))
        y_var = torch.autograd.Variable(y_sequences.cuda(async=True))

        # compute output
        output = model(x_temp_var,x_meta_var,x_ar_var)
        loss = criterion(output, y_var)
        
        # measure accuracy and record loss
        losses.update(loss.data[0], X_sequences_meta.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'valid_loss': losses.val,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Avg Val Loss {loss.avg:.4f}'.format(loss=losses))

    return losses.avg

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()