import torch
import torch.nn as nn
from torch.autograd import Variable

def create_emb(cat_size = 7,
               max_emb_size = 50,
               output_size = 10,
               non_trainable=False):

    emb_size = min([(cat_size+2)//3, max_emb_size])    
    emb = nn.Embedding(output_size, emb_size)

    if non_trainable:
        for param in emb.parameters(): 
            param.requires_grad = False
    return emb, emb_size, output_size

class E2ELSTM_day(nn.Module):
    def __init__(self,
                 in_sequence_len = 30,
                 out_sequence_len = 30,
                 features_meta_total = 44,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 30,
                 ar_hidden_layer_length = 30,
                 meta_hidden_layers = 2,
                 ar_hidden_layers = 1,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 256):
        
        super(E2ELSTM_day, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 72,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)          
   
        self.lstm_meta = nn.LSTM(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.lstm_ar = nn.LSTM(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)  


        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, out_sequence_len),
        )
        
        """ 
        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.ReLU(True),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.ReLU(True),
            nn.Linear(classifier_hidden_length, out_sequence_len),
            nn.ReLU(True)
        ) 
        """


    
    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_month = self.month_emb(x_meta[:,:,1])
        x_day = self.day_emb(x_meta[:,:,2])
        x_dow = self.dow_emb(x_meta[:,:,3])        

        x_meta = torch.cat([x_temp,x_hol,x_month,x_day,x_dow],dim=2)
        
        # initial values for LSTMs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda()) 
        c0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda())
        
        h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        c0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda())         
        
        # Forward propagate LSTMs
        out_meta, _ = self.lstm_meta(x_meta, (h0_meta, c0_meta))  
        out_meta = out_meta[:, -1, :]
        
        out_ar, _ = self.lstm_ar(x_ar, (h0_ar, c0_ar))  
        out_ar = out_ar[:, -1, :]    
        
        out = torch.cat([out_meta,out_ar],dim=1)
       
        out = self.classifier(out)

        return out

class E2ELSTM(nn.Module):
    def __init__(self,
                 in_sequence_len = 192,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 192,
                 ar_hidden_layer_length = 192,
                 meta_hidden_layers = 2,
                 ar_hidden_layers = 1,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 192 * 2):
        
        super(E2ELSTM, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 66,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = 100) 
        
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)        
        
   
        self.lstm_meta = nn.LSTM(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.lstm_ar = nn.LSTM(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)  


        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, out_sequence_len),
        )
        
        """ 
        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.ReLU(True),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.ReLU(True),
            nn.Linear(classifier_hidden_length, out_sequence_len),
            nn.ReLU(True)
        ) 
        """


    
    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow],dim=2)
        
        # initial values for LSTMs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda()) 
        c0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda())
        
        h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        c0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda())         
        
        # Forward propagate LSTMs
        out_meta, _ = self.lstm_meta(x_meta, (h0_meta, c0_meta))  
        out_meta = out_meta[:, -1, :]
        
        out_ar, _ = self.lstm_ar(x_ar, (h0_ar, c0_ar))  
        out_ar = out_ar[:, -1, :]    
        
        out = torch.cat([out_meta,out_ar],dim=1)
       
        out = self.classifier(out)

        return out

def WRMSE(y, pred_y):
    sq_err = np.square(y - pred_y)
    Tn = len(y)
    t = np.arange(1, Tn+1)
    W = (3*Tn - 2*t + 1)/(2*Tn**2)
    mu = y.mean()+1e-15
    return np.sqrt(np.sum(W*sq_err))/mu

class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()
    def forward(self, input, target):
        sq_err = (input - target) ** 2
        Tn = len(target)
        t = torch.arange(1, Tn+1)
        W = Variable( (3*Tn - 2*t + 1)/(2*Tn**2) )
        mu = target.mean()+1e-15
        
        return ((torch.sum(W*sq_err)) ** 0.5)/mu 

class E2ELSTMSONLY(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 features_final_total = 512*2,
                 meta_hidden_layer_length = 512,
                 ar_hidden_layer_length = 512,
                 final_hidden_layer_length = 1,
                 meta_hidden_layers = 2,
                 ar_hidden_layers = 2,
                 final_hidden_layers = 2,
                 lstm_dropout = 0,
                 merge_type = 'cat'):
        
        super(E2ELSTMSONLY, self).__init__()

        self.merge_type = merge_type
        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.final_hidden_layer_length = final_hidden_layer_length
        
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers
        self.final_hidden_layers = final_hidden_layers
        self.out_sequence_len = out_sequence_len
        
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 66,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = 100) 
        
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)        
        
   
        self.lstm_meta = nn.LSTM(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.lstm_ar = nn.LSTM(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.lstm_final = nn.LSTM(features_final_total,
                            final_hidden_layer_length,
                            final_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)        
   
    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow],dim=2)
        
        # initial values for LSTMs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda()) 
        c0_meta = Variable(torch.zeros(self.meta_hidden_layers, x_meta.size(0), self.meta_hidden_layer_length).cuda())
        
        h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        c0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda())
        
        # Forward propagate LSTMs
        out_meta, _ = self.lstm_meta(x_meta, (h0_meta, c0_meta))  
        out_meta = out_meta[:, -self.out_sequence_len:, :]
        
        out_ar, _ = self.lstm_ar(x_ar, (h0_ar, c0_ar))  
        out_ar = out_ar[:, -self.out_sequence_len:, :]    

        if self.merge_type == 'cat':
            out = torch.cat([out_meta,out_ar],dim=2)
            h0_final = Variable(torch.zeros(self.final_hidden_layers, out.size(0), self.final_hidden_layer_length).cuda()) 
            c0_final = Variable(torch.zeros(self.final_hidden_layers, out.size(0), self.final_hidden_layer_length).cuda()) 
            out, _ = self.lstm_final(out, (h0_final, c0_final))  
        elif self.merge_type == 'sum':
            out = out_meta + out_ar
            h0_final = Variable(torch.zeros(self.final_hidden_layers, out.size(0), self.final_hidden_layer_length).cuda()) 
            c0_final = Variable(torch.zeros(self.final_hidden_layers, out.size(0), self.final_hidden_layer_length).cuda())             
            out, _ = self.lstm_final(out, (h0_final, c0_final))
           
        out = out.contiguous().view(-1,self.out_sequence_len)
        return out

class E2EGRU(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 192,
                 ar_hidden_layer_length = 192,
                 meta_hidden_layers = 3,
                 ar_hidden_layers = 3,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 192 * 2):
        
        super(E2EGRU, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 66,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = 100) 
        
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)        
        
   
        self.gru_meta = nn.GRU(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)

        
        self.gru_ar = nn.GRU(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)  


        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, out_sequence_len),
        )
    
    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow],dim=2)
        
        # initial values for GRUs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers,  x_meta.size(0), self.meta_hidden_layer_length).cuda())
        h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        
        # Forward propagate GRUs
        out_meta, _ = self.gru_meta(x_meta, h0_meta)  
        out_meta = out_meta[:, -1, :]
        
        out_ar, _ = self.gru_ar(x_ar, h0_ar)  
        out_ar = out_ar[:, -1, :]    
        
        out = torch.cat([out_meta,out_ar],dim=1)
       
        out = self.classifier(out)

        return out    
    
class EncoderDecoderGRU(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 192,
                 ar_hidden_layer_length = 192,
                 meta_hidden_layers = 3,
                 ar_hidden_layers = 3,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 192 * 2,
                 use_output = 'last',
                 ):
        
        super(EncoderDecoderGRU, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        self.use_output = use_output
                 
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 66,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = 100) 
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)        
   
        self.encoder_gru_meta = nn.GRU(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)

        self.encoder_gru_ar = nn.GRU(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.decoder_gru_meta = nn.GRU(meta_hidden_layer_length,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)

        self.decoder_gru_ar = nn.GRU(ar_hidden_layer_length,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)          

        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, out_sequence_len),
        )

    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow],dim=2)
        
        # Encoder part of the network # 
        # Initial values for GRUs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers,  x_meta.size(0), self.meta_hidden_layer_length).cuda())
        h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        # Forward propagate GRUs
        meta_encoded, meta_hidden = self.encoder_gru_meta(x_meta, h0_meta)
        ar_encoded, ar_hidden = self.encoder_gru_ar(x_ar, h0_ar)

        # Decoder part of the network # 
        # Use hidden states of encoders for decode
        meta_decoded,_ = self.decoder_gru_meta(meta_encoded, meta_hidden)
        ar_decoded,_ = self.decoder_gru_ar(ar_encoded, ar_hidden)          
        
        if self.use_output  == 'last':
            meta_decoded = meta_decoded[:, -1, :]
            ar_decoded = ar_decoded[:, -1, :]
        elif self.use_output  == 'first': 
            meta_decoded = meta_decoded[:, 0, :]
            ar_decoded = ar_decoded[:, 0, :]                 

        out = torch.cat([meta_decoded,ar_decoded],dim=1)
        out = self.classifier(out)

        return out   
    
class EncoderDecoderGRUNoAr(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 192,
                 ar_hidden_layer_length = 192,
                 meta_hidden_layers = 3,
                 ar_hidden_layers = 3,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 192 * 2,
                 use_output = 'last',
                 ):
        
        super(EncoderDecoderGRUNoAr, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        self.use_output = use_output
                 
        # create an embedding for each categorical feature
        self.hol_emb, emb_size, output_size = create_emb(cat_size = 66,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = 100) 
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = 100)  
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = 100)
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = 100)        
   
        self.encoder_gru_meta = nn.GRU(features_meta_total,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)
        
        self.decoder_gru_meta = nn.GRU(meta_hidden_layer_length,
                            meta_hidden_layer_length,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=False)

        self.classifier = nn.Sequential(
            nn.Linear(meta_hidden_layer_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, classifier_hidden_length),
            nn.BatchNorm2d(classifier_hidden_length),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_hidden_length, out_sequence_len),
        )

    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow],dim=2)
        
        # Encoder part of the network # 
        # Initial values for GRUs
        h0_meta = Variable(torch.zeros(self.meta_hidden_layers,  x_meta.size(0), self.meta_hidden_layer_length).cuda())
        # Forward propagate GRUs
        meta_encoded, meta_hidden = self.encoder_gru_meta(x_meta, h0_meta)

        # Decoder part of the network # 
        # Use hidden states of encoders for decode
        meta_decoded,_ = self.decoder_gru_meta(meta_encoded, meta_hidden)
        
        if self.use_output  == 'last':
            meta_decoded = meta_decoded[:, -1, :]
        elif self.use_output  == 'first': 
            meta_decoded = meta_decoded[:, 0, :]

        out = self.classifier(meta_decoded)

        return out       
    
class EncoderDecoderGRUSK(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 72,
                 features_ar_total = 1,
                 meta_hidden_layer_length = 192,
                 ar_hidden_layer_length = 192,
                 meta_hidden_layers = 3,
                 ar_hidden_layers = 3,
                 lstm_dropout = 0.5,
                 classifier_hidden_length = 192 * 2,
                 use_output = 'last',
                 use_bi = False
                 ):
        
        super(EncoderDecoderGRUSK, self).__init__()

        self.meta_hidden_layer_length = meta_hidden_layer_length
        self.ar_hidden_layer_length = ar_hidden_layer_length
        self.meta_hidden_layers = meta_hidden_layers
        self.ar_hidden_layers = ar_hidden_layers      
        self.use_output = use_output
        self.use_bi = use_bi
                 
        # create an embedding for each categorical feature
        self.hol_emb, _, _ = create_emb(cat_size = 66,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.year_emb, _, _ = create_emb(cat_size = 8,
                                       max_emb_size = 50,
                                       output_size = 100) 
        self.month_emb, _, _ = create_emb(cat_size = 12,
                                       max_emb_size = 50,
                                       output_size = 100)  
        self.day_emb, _, _ = create_emb(cat_size = 31,
                                       max_emb_size = 50,
                                       output_size = 100)  
        self.hour_emb, _, _ = create_emb(cat_size = 24,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.min_emb, _, _ = create_emb(cat_size = 60,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.dow_emb, _, _ = create_emb(cat_size = 7,
                                       max_emb_size = 50,
                                       output_size = 100)
        
        # additional binary and categorical embeddings
        self.has_weather_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)        
        self.is_hol_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.is_weekend_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.series_type_emb, _, _ = create_emb(cat_size = 3,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.site_id_emb, _, _ = create_emb(cat_size = 267,
                                           max_emb_size = 50,
                                           output_size = 300)        
   

        self.encoder_gru_meta = nn.GRU(features_meta_total,
                            meta_hidden_layer_length ,
                            meta_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=use_bi)

        self.encoder_gru_ar = nn.GRU(features_ar_total,
                            ar_hidden_layer_length,
                            ar_hidden_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=use_bi)            
        
        if self.use_bi == False:
            self.decoder_gru_meta = nn.GRU(meta_hidden_layer_length,
                                meta_hidden_layer_length,
                                meta_hidden_layers,
                                batch_first=True,
                                dropout=lstm_dropout,
                                bidirectional=use_bi)

            self.decoder_gru_ar = nn.GRU(ar_hidden_layer_length,
                                ar_hidden_layer_length,
                                ar_hidden_layers,
                                batch_first=True,
                                dropout=lstm_dropout,
                                bidirectional=use_bi)
        else:
            self.decoder_gru_meta = nn.GRU(meta_hidden_layer_length * 2,
                                meta_hidden_layer_length,
                                meta_hidden_layers,
                                batch_first=True,
                                dropout=lstm_dropout,
                                bidirectional=use_bi)

            self.decoder_gru_ar = nn.GRU(ar_hidden_layer_length * 2,
                                ar_hidden_layer_length,
                                ar_hidden_layers,
                                batch_first=True,
                                dropout=lstm_dropout,
                                bidirectional=use_bi)            
        if self.use_bi == False:
            self.classifier = nn.Sequential(
                nn.Linear(meta_hidden_layer_length + ar_hidden_layer_length, classifier_hidden_length),
                nn.BatchNorm2d(classifier_hidden_length),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_hidden_length, classifier_hidden_length),
                nn.BatchNorm2d(classifier_hidden_length),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_hidden_length, out_sequence_len),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear((meta_hidden_layer_length + ar_hidden_layer_length)*2, classifier_hidden_length),
                nn.BatchNorm2d(classifier_hidden_length),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_hidden_length, classifier_hidden_length),
                nn.BatchNorm2d(classifier_hidden_length),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_hidden_length, out_sequence_len),
            )            

    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_has_weather = self.has_weather_emb(x_meta[:,:,7])
        x_is_hol = self.is_hol_emb(x_meta[:,:,8])
        x_is_weekend = self.is_weekend_emb(x_meta[:,:,9])
        x_series_type = self.series_type_emb(x_meta[:,:,10])        
        x_site_id = self.site_id_emb(x_meta[:,:,11])           
        
        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow,\
                            x_has_weather,x_is_hol,x_is_weekend,x_series_type,x_site_id],dim=2)
        
        # Encoder part of the network # 
        # Initial values for GRUs
        if self.use_bi == False:
            h0_meta = Variable(torch.zeros(self.meta_hidden_layers,  x_meta.size(0), self.meta_hidden_layer_length).cuda())
            h0_ar = Variable(torch.zeros(self.ar_hidden_layers, x_ar.size(0), self.ar_hidden_layer_length).cuda()) 
        else:
            h0_meta = Variable(torch.zeros(self.meta_hidden_layers * 2,  x_meta.size(0), self.meta_hidden_layer_length).cuda())
            h0_ar = Variable(torch.zeros(self.ar_hidden_layers * 2, x_ar.size(0), self.ar_hidden_layer_length).cuda())             
        # Forward propagate GRUs
        meta_encoded, meta_hidden = self.encoder_gru_meta(x_meta, h0_meta)
        ar_encoded, ar_hidden = self.encoder_gru_ar(x_ar, h0_ar)

        # Decoder part of the network # 
        # Use hidden states of encoders for decode
        meta_decoded,_ = self.decoder_gru_meta(meta_encoded, meta_hidden)
        ar_decoded,_ = self.decoder_gru_ar(ar_encoded, ar_hidden)          
        
        if self.use_output  == 'last':
            meta_decoded = meta_decoded[:, -1, :]
            ar_decoded = ar_decoded[:, -1, :]
        elif self.use_output  == 'first': 
            meta_decoded = meta_decoded[:, 0, :]
            ar_decoded = ar_decoded[:, 0, :]                 

        out = torch.cat([meta_decoded,ar_decoded],dim=1)
        out = self.classifier(out)

        return out      
    
class DilatedConvModel(nn.Module):
    def __init__(self,
                 in_sequence_len = 700,
                 out_sequence_len = 192,
                 features_meta_total = 138,
                 features_ar_total = 7,
                 ):
        
        super(DilatedConvModel, self).__init__()

        # create an embedding for each categorical feature
        self.hol_emb, _, _ = create_emb(cat_size = 66,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.year_emb, _, _ = create_emb(cat_size = 8,
                                       max_emb_size = 50,
                                       output_size = 100) 
        self.month_emb, _, _ = create_emb(cat_size = 12,
                                       max_emb_size = 50,
                                       output_size = 100)  
        self.day_emb, _, _ = create_emb(cat_size = 31,
                                       max_emb_size = 50,
                                       output_size = 100)  
        self.hour_emb, _, _ = create_emb(cat_size = 24,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.min_emb, _, _ = create_emb(cat_size = 60,
                                       max_emb_size = 50,
                                       output_size = 100)
        self.dow_emb, _, _ = create_emb(cat_size = 7,
                                       max_emb_size = 50,
                                       output_size = 100)
        
        # additional binary and categorical embeddings
        self.has_weather_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)        
        self.is_hol_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.is_weekend_emb, _, _ = create_emb(cat_size = 2,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.series_type_emb, _, _ = create_emb(cat_size = 3,
                                           max_emb_size = 50,
                                           output_size = 10)
        self.site_id_emb, _, _ = create_emb(cat_size = 267,
                                           max_emb_size = 50,
                                           output_size = 300)        
   

        self.encoder_meta = self.classifier = nn.Sequential(
            nn.Conv1d(features_meta_total, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4, groups=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=8, dilation=8, groups=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),             
        )
    
        self.encoder_ar = self.classifier = nn.Sequential(
            nn.Conv1d(features_ar_total, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4, groups=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=8, dilation=8, groups=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=16, dilation=16, groups=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),             
        )
 
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 * 2, 1024, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True), 
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True), 
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True), 
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(4),
            nn.ReLU(True), 
            nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(True),               
        )        

    def forward(self,
                x_temp,
                x_meta,
                x_ar):
        
        # embed and extract various features
        x_hol = self.hol_emb(x_meta[:,:,0])
        x_year = self.year_emb(x_meta[:,:,1])
        x_month = self.month_emb(x_meta[:,:,2])
        x_day = self.day_emb(x_meta[:,:,3])
        x_hour = self.hour_emb(x_meta[:,:,4])
        x_min = self.min_emb(x_meta[:,:,5])
        x_dow = self.dow_emb(x_meta[:,:,6])

        x_has_weather = self.has_weather_emb(x_meta[:,:,7])
        x_is_hol = self.is_hol_emb(x_meta[:,:,8])
        x_is_weekend = self.is_weekend_emb(x_meta[:,:,9])
        x_series_type = self.series_type_emb(x_meta[:,:,10])        
        x_site_id = self.site_id_emb(x_meta[:,:,11])           
        
        x_meta = torch.cat([x_temp,x_hol,x_year,x_month,x_day,x_hour,x_min,x_dow,\
                            x_has_weather,x_is_hol,x_is_weekend,x_series_type,x_site_id],dim=2)\
        
        # transpose tensors to apply dilated convs 
        x_meta = x_meta.view(-1,x_meta.size(2),x_meta.size(1))
        x_ar = x_ar.view(-1,x_ar.size(2),x_ar.size(1))
        
        out = torch.cat((self.encoder_meta(x_meta),self.encoder_ar(x_ar)),dim=1)
        print(out.size())

        out = self.decoder(out)
        print(out.size())

        return out.view(-1,out.size(2))      