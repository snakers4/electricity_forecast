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
                                                       output_size = in_sequence_len+out_sequence_len)
        
        self.year_emb, emb_size, output_size = create_emb(cat_size = 8,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len) 
        
        self.month_emb, emb_size, output_size = create_emb(cat_size = 12,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len)  
        
        self.day_emb, emb_size, output_size = create_emb(cat_size = 31,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len)  
        
        self.hour_emb, emb_size, output_size = create_emb(cat_size = 24,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len)
        
        self.min_emb, emb_size, output_size = create_emb(cat_size = 60,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len)
        
        self.dow_emb, emb_size, output_size = create_emb(cat_size = 7,
                                                       max_emb_size = 50,
                                                       output_size = in_sequence_len+out_sequence_len)        
        
   
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