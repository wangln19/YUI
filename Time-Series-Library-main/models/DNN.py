import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# development of a DNN model
class Model(nn.Module):
    """
    selected features: t_delta = [Period, 2*Period, 3*Period, 7*Period]
    """

    def __init__(self, configs):
        """
    
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if configs.data == 'iso_new_england':
            self.period = 24
        elif configs.data == 'shanxi':
            self.period = 96
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.rows_to_select = list(range(-7*self.period, -6*self.period)) + list(range(-3*self.period, 0))
        self.cols_to_select = [0, 3, 6]
        self.linear1 = nn.Linear(12*96, 2000).to(device)
        self.linear2 = nn.Linear(2000, 1000).to(device)
        self.linear3 = nn.Linear(1000, self.period).to(device)

    def forecast(self, x):
        x_slt = x[np.ix_(np.arange(x.shape[0]), self.rows_to_select, self.cols_to_select)].reshape(x.shape[0], -1)
        # DNN
        dnn_out = F.relu(self.linear1(x_slt))
        dnn_out = F.relu(self.linear2(dnn_out))
        dnn_out = self.linear3(dnn_out).reshape(x.shape[0], -1, 1)
        return dnn_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

