import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from vmdpy import VMD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# development of a VMD-LSTM model
class Model(nn.Module):
    """
    selected features: t_delta = [1, 2, Period-1, Period, Period+1, 2*Period, 7*Period]
    alpha = 2000       # moderate bandwidth constraint
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    K = 3              # 3 modes
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7
    """

    def __init__(self, configs):
        """
    
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if configs.data == 'iso_new_england_per_point':
            self.period = 24
        elif configs.data == 'shanxi_per_point':
            self.period = 96
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # self.t_delta = [-1, -2, - self.period + 1, - self.period, - self.period - 1, - 2 * self.period, - 7 * self.period]
        self.t_delta = [- self.period, - self.period -1, - self.period - 2, - 2 * self.period, - 2 * self.period - 1, - 2 * self.period - 2, - 7 * self.period]
        self.alpha = 2000
        self.tau = 0.
        self.K = 3
        self.DC = 0
        self.init = 1
        self.tol = 1e-7
        self.lstm = nn.LSTM(input_size=3, hidden_size=10, num_layers=1, batch_first=True).to(device)
        self.linear = nn.Linear(10, 1).to(device)

    def forecast(self, x):
        x_slt = x[:, self.t_delta, :]
        # VMD
        u_all = np.zeros((x_slt.size(0), x_slt.size(1) - 1, self.K))
        for i in range(x_slt.size(0)):
            vmd_input = x_slt[i, :, :].detach().cpu().numpy()
            # print(vmd_input)
            u, u_hat, omega = VMD(vmd_input, alpha=self.alpha, tau=self.tau, K=self.K, DC=self.DC, init=self.init, tol=self.tol)
            u_all[i, :, :] = u.T
        u_all = torch.from_numpy(u_all).float().to(device)
        # LSTM
        lstm_out, _ = self.lstm(u_all)
        # lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)
        # Linear
        dec_out = self.linear(lstm_out)
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

