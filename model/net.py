# Dueling Network型のディープ・ニューラルネットワークの構築
import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelNet(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(DuelNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid, n_out)  # Advantage側
        self.fc3_v = nn.Linear(n_mid, 1)  # 価値V側

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)  # この出力はReLUしない
        val = self.fc3_v(h2).expand(-1, adv.size(1))  # この出力はReLUしない
        # valはadvと足し算するために、サイズを[minibatchx1]から[minibatchx2]にexpandする
        # adv.size(1)は出力する行動の種類数の2

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        # val+advからadvの平均値を引き算する
        # adv.mean(1, keepdim=True) で列方向(行動の種類方向)に平均し、サイズが[minibatch×1]
        # expandで展開して、サイズ[minibatchx2]

        return output


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)  # この出力はReLUしない

        return h3


# 1hour 30minites 15minites
class TimeNet(nn.Module):

    def __init__(self, n_in, n_mid, n_out, n_in_2=30, n_mid_2=10, n_in_3=15, n_mid_3=10):  
        super(TimeNet, self).__init__()
        self.middle_priod = n_in_2
        self.short_priod = n_in_3

        self.long = nn.Linear(n_in, n_mid) 
        self.long_mid = nn.Linear(n_mid, n_out)

        self.middle = nn.Linear(n_in_2, n_mid_2) 
        self.middle_mid = nn.Linear(n_mid_2, n_out)

        self.short = nn.Linear(n_in_3, n_mid_3) 
        self.short_mid = nn.Linear(n_mid_3, n_out)

        self.out = nn.Linear(n_out * 3, n_out)

    def forward(self, state):
        long_out = F.relu(self.long(state))
        long_out = self.long_mid(long_out)
        long_out = long_out - torch.mean(long_out, axis=1, keepdim=True).expand_as(long_out)

        middle_out = F.relu(self.middle(state[:,:self.middle_priod]))
        middle_out = self.middle_mid(middle_out)
        middle_out = middle_out - torch.mean(middle_out, axis=1, keepdim=True).expand_as(middle_out)

        short_out = F.relu(self.short(state[:,:self.short_priod]))
        short_out = self.short_mid(short_out)
        short_out = short_out - torch.mean(short_out, axis=1, keepdim=True).expand_as(short_out)

        output = long_out + middle_out + short_out

        #output = torch.cat([long_out, middle_out, short_out], axis=1)
        #output = self.out(output)

        return output
        