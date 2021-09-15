from torch import nn
import torch.nn.functional as F


class RepNet(nn.Module):
    def __init__(self, base_net, d_rep, n_classes, freeze_base=False):
        super(RepNet, self).__init__()
        self.base_net = base_net
        self.d_rep = d_rep
        self.n_classes = n_classes
        self.freeze_base = freeze_base
        self.linear = nn.Linear(self.d_rep, self.n_classes)
        self.bn = nn.BatchNorm1d(self.d_rep)

    def forward(self, x):
        base_net = self.base_net(x)
        norm = self.bn(F.relu((base_net)))
        output = F.softmax(self.linear(norm), 1)
        return output, base_net
