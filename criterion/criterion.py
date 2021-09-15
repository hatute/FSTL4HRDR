from __future__ import print_function
from .memory import ContrastMemory
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch
import time

'''
SNNL_v2
'''


class CrossEntropy_SNNL(nn.Module):
    def __init__(self, T, alpha):
        super(CrossEntropy_SNNL, self).__init__()
        self.T = T
        self.alpha = alpha

    @staticmethod
    def pairwise_distance(x_i, x_j, use_cosine_distance=True):
        return (1. - F.cosine_similarity(x_i, x_j)) if use_cosine_distance else F.pairwise_distance(x_i, x_j)

    @staticmethod
    def exp_dis(x_i, x_j, temperature, use_cosine_distance=True):
        distance = CrossEntropy_SNNL.pairwise_distance(
            x_i, x_j, use_cosine_distance)
        return (-(distance / temperature)).exp_()

    def SNNL(self, x, y):
        # time_start = time.time()
        assert x.shape[0] == y.shape[0]
        len_y = y.shape[0]
        if torch.cuda.is_available():
            init_mask = torch.ones(len_y, len_y).cuda() - \
                torch.eye(len_y).cuda()
        else:
            init_mask = torch.ones(len_y, len_y) - torch.eye(len_y)
        mask = torch.squeeze(y.eq(torch.unsqueeze(y, 1))) + 0.
        top_mask = mask * (init_mask)
        bot_mask = init_mask

        total_sum = torch.tensor(
            0.) if not torch.cuda.is_available() else torch.tensor(0.).cuda()
        for i in range(len_y):
            if top_mask[i].nonzero().numel() == 0:
                top_exp = torch.tensor(
                    0. + 1e-6) if not torch.cuda.is_available() else torch.tensor(0. + 1e-6).cuda()
            else:
                top_fit = torch.stack(
                    [x[i] for i in (top_mask[i] == 1.0).nonzero().squeeze(1)])
                top_base = x[i].repeat(top_fit.shape[0], 1)
                top_dis = CrossEntropy_SNNL.pairwise_distance(
                    top_base, top_fit)
                top_exp = torch.exp(-top_dis / (self.T+1e-6))

            bot_fit = torch.stack(
                [x[i] for i in (bot_mask[i] == 1.0).nonzero().squeeze(1)])
            bot_base = x[i].repeat(bot_fit.shape[0], 1)
            bot_dis = CrossEntropy_SNNL.pairwise_distance(bot_base, bot_fit)
            bot_exp = torch.exp(-bot_dis / (self.T+1e-6))

            log_div = torch.log(top_exp.sum().div_(bot_exp.sum()))
            total_sum.add_(log_div)

        # time_end = time.time()
        return -total_sum.div_(len_y)

    def forward(self, x_r, y_, y):
        snnl = self.SNNL(x=x_r, y=y)
        cross_entropy = F.cross_entropy(y_, y)
        loss = cross_entropy + self.alpha * snnl
        return loss


class SNNL_base(nn.Module):

    def __init__(self, use_cosine_distance=True):
        super(SNNL_base, self).__init__()
        self.use_cosine_distance = use_cosine_distance

    @staticmethod
    def pd(x_i, x_j, use_cosine_distance=True):
        dim = 0 if (len(x_i.shape) == len(x_j.shape) == 1) else 1
        return 1. - F.cosine_similarity(x_i, x_j, dim=dim)

    @staticmethod
    def fits(x_i, x_j, temperature, use_cosine_distance=True):
        distance = SNNL_base.pd(x_i, x_j, use_cosine_distance)
        return (-(distance / temperature)).exp_()

    def SNNL(self, x, y, T):
        # time_start = time.time()
        assert x.shape[0] == y.shape[0]
        sum = torch.tensor(0).double()
        for i in range(len(x)):
            _top = torch.tensor(0).double()
            _bot = torch.tensor(0).double()

            for j in range(len(x)):
                if i == j:
                    continue
                if y[i] == y[j]:
                    _top += self.fits(x[i], x[j], T)
                _bot += self.fits(x[i], x[j], T)
            _div = (_top.add_(1e-9)).div_(_bot)
            # print(_top)
            # print(_bot)
            # print(_div)
            # print("*" * 20)
            sum.add_(torch.log(_div))

        # time_end = time.time()
        return -sum / (x.shape[0])


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * \
            (self.T**2) / y_s.shape[0]
        return loss


# !>>>>>>>>>>>>>>>>>>>>> CRDLoss <<<<<<<<<<<<<<<<<<<<
eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """

    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
                           P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':
    x = torch.tensor([[1, 4, 7, 19], [2, 2, 4, 7], [6, 3, 7, 5], [
                     1, 4, 7, 10], [9, 9, 9, 9]]).float()
    y = torch.tensor([[0], [1], [0], [0], [1]]).float()
    T = 1

    # baseline: (tensor(0.9954, dtype=torch.float64), 0.0020508766174316406)
    #
    # v1: (tensor(0.9954), 0.002264261245727539)
    #
    # v2: (tensor(0.9954), 0.001482248306274414)

    base = SNNL_base()
    a = base.SNNL(x, y, T)
    print("baseline:{}\n".format(base.SNNL(x, y, T)))

    # v1 = SNNL_v1(x, y, T)
    # print("v1:{}\n".format(v1.SNNL(x, y, T)))

    # v2 = SNNL_v2(x, y, T)
    # print("v2:{}\n".format(v2.SNNL(x, y, T)))

    # print(-(torch.sum(torch.log(torch.tensor(1e-5) + torch.tensor(1)))))
