import sys
from torch.nn import Linear
sys.path.append('/home/lcheng/ZhuJunTong/CIForm-main/ciform/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import math
from torch.utils.data import (DataLoader, Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
torch.set_default_tensor_type(torch.FloatTensor)
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Current device:", device)



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):  # 原来是true
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,active=True):
        input = input.float()
        support = torch.mm(input, self.weight)
        # Debugging information
        # print(f"input shape: {input.shape}")
        # print(f"adj shape: {adj.shape}")
        # print(f"support shape: {support.shape}")
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False).float()

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        emb = torch.sum(beta * z, dim=1)
        return emb, w



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

class STAGCN(nn.Module):
    def __init__(self, n_latent, genenumber, num_classes):
        super(STAGCN, self).__init__()
        self.gnn_1 = GraphConvolution(genenumber, 512)
        self.gnn_2 = GraphConvolution(512, 256)
        self.gnn_3 = GraphConvolution(256, 128)
        self.gnn_4 = GraphConvolution(128, 128)
        self.gnn_5 = GraphConvolution(128, num_classes)
        self.ae = AE(
            n_enc_1=512,
            n_enc_2=256,
            n_enc_3=128,
            n_dec_1=128,
            n_dec_2=256,
            n_dec_3=512,
            n_input=genenumber,
            n_z=128)
        self.scale = nn.Parameter(torch.randn(genenumber))
        self.additive = nn.Parameter(torch.randn(genenumber))
        self.attention1 = Attention(512)
        self.attention2 = Attention(256)
        self.attention3 = Attention(128)
        self.attention4 = Attention(128)


    def forward(self, x, adj1):
        x = x.float()
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        h = self.gnn_1(x, adj1)
        emb = torch.stack([h, tra1], dim=1)
        emb, _ = self.attention1(emb)
        h = self.gnn_2(emb, adj1)
        emb = torch.stack([h, tra2], dim=1)
        emb, _ = self.attention2(emb)
        h = self.gnn_3(emb, adj1)
        emb = torch.stack([h, tra3], dim=1)
        emb, _ = self.attention3(emb)
        h = self.gnn_4(emb, adj1)
        emb = torch.stack([h, z], dim=1)
        emb, _ = self.attention4(emb)
        output = self.gnn_5(emb, adj1, active=False)

        recon = x_bar.to(device)
        output = output.to(device)
        scale = nn.Sigmoid()(self.scale)
        additive = torch.exp(self.additive)

        return output, recon, scale, additive


