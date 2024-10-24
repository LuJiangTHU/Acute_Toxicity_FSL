import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCL(nn.Module):  # feed-forward layer
    def __init__(self, in_features, out_features, dropout_prob = 0):
        super(FCL, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Linear',nn.Linear(in_features=in_features,
                                                  out_features=out_features,
                                                  bias=True))
        self.layers.add_module('BatchNorm',nn.BatchNorm1d(out_features))
        self.layers.add_module('Dropout', nn.Dropout(dropout_prob))
        # self.layers.add_module('Relu', nn.Tanh())
        self.layers.add_module('Relu', nn.ReLU())

    def forward(self, x):
        out  = self.layers(x)
        return out


class GCN_layer(nn.Module): # graph convolution layer
    def __init__(self, input_dim, output_dim, A, nonlinear=True):
        super(GCN_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.W = nn.Parameter(0.1 * (torch.rand(self.input_dim, self.output_dim)-0.5),
                              requires_grad=True)
        self.nonlinear_ = nonlinear

    def forward(self, x):
        # x: (num_nodes, input_dim)
        x = torch.mm(x, self.W)
        x = torch.mm(self.A, x)   #(num_nodes, output_dim)
        if self.nonlinear_ == True:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class Correlation_layer(nn.Module):  # adjoint correlation layer
    def __init__(self, num_node, fea_dim):
        super(Correlation_layer, self).__init__()
        self.num_node = num_node
        self.fea_dim = fea_dim
        self.W = nn.Parameter(0.1 * (torch.rand(self.num_node, self.fea_dim)-0.5),
                              requires_grad=True)  # (num_nodes, batchsize)

    def forward(self, input_mlp, input_gcn):
        # input_mlp: (batchsize, n_dim)
        # input_gcn: (num_nodes, n_dim)
        assert (input_mlp.shape[1] == input_gcn.shape[1] and input_mlp.shape[1] == self.fea_dim)
        x1 = torch.mm(input_mlp, F.leaky_relu(input_gcn.t(), negative_slope=0.1))  #(batchsize, num_nodes)
        x2 = torch.mm(x1, self.W)  #(batchsize, n_dim)
        return x2 + input_mlp  #(batchsize, n_dim)


class CorrelationNet(nn.Module):
    def __init__(self, opt):
        super(CorrelationNet, self).__init__()
        self.in_features_dnn = opt['in_features_dnn']  # 1024
        self.in_features_gcn = opt['in_features_gcn']  # 26
        self.out_features = opt['out_features']
        self.num_layers = opt['num_layers']
        self.Dropout_p = opt['Dropout_p']
        self.num_node = opt['num_task']

        A = self.init_adjacency_matrix()
        self.A = nn.Parameter(A, requires_grad=False)

        gcn_input = self.init_gcn_input()
        self.gcn_input = nn.Parameter(gcn_input, requires_grad=False)

        self.tail_w = nn.Parameter(0.1 * torch.rand(self.out_features[-1], self.num_node), requires_grad=True)

        if type(self.out_features) == int:
            self.out_features = [self.out_features for _ in range(self.num_layers)]
        assert (type(self.out_features) ==list and len(self.out_features) == self.num_layers)

        if type(self.Dropout_p) == float:
            self.Dropout_p =[self.Dropout_p for _ in range(self.num_layers)]
        assert(type(self.Dropout_p)==list and len(self.Dropout_p)==self.num_layers)

        num_features_dnn = [self.in_features_dnn, ] + self.out_features  # [1024, 768, 512, 384, 64]
        num_features_gcn = [self.in_features_gcn, ] + self.out_features  # [26, 768, 512, 384, 64]


        self.dnn = nn.ModuleList(
            [FCL(in_features =num_features_dnn[i],
                 out_features=num_features_dnn[i+1],
                 dropout_prob=self.Dropout_p[i]) for i in range(self.num_layers)]
        )


        self.gcn = nn.ModuleList(
            [GCN_layer(input_dim =num_features_gcn[i],
                       output_dim=num_features_gcn[i+1],
                       A = self.A,
                       nonlinear=False) for i in range(self.num_layers)]

        )


        self.correlation_net = nn.ModuleList(
            [Correlation_layer(num_node=self.num_node, fea_dim=d) for d in self.out_features]
        )


    def forward(self, x):
        # x : (batchsize, 1024)

        x_gcn = self.gcn_input

        for i in range(self.num_layers):
            x = self.dnn[i](x)
            x_gcn = self.gcn[i](x_gcn)

            x = self.correlation_net[i](x, x_gcn)
            if i < self.num_layers-1:
                x_gcn = F.leaky_relu(x_gcn, negative_slope=0.1)


        # x: (batchsize, 64)
        # x_gcn: (num_task, 64)

        # print('--x：',x)
        # print('--gcn:', x_gcn)
        out1 = torch.mm(x, x_gcn.t())    #(batchsize, num_task)
        out2 = torch.mm(x, self.tail_w)  #(batchsize, num_task)
        out = out1 + out2

        return out



    def init_adjacency_matrix(self, file_read='./Adjecency_Matrix1.csv'):
        A = torch.FloatTensor(np.loadtxt(file_read, delimiter=',')[1:])
        AA = torch.eye(A.shape[0]) + A
        D = torch.diag(1 / torch.sqrt(AA.sum(1)))
        return D @ AA @ D


    def init_gcn_input(self, file_read='./input_feature.csv'):
        input_feature = torch.FloatTensor(np.loadtxt(file_read, delimiter=',')[1:])
        return input_feature



def create_model(opt):
    return CorrelationNet(opt)


