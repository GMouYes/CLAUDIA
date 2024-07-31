import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import HypergraphConv
from mask import mask_softmax, mask_mean, mask_max
from torchmetrics.functional import pairwise_cosine_similarity as pss

def save_model(model, args):
    """Save model."""
    torch.save(model.state_dict(), args["outputPath"] + '/' + args["modelPath"])
    print("Saved better model selected by validation.")
    return True


class DecisionMaking(nn.Module):
    """docstring for DecisionMaking"""
    def __init__(self, args):
        super(DecisionMaking, self).__init__()
        self.args = args
        # self.GraphEncoder = GraphEncoder(args)
        self.InstanceEncoder = InstanceEncoder(args)
        self.i2t = nn.ModuleList([nn.Linear(self.args["model_commonDim"], self.args[target]) for target in ['users','phonePlacements','activities']])
    def loss_bce(self, pred, label, weight):
        # loss_fn1 = nn.BCELoss()
        # loss_fn2 = nn.MSELoss()
        rows, columns = label.shape
        users, phonePlacements, activity = self.args["users"], self.args["phonePlacements"], self.args["activities"]
        

        if self.args["predict_user"]:
            label_user = label[:, :users]
            label_pp = label[:, users:users+phonePlacements]
            label_act = label[:, -activity:]

            pred_user = pred[:, :users]
            pred_pp = pred[:, users:users+phonePlacements]
            pred_act = pred[:, -activity:]

            weight_pp = weight[:, users:users+phonePlacements]
            weight_act = weight[:, -activity:]

            # each user has equal weight(mask), weight in ce is class weight
            loss_user_fn = nn.CrossEntropyLoss(reduction=self.args["reduction"])
            loss_user = loss_user_fn(pred_user, torch.argmax(label_user,dim=-1))

            # pp has different weights(mask) to represent missing labels
            loss_pp_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight_pp)
            loss_pp = loss_pp_fn(pred_pp, label_pp)

            # act also has missing labels
            loss_act_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight_act)
            loss_act = loss_act_fn(pred_act, label_act)

            # act is the main target, the other two are contexts
            cls_loss = self.args["lambda_user"] * loss_user + self.args["lambda_pp"] * loss_pp + loss_act
            return cls_loss

        else:
            label_no_user = label[:, users:]
            loss_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight[:, users:])
            return loss_fn(pred, label_no_user)

    def loss_supervised_contrast(self, x, y):
        # (n,c) * (c, n) = (n, n)
        # 0 means no shared positive label
        relation = torch.mm(y, y.T)
        r_0 = (relation==0)
        r_1 = (relation>0)

        # (n, d) * (n, d) = (n, n)
        similarity = torch.mm(x, x.T)
        x_0 = similarity[r_0]
        x_1 = similarity[r_1]

        pos = torch.zeros(x.shape[0]).to(x.device)
        pos = pos.scatter_reduce_(dim=0, index=r_1.nonzero()[:,0], src=x_1, reduce="mean", include_self=False)
        pos = pos.reshape(x.shape[0], 1)

        neg = torch.zeros(x.shape[0]).to(x.device)
        neg = neg.scatter_reduce_(dim=0, index=r_0.nonzero()[:,0], src=x_0, reduce="mean", include_self=False)
        neg = neg.reshape(x.shape[0], 1)

        inputs = torch.cat([pos, neg], dim=1)
        target = torch.zeros(x.shape[0]).long().to(x.device)
        loss_fn = nn.CrossEntropyLoss()
        return self.args["lambda3"] * loss_fn(inputs, target)


    # def loss_contrast(self, g):
    #     return self.args["lambda1"] * self.loss_homo(g) - self.args["lambda2"] * self.loss_hetero(g)

    # def loss_homo(self, g):
    #     loss = [1. - torch.mean(pss(g[i], g[i])) for i in range(3)]
    #     return sum(loss) / 3. 

    # def loss_hetero(self, g):
    #     loss = [1. - torch.mean(pss(g[i], torch.cat([g[j] for j in range(3) if j!=i], dim=0))) for i in range(3)]
    #     return sum(loss) / 3.

    def loss(self, pred, label, weight, x):
        return self.loss_bce(pred, label, weight) + self.loss_supervised_contrast(x, label)


    

    def forward(self, hc_feature, raw_feature):
        # g = self.GraphEncoder(graphData)
        x, x_ = self.InstanceEncoder(hc_feature, raw_feature)

        # matrix multiplication for final prediction
        # result = [torch.mm(x[i], g[i].T) for i in range(3)]

        result = [layer(x[i]) for i, layer in enumerate(self.i2t)]
        if self.args["predict_user"]:
            result = torch.cat(result, axis=1)
        else:
            result = torch.cat(result[1:], axis=1)
        x = torch.cat(x, dim=0)
        return result, x_
        # return result, torch.cat(g, dim=0), x_

class InstanceEncoder(nn.Module):
    """docstring for InstanceEncoder"""
    def __init__(self, args):
        super(InstanceEncoder, self).__init__()
        self.args = args

        self.lstm = nn.LSTM(
            input_size=args["raw_dim"],
            hidden_size=args["hidden_dim"],
            num_layers=args["lstm_num_layers"],
            batch_first=True,
            dropout=args["lstm_dropout"],
            bidirectional=args["lstm_bidirectional"],
            )
        # map x_dim to commonD
        if args["lstm_bidirectional"]:
            self.x_dim = args["hgcn_l1_in_channels"] + 2 * args["hidden_dim"]
        else:
            self.x_dim = args["hgcn_l1_in_channels"] + args["hidden_dim"]
        self.x2c = nn.ModuleList([nn.Linear(self.x_dim, args["model_commonDim"]) for _ in range(3)])
        self.x2c_act = nn.LeakyReLU(args["model_leakySlope_x"])

    def forward(self, fc_feature, raw_feature):
        N, L, raw_dim = raw_feature.shape
        raw_out, _ = self.lstm(raw_feature)
        raw_out = raw_out[:, -1, :].reshape(N, -1)

        x_ = torch.cat([raw_out, fc_feature], dim=-1)
        x = [self.x2c_act(layer(x_)) for i, layer in enumerate(self.x2c)]

        return x, x_


class GraphEncoder(nn.Module):
    """docstring for GraphEncoder"""
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        # 3 different hyperConv, u_pp_a, u_pp, u_a
        # 2 layers
        self.hyper1 = nn.ModuleList([
            HypergraphConv(
                in_channels=args["hgcn_l1_in_channels"],
                out_channels=args["hgcn_l1_out_channels"],
                use_attention=args["hgcn_l1_use_attention"],
                heads=args["hgcn_l1_heads"],
                concat=args["hgcn_l1_concat"],
                negative_slope=args["hgcn_l1_negative_slope"],
                dropout=args["hgcn_l1_dropout"],
                bias=args["hgcn_l1_bias"],
                ) for _ in range(3)
            ])
        self.hyper2 = nn.ModuleList([
            HypergraphConv(
                in_channels=args["hgcn_l2_in_channels"],
                out_channels=args["hgcn_l2_out_channels"],
                use_attention=args["hgcn_l2_use_attention"],
                heads=args["hgcn_l2_heads"],
                concat=args["hgcn_l2_concat"],
                negative_slope=args["hgcn_l2_negative_slope"],
                dropout=args["hgcn_l2_dropout"],
                bias=args["hgcn_l2_bias"],
                ) for _ in range(3)
            ])

        # activation and dropout after each conv
        self.act1 = nn.Sequential(
            nn.LeakyReLU(args["hgcn_l1_after_leakySlope"]),
            nn.Dropout(p=args["model_dropout1"]),
            )
        self.act2 = nn.Sequential(
            nn.LeakyReLU(args["hgcn_l2_after_leakySlope"]),
            nn.Dropout(p=args["model_dropout2"]),
            )
        # linear and act before each conv
        self.layer0_linear = nn.ModuleList([nn.Linear(args["hgcn_l1_in_channels"], args["hgcn_l1_in_channels"]) for _ in range(3)])
        self.layer0_act = nn.LeakyReLU(args["hgcn_l1_before_leakySlope"])

        self.layer1_linear = nn.ModuleList([nn.Linear(args["hgcn_l1_out_channels"], args["hgcn_l1_out_channels"]) for _ in range(3)])
        self.layer1_act = nn.LeakyReLU(args["hgcn_l2_before_leakySlope"])

        # map g_dim to commonD
        self.g2c = nn.ModuleList([nn.Linear(args["hgcn_l2_out_channels"], args["model_commonDim"]) for _ in range(3)])
        self.g2c_act = nn.LeakyReLU(args["model_leakySlope_g"])
        
        # init two conv layers
        for m in self.hyper1:
            m.reset_parameters()
        for m in self.hyper2:
            m.reset_parameters()


    def forward(self, graphData):
        # prepare graph info
        g, hyperWeight, hyperAttr = graphData[:3]
        hyperIndex = graphData[3:]

        # layer 0
        g = [
            g[:self.args["users"], :], 
            g[self.args["users"]:self.args["users"]+self.args["phonePlacements"], :], 
            g[self.args["users"]+self.args["phonePlacements"]:, :],
            ]
        g = [layer(g[i]) for i, layer in enumerate(self.layer0_linear)]
        g = torch.cat(g, dim=0)
        g = self.layer0_act(g)

        # layer1 Conv, u_2, pp_2, a_4
        g = [layer(g, hyperIndex[i], hyperWeight, hyperAttr) for i,layer in enumerate(self.hyper1)]
        g = torch.sum(torch.stack(g, dim=0), dim=0)

        # layer1 act and dropout
        g = self.act1(g)
        # layer1 linear
        g = [
            g[:self.args["users"], :], 
            g[self.args["users"]:self.args["users"]+self.args["phonePlacements"], :], 
            g[self.args["users"]+self.args["phonePlacements"]:, :],
            ]
        g = [layer(g[i]) for i, layer in enumerate(self.layer1_linear)]
        g = torch.cat(g, dim=0)
        g = self.layer1_act(g)

        # layer2 conv
        g = [layer(g, hyperIndex[i], hyperWeight, hyperAttr) for i,layer in enumerate(self.hyper2)]
        g = torch.sum(torch.stack(g, dim=0), dim=0)

        # layer2 act and dropout
        g = self.act2(g)
        
        # map matrices to same dim
        g = [
            g[:self.args["users"], :], 
            g[self.args["users"]:self.args["users"]+self.args["phonePlacements"], :], 
            g[self.args["users"]+self.args["phonePlacements"]:, :],
            ]
        new_g = [self.g2c_act(layer(g[i])) for i, layer in enumerate(self.g2c)]

        return new_g


