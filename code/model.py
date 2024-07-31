import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import HypergraphConv
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

    def loss(self, pred, label, weight, x):
        return self.loss_bce(pred, label, weight) + self.loss_supervised_contrast(x, label)

    def forward(self, hc_feature, raw_feature):
        x, x_ = self.InstanceEncoder(hc_feature, raw_feature)

        result = [layer(x[i]) for i, layer in enumerate(self.i2t)]
        if self.args["predict_user"]:
            result = torch.cat(result, axis=1)
        else:
            result = torch.cat(result[1:], axis=1)
        x = torch.cat(x, dim=0)
        return result, x_

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

