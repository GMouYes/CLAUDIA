import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from pprint import pprint

from model import DecisionMaking
from model import save_model
from util import *
from data import loadData
# import warnings filter
from warnings import simplefilter
from torch.nn import utils
simplefilter(action='ignore', category=UserWarning)


class Trainer(object):
    """Trainer."""

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'
        self.bestResult = np.inf

    def train(self, network, train_data, dev_data=None):
        network = network.to(self.device)
        train_loss, valid_loss = [], []
        validator = Tester(self.args)
        self.optimizer = torch.optim.RAdam(network.parameters(), lr=self.args["lr"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args["gamma"])
        for epoch in range(1, self.args["epoch"] + 1):
            network.train()
            train_epoch_loss = self._train_step(train_data, network, epoch=epoch)
            train_loss.append(train_epoch_loss)
            print('[Trainer] loss: {}'.format(train_epoch_loss))

            # validation
            test_epoch_loss = validator.test(network, dev_data, epoch=epoch)
            valid_loss.append(test_epoch_loss)
            if self.best_eval_result(test_epoch_loss):
                save_model(network, self.args)
                self.args["best_epoch"] = epoch
                save_config(self.args, self.args["outputPath"] + '/' + "best_config.yml")
            self.scheduler.step()
        self.plot_loss(train_loss, valid_loss)

        return True

    def plot_loss(self, train_loss, valid_loss):

        plt.figure()
        ax = plt.subplot(121)
        ax.set_title('train loss')
        ax.plot(train_loss, 'r-')

        ax = plt.subplot(122)
        ax.set_title('validation loss')
        ax.plot(valid_loss, 'b-')

        plt.savefig('{}/{}_{}'.format(self.args["outputPath"], self.args["lr"], self.args["lossPath"]))
        plt.close()

        return True

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        # train_acc = 0.
        loss_record = 0.
        # graphData = data_iterator.dataset.getGraph()
        # graphData = [item.to(self.device) for item in graphData]
        for data in tqdm(data_iterator, desc="Train epoch {}".format(kwargs["epoch"])):
            hc_feature = data[0].to(self.device)
            raw_feature = data[1].to(self.device)
            # x = [item.to(self.device) for item in data[:1]+graphData]
            label = data[3].to(self.device)
            weight = data[2].to(self.device)

            self.optimizer.zero_grad()
            pred, x = network(hc_feature, raw_feature)
            loss = network.loss(pred, label, weight,  x)
            loss_record += loss.item()
            loss.backward()
            utils.clip_grad_norm_(network.parameters(), self.args["clip_grad"])
            self.optimizer.step()

            # rewrite this one
            # train_acc += metric(label.detach().cpu(), pred.detach().cpu(), self.args)

        return loss_record / len(data_iterator.dataset)

    def best_eval_result(self, test_loss):
        """Check if the current epoch yields better validation results.

        :param test_acc, a floating number
        :return: bool, True means current results on dev set is the best.
        """

        if test_loss < self.bestResult:
            self.bestResult = test_loss
            return True
        return False


class Tester(object):
    """Tester."""

    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'

    def test(self, network, dev_data, **kwargs):
        network = network.to(self.device)
        network.eval()

        # test_acc = 0.0
        valid_loss = 0.0
        # graphData = dev_data.dataset.getGraph()
        # graphData = [item.to(self.device) for item in graphData]

        for data in tqdm(dev_data, desc="Test epoch {}".format(kwargs["epoch"])):
            hc_feature = data[0].to(self.device)
            raw_feature = data[1].to(self.device)
            label = data[3].to(self.device)
            weight = data[2].to(self.device)

            with torch.no_grad():
                pred, x = network(hc_feature, raw_feature)
                loss = network.loss(pred, label, weight, x)
                valid_loss += loss.item()

            # rewrite this one
            # test_acc += metric(label.detach().cpu(), pred.detach().cpu(), self.args)

        # Compute the average acc and loss over all test instances
        # test_acc = test_acc / len(dev_data.dataset) * self.args.batch_size
        valid_loss /= len(dev_data.dataset)
        print("[Tester] Loss: {}".format(valid_loss))

        return valid_loss


class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'

    def predict(self, network, test_data):
        network = network.to(self.device)
        network.eval()

        pred_list, truthList, mask_list = [], [], []
        # graphData = test_data.dataset.getGraph()
        # graphData = [item.to(self.device) for item in graphData]
        test_acc = 0.0

        for data in tqdm(test_data, desc="Final Infer:"):
            hc_feature = data[0].to(self.device)
            raw_feature = data[1].to(self.device)
            label = data[3].to(self.device)
            weight = data[2].to(self.device)

            with torch.no_grad():
                pred, _ = network(hc_feature, raw_feature)

            # rewrite
            # test_acc += metric(label.detach().cpu(), pred.detach().cpu(), self.args)
            pred_list.append(pred.detach().cpu())
            truthList.append(label.detach().cpu())
            mask_list.append(weight.detach().cpu())

        pred_list = torch.cat(pred_list, axis=0)
        truthList = torch.cat(truthList, axis=0)
        mask_list = torch.cat(mask_list, axis=0)

        # print(pred_list.shape, truthList.shape)
        # Compute the average acc and loss over all test instances
        if self.args["mask"]:
            test_result = metric(truthList, pred_list, self.args, mask_list)
        else:
            test_result = metric(truthList, pred_list, self.args)

        metricNames = ['BA', 'MCC', 'MacroF1']
        metricResult = dict(zip(metricNames, test_result))
        with open(self.args["outputPath"] + "/" + "metric_result.pkl", "wb") as f:
            pickle.dump(metricResult, f)

        users, phonePlacements, activity = self.args["users"], self.args["phonePlacements"], self.args["activities"]
        for names, result in zip(metricNames, test_result):
            print(names)
            print("[Final tester]: {:.4f}".format(np.nanmean(result)))
            print("Total: {:.4f}".format(np.nanmean(result[1:])))
            print("User: {:.4f}".format(np.nanmean(result[:1])))
            print("PP: {:.4f}".format(np.nanmean(result[1:1+phonePlacements])))
            print("Activity: {:.4f}".format(np.nanmean(result[-activity:])))
            print(result)
            print("")
        return pred_list

def model(args):
    pprint(args)

    if not os.path.isdir(args["outputPath"]):
        os.mkdir(args["outputPath"])

    seed_all(args["seed"])
    trainData, validData, testData = [loadData(args, dataType) for dataType in ['train', 'valid', 'test']]
    model = DecisionMaking(args)

    trainer = Trainer(args)
    trainer.train(model, trainData, validData)

    model.load_state_dict(torch.load(args["outputPath"] + '/' + args["modelPath"]))
    predictor = Predictor(args)
    pred = predictor.predict(model, testData)
    np.save(args["outputPath"] + '/' + args["resultPath"], pred)
    #np.save(args["outputPath"] + '/' + 'nodes_' + args["resultPath"], g)
    # with open(args.outputPath + args.expName + '_' + args.resultPath, 'wb') as f:
    #     pickle.dump(pred, f)

def checkInput(args):
    # placeholder
    return True

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--expName', type=str, default='debug')
    parser.add_argument('--outputPath', type=str, default='../output/')
    parser.add_argument('--lossPath', type=str, default='loss.pdf')

    args = parser.parse_args()
    cfgs = read_config(args.config_path)
    args = {**vars(args), **cfgs}

    if checkInput(args):
        model(args)
    else:
        print('bug in parser!')
    return

if __name__ == '__main__':
    main()
