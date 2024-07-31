import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class myDataset(Dataset):
    """ dataset reader
    """

    def __init__(self, args, dataType):
        super(myDataset, self).__init__()

        self.hc_feature = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["hcPath"] + '.npy'), dtype=torch.float)
        self.raw_feature = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["rawPath"] + '.npy'), dtype=torch.float)
        self.y = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["yPath"] + '.npy'), dtype=torch.float)
        self.weight = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["weightPath"] + '.npy'), dtype=torch.float)


        # print(self.x.shape, self.y.shape, self.g.shape, self.hyperIndex.shape, self.hyperWeight.shape, self.hyperAttr.shape)
    
    # def getGraph(self):
    #     return [self.g, self.hyperWeight, self.hyperAttr] + self.hyperIndex
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.hc_feature[index], self.raw_feature[index], self.weight[index], self.y[index]


def loadData(args, dataType):
    data = myDataset(args, dataType)
    print("{} size: {}".format(dataType, len(data.y)))

    shuffle = (dataType == "train")
    loader = DataLoader(data, batch_size=args["batch_size"], shuffle=shuffle, num_workers=args["num_workers"])
    return loader
