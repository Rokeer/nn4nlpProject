import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as O
import os.path
import numpy as np
import random
from torch import LongTensor, FloatTensor
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, config, model):
        # assert isinstance(model, Model)
        # self.config = config
        self.model = model
        self.optimizer = O.Adadelta(self.model.parameters(),config.init_learningRate)

    def step(self, instances, config, isTraining, get_summary=False):
        # config = self.config
        # get input Sample
        m_start, m_end, start_Logits, end_logits = self.model.forward(instances, config)

        startVlas = Variable(torch.LongTensor([int(i[6]) for i in instances]))
        startLoss = self.model.getLoss(start_Logits, startVlas)
        endVlas = Variable(torch.LongTensor([int(i[7]) for i in instances]))
        endLoss = self.model.getLoss(end_logits, endVlas)
        self.optimizer.zero_grad()
        Finalloss = startLoss + endLoss#.backward()
        Finalloss.backward()
        self.optimizer.step()
        return Finalloss.data[0]
