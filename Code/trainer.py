import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as O
import os.path
import numpy as np
import random


class Trainer(object):
    def __init__(self, config, model):
        # assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.optimizer = O.Adadelta(self.model.parameters(),config.init_learningRate)

    def step(self, instance, isTraining, get_summary=False):
        config = self.config
        # get input Sample
        m_start, m_end, start_Logits, end_logits = self.model.forward(instance)
        startLoss = self.model.getLoss(start_Logits, instance[6])
        endLoss = self.model.getLoss(end_logits, instance[7])
        self.optimizer.zero_grad()
        Finalloss = startLoss + endLoss#.backward()
        Finalloss.backward()
        self.optimizer.step()
        return Finalloss.data[0]
