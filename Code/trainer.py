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
        self.optimizer = O.Adadelta(config.init_lr)

    def step(self, instance, isTraining, get_summary=False):
        config = self.config
        # get input Sample
        m_start, m_end = self.model(instance)
        startLoss = self.model.getLoss(m_start)
        endLoss = self.model.getLoss(m_end)
        self.optimizer.zero_grad()
        startLoss.backward()
        endLoss.backward()
        self.optimizer.step()
        return startLoss + endLoss
