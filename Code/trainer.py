import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as O
import os.path
import numpy as np
import random
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
# import pdb

if torch.cuda.is_available():
    usecuda = True
else:
    usecuda = False
# usecuda = False
class Trainer(object):
    def __init__(self, config, model):
        # assert isinstance(model, Model)
        # self.config = config
        self.model = model
        self.optimizer = O.Adadelta(self.model.parameters(),config.init_learningRate)

    def beamSearch(self, m_start, m_end, mask, size, instances):
        m_start = m_start * mask
        m_end = m_end * mask
        m_start, m_start_index = m_start.sort(dim=1,descending=True)
        # m_end, m_end_index = m_end.sort(dim=1, descending=True)
        m_start = m_start.squeeze(2)
        m_start = m_start.squeeze(2)
        m_end = m_end.squeeze(2)
        m_start = m_start[:,:size]
        m_start_index = m_start_index[:, :size]
        answers = []
        for i in range(len(m_start[0])):
            max_start_ind = 0
            max_end_ind = 0
            maxprob = 0
            for m in range(size):
                for n in range(m_start_index[i][m],len(m_end[0])):
                    if m_start[i][m] * m_end[i][n] > maxprob:
                        maxprob = m_start[i][m] * m_end[i][n] > maxprob
                        max_start_ind = m_start_index[i][m]
                        max_end_ind = m_start_index[i][n]

            start, end = self.findWordPosition(instances[i][3], max_start_ind, max_end_ind)
            answers.append([start, end])
        return answers

    def findWordPosition(self, text, max_start_ind, max_end_ind):
        ttext = text.split()
        startWord = ttext[max_start_ind]
        endWord = ttext[max_end_ind]

        tmp = text.split(startWord)
        for i in range(len(tmp)):
            if i == 0:
                before = tmp[0]
            else:
                before = before + startWord + tmp[i]
            if before.split() == max_start_ind:
                start = len(before)-1
                break

        tmp = text.split(endWord)
        for i in range(len(tmp)):
            if i == 0:
                before = tmp[0]
            else:
                before = before + endWord + tmp[i]
            if before.split() == max_end_ind:
                end = len(before+endWord)-1
                break
        return start, end


    def step(self, instances, config, isSearching=False, get_summary=False):
        # config = self.config
        # get input Sample
        m_start, m_end, start_Logits, end_logits, mask = self.model.forward(instances, config)
        # pdb.set_trace()
        if usecuda:
            startVlas = Variable(torch.cuda.LongTensor([int(i[6]) for i in instances]))
        else:
            startVlas = Variable(torch.LongTensor([int(i[6]) for i in instances]))
        # pdb.set_trace()
        startLoss = self.model.getLoss(start_Logits, startVlas)
        if usecuda:
            endVlas = Variable(torch.cuda.LongTensor([int(i[7]) for i in instances]))
        else:
            endVlas = Variable(torch.LongTensor([int(i[7]) for i in instances]))
        endLoss = self.model.getLoss(end_logits, endVlas)
        self.optimizer.zero_grad()
        Finalloss = startLoss + endLoss#.backward()
        if config.is_train == True:
            Finalloss.backward()
            self.optimizer.step()
        if isSearching:
            answers = self.beamSearch(m_start, m_end, mask, config.searchSize)
            return Finalloss.data[0], answers

        return Finalloss.data[0]






