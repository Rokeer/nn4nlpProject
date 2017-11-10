
from collections import defaultdict
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Embedding
from torch.autograd import Variable
from torch import LongTensor, FloatTensor

from Code.Layers import attentionLayer as atLayer
from Code.Layers import BiModeling
from Code.Layers import Outputs
from Code.Layers import selection
from Code.ConfigFile import Configuration
from Code.Layers import BiLSTM


class BiDAFModel(nn.Module):
    def __init__(self, config):
        self.train_src_file = config.train_src_file  # "../data/train_lines"
        self.dev_src_file = config.dev_src_file  # "../data/dev_lines"
        self.glove_path = config.glove_path  # "../data/glove/"
        self.BATCH_SIZE = config.BatchSize
        self.word_emb_size = config.word_emb_size
        self.input_size = config.word_emb_size
        self.hidden_size = config.hidden_size
        self.lstm_layers = config.numOfLSTMLayers
        self.is_train = config.is_train
        self.EPOCHS = config.EPOCHS
        self.outputDropout = config.outputDropout
        self.maxSentenceLength = config.MaxSentenceLength
        self.maxquestionLength = config.MaxQuestionLength
        self.maxNumberofSentence = config.MaxNumberOfSentences
        self.maxNumberofSentence = 1
        self.maxquestionLength = config.max_Query_Length
        self.maxSentenceLength = config.max_length
        self.loss = nn.CrossEntropyLoss()

        lstm_x = BiLSTM(self.input_size, self.hidden_size, self.lstm_layers)
        lstm_q = BiLSTM(self.input_size, self.hidden_size, self.lstm_layers)
        biattention = atLayer(self.maxSentenceLength, self.maxNumberofSentence, self.maxquestionLength, self.hidden_size)

        m1_bilstm = BiModeling(8 * self.hidden_size, self.hidden_size, self.lstm_layers)  # input_size = 100, hidden_size = 100, lstm_layers = 1 , dropout = 0.2
        m2_bilstm = BiModeling(2 * self.hidden_size, self.hidden_size, self.self.lstm_layers)

        o1_output = Outputs(self.is_train, 10 * self.hidden_size, self.outputDropout)  # is_train, input_size = 100, dropout=0.0, output_size=1
        o2_bilstm = BiModeling(14 * self.hidden_size, self.hidden_size, self.lstm_layers)
        o3_output = Outputs(self.is_train, 10 * self.hidden_size, self.outputDropout)

    def forward(self, instance):
        x = instance[0]
        q = instance[1]

        Ax = np.array(loadSentVectors(x, self.maxSentenceLength))
        Aq = np.array(loadSentVectors(q, self.maxquestionLength))

        Ax_tensor = Variable(FloatTensor([Ax]))
        Aq_tensor = Variable(FloatTensor([Aq]))

        h = self.lstm_x(Ax_tensor)  # add dimension for batch
        h = h.unsqueeze(0)
        u = self.lstm_q(Aq_tensor)

        Ax_tensor = Ax_tensor.unsqueeze(0)
        Aq_tensor = Aq_tensor.unsqueeze(0)

        JX = Ax_tensor.size()[2]
        M = Ax_tensor.size()[1]
        N = Ax_tensor.size()[0]

        attentionOutput = self.biattention(h, u, self.is_train)

        m1 = self.m1_bilstm(attentionOutput.view(N, M * JX, -1))
        m2 = self.m2_bilstm(m1.view(N, M * JX, -1))

        o1 = self.o1_output((m2, attentionOutput))  # Colin: what is x_mask

        a1i = selection(m2.view(N, M * JX, 2 * self.hidden_size), o1.view(N, M * JX))
        a1i = a1i.unsqueeze(1).unsqueeze(1).repeat(1, M, JX, 1)

        m2 = m2.view(N, M, JX, -1)
        o2 = self.o2_bilstm(torch.cat([attentionOutput, m2, a1i, m2 * a1i], 3).squeeze(1))  # we removed the number of sentences
        o3 = self.o3_output((o2, attentionOutput))

        flat_o1 = o1.view(-1, M * JX)
        flat_start = F.softmax(flat_o1)
        flat_o2 = o2.view(-1, M * JX)
        flat_end = F.softmax(flat_o2)

        start = flat_start.view(-1, M, JX)
        end = flat_end.view(-1, M, JX)

        return start, end

    def getLoss(self,predict, true):
        #input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        #target = autograd.Variable(torch.LongTensor(3).random_(5))
        #output = loss(input, target)
        output = self.loss(predict, true)
        return output
        return -1.0 * math.log(predict.data[0][0][int(true)])
#
# if __name__ == '__main__':
#     ModelConfiguration = Configuration()
#     # format of files: each line is "word1 word2 ..." aligned line-by-line
#     train_src_file = ModelConfiguration.train_src_file#"../data/train_lines"
#     dev_src_file = ModelConfiguration.dev_src_file #"../data/dev_lines"
#     glove_path = ModelConfiguration.glove_path# "../data/glove/"
#     BATCH_SIZE = ModelConfiguration.BatchSize
#     word_emb_size = ModelConfiguration.word_emb_size
#     input_size = ModelConfiguration.word_emb_size
#     hidden_size = ModelConfiguration.hidden_size
#     lstm_layers = ModelConfiguration.numOfLSTMLayers
#     is_train = ModelConfiguration.is_train
#     EPOCHS = ModelConfiguration.EPOCHS
#     outputDropout = ModelConfiguration.outputDropout
#     maxSentenceLength = ModelConfiguration.MaxSentenceLength
#     maxquestionLength = ModelConfiguration.MaxQuestionLength
#     maxNumberofSentence = ModelConfiguration.MaxNumberOfSentences
#     maxNumberofSentence = 1
#     maxquestionLength = max_Query_Length
#     maxSentenceLength = max_length
#     #global max_length
#     #global max_Query_Length
#     # the max length of context in train
#     #max_length = 0
#     #max_Query_Length = 0
#     #word_emb_size = 100
#     #char_vocab_size = 100
#     #char_emb_size = 8
#     #max_num_sents = 10
#     #EPOCHS = 1
#
#     #LSTM initializer
#     #input_size =  100
#     #hidden_size = 100
#     #lstm_layers = 1
#     #is_train = True
#     #outputDropout = 0.2
#
#     # Initialize dictionary
#     w2i = defaultdict(lambda: len(w2i))
#
#     train = list(read_train(train_src_file))
#     unk_src = w2i["<unk>"]
#     w2i = defaultdict(lambda: unk_src, w2i)
#     word_vocab_size = len(w2i)
#     dev = list(read_dev(dev_src_file))
#
#     word2vec_dict = get_GLOVE_word2vec()
#     widx2vec_dict = {w2i[word]: vec for word, vec in word2vec_dict.items() if word in w2i}
#     emb_mat = np.array([widx2vec_dict[wid] if wid in widx2vec_dict
#                         else np.random.multivariate_normal(np.zeros(word_emb_size), np.eye(word_emb_size))
#                         for wid in range(word_vocab_size)])
#
#
#     maxquestionLength = max_Query_Length
#     maxSentenceLength = max_length
#
#     # char_embedding = Embedding(char_vocab_size, char_emb_size)
#     #word_embed = Embedding(word_vocab_size, word_emb_size)
#
#     """if use char emb: inputsize = char out size + word emb size else: input size = word emb size"""
#
#
#
#
