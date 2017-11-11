
from collections import defaultdict
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Embedding
from torch.autograd import Variable
from torch import LongTensor, FloatTensor

from Layers import attentionLayer as atLayer
from Layers import BiModeling, Outputs, selection, BiLSTM, Multi_Conv1D, Conv1D
from ConfigFile import Configuration


class BiDAFModel(nn.Module):
    def __init__(self, config):
        super(BiDAFModel, self).__init__()
        self.Config = config
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
        self.MaxQuestionLength = config.MaxQuestionLength
        self.maxNumberofSentence = config.MaxNumberOfSentences
        self.maxNumberofSentence = 1
        self.maxSentenceLength = config.MaxSentenceLength
        self.use_char_emb = config.use_char_emb
        self.cnn_dropout_keep_prob = config.cnn_dropout_keep_prob
        self.char_vocab_size = config.char_vocab_size
        self.char_emb_size = config.char_emb_size
        self.emb_mat = config.emb_mat
        self.max_word_size = config.max_word_size
        self.padding = config.padding

        self.lstm_x = BiLSTM(self.input_size, self.hidden_size, self.lstm_layers)
        self.lstm_q = BiLSTM(self.input_size, self.hidden_size, self.lstm_layers)
        if self.use_char_emb:
            self.filter_sizes = [100]
            self.heights = [5]
            self.cnn_x = Multi_Conv1D(self.is_train, self.cnn_dropout_keep_prob)
            self.cnn_q = Multi_Conv1D(self.is_train, self.cnn_dropout_keep_prob)
            self.char_embed = Embedding(self.char_vocab_size, self.char_emb_size)

        self.biattention = atLayer(self.maxSentenceLength, self.maxNumberofSentence, self.MaxQuestionLength, self.hidden_size)

        self.m1_bilstm = BiModeling(8 * self.hidden_size, self.hidden_size, self.lstm_layers)  # input_size = 100, hidden_size = 100, lstm_layers = 1 , dropout = 0.2
        self.m2_bilstm = BiModeling(2 * self.hidden_size, self.hidden_size, self.lstm_layers)

        self.o1_output = Outputs(self.is_train, 10 * self.hidden_size, self.outputDropout)  # is_train, input_size = 100, dropout=0.0, output_size=1
        self.o2_bilstm = BiModeling(14 * self.hidden_size, self.hidden_size, self.lstm_layers)
        self.o3_output = Outputs(self.is_train, 10 * self.hidden_size, self.outputDropout)

        self.loss = nn.CrossEntropyLoss()

    def loadSentVectors(self, sentence):
        M = []
        for i in sentence:
            vec = self.emb_mat[i]
            M.append(vec)
        return M

    def padVectors(self, V, max_vector_size, embed_size):
        l = max_vector_size - len(V)
        for i in range(l):
            V.append(embed_size * [0])
        return V

    def forward(self, instance):
        x = instance[0]
        q = instance[1]
        cx = []
        cq = []

        if self.use_char_emb:
            cx = instance[8]
            cq = instance[9]

            cx = self.padVectors(cx, self.maxSentenceLength, self.max_word_size)
            cq = self.padVectors(cq, self.MaxQuestionLength, self.max_word_size)

            Acx = self.char_embed(Variable(LongTensor(cx)))
            Acx = Acx.unsqueeze(0)
            Acq = self.char_embed(Variable(LongTensor(cq)))
            Acq = Acq.unsqueeze(0)

            Cx = self.cnn_x(Acx, self.filter_sizes, self.heights, self.padding)
            Cq = self.cnn_q(Acq, self.filter_sizes, self.heights, self.padding)

            Cx = Cx.permute(0, 2, 1)
            Cq = Cq.permute(0, 2, 1)

        #print(Cx.size())
        #print(Cq.size())

        Ax = self.loadSentVectors(x)
        Aq = self.loadSentVectors(q)
        Ax = np.array(self.padVectors(Ax, self.maxSentenceLength, self.word_emb_size))
        Aq = np.array(self.padVectors(Aq, self.MaxQuestionLength, self.word_emb_size))

        Ax_tensor = Variable(FloatTensor([Ax]))
        Aq_tensor = Variable(FloatTensor([Aq]))

        #print(Ax_tensor.size())
        #print(Aq_tensor.size())

        xx = torch.cat((Ax_tensor, Cx), 2)
        qq = torch.cat((Aq_tensor, Cq), 1)

        xx = xx.unsqueeze(0)
        qq = qq.unsqueeze(0)

        h = self.lstm_x(Ax_tensor)# add dimension for batch
        h = h.unsqueeze(0)
        u = self.lstm_q(Aq_tensor)
#old code:
        '''Ax = np.array(self.loadSentVectors(x, self.maxSentenceLength))
        Aq = np.array(self.loadSentVectors(q, self.maxquestionLength))

        Ax_tensor = Variable(FloatTensor([Ax]))
        Aq_tensor = Variable(FloatTensor([Aq]))



        h = self.lstm_x(Ax_tensor)  # add dimension for batch
        h = h.unsqueeze(0)
        u = self.lstm_q(Aq_tensor)'''

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
        flat_o3 = o3.view(-1, M * JX)
        flat_end = F.softmax(flat_o3)

        start = flat_start.view(-1, M, JX)
        end = flat_end.view(-1, M, JX)

        return start, end, o1, o3

    def getLoss(self,predict, true):
        target = Variable(torch.LongTensor([int(true)]))
        output = self.loss(predict, target)
        return output
        #return -1.0 * math.log(predict.data[0][0][int(true)])
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
