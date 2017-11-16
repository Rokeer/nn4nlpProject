
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
from Layers import BiModeling, Outputs, selection, BiLSTM, Multi_Conv1D, Conv1D, HighwayNetwork
from ConfigFile import Configuration

if torch.cuda.is_available():
    usecuda = True
else:
    usecuda = False
# usecuda = False

class BiDAFModel(nn.Module):
    def __init__(self, config):
        super(BiDAFModel, self).__init__()
        self.Config = config
        self.train_src_file = config.train_src_file  # "../data/train_lines"
        self.dev_src_file = config.dev_src_file  # "../data/dev_lines"
        self.glove_path = config.glove_path  # "../data/glove/"
        self.BATCH_SIZE = config.BatchSize
        self.word_emb_size = config.word_emb_size
        self.input_size = config.word_emb_size + config.CNNEmbeddingSize
        self.hidden_size = config.hidden_size
        self.lstm_layers = config.numOfLSTMLayers
        self.is_train = config.is_train
        self.EPOCHS = config.EPOCHS
        self.outputDropout = config.outputDropout
        self.MaxQuestionLength = config.MaxQuestionLength
        self.maxNumberofSentence = config.MaxNumberOfSentences
        self.maxNumberofSentence = 1
        # self.maxSentenceLength = config.MaxSentenceLength
        self.use_char_emb = config.use_char_emb
        self.cnn_dropout_keep_prob = config.cnn_dropout_keep_prob
        self.char_vocab_size = config.char_vocab_size
        self.char_emb_size = config.char_emb_size
        self.emb_mat = config.emb_mat
        self.max_word_size = config.max_word_size
        self.padding = config.padding

        self.hw_1 = HighwayNetwork(config.numOfHighwayLayers, config.hidden_size)

        self.lstm_x = BiModeling(self.input_size, self.hidden_size, self.lstm_layers)
        self.lstm_q = BiModeling(self.input_size, self.hidden_size, self.lstm_layers)

        # if usecuda:
        #     self.lstm_x = self.lstm_x.cuda()
        #     self.lstm_q = self.lstm_q.cuda()

        if self.use_char_emb:
            self.filter_sizes = [100]
            self.heights = [5]
            self.cnn_x = Multi_Conv1D(self.is_train, self.cnn_dropout_keep_prob)
            self.cnn_q = Multi_Conv1D(self.is_train, self.cnn_dropout_keep_prob)
            self.char_embed = Embedding(self.char_vocab_size, self.char_emb_size)

        self.biattention = atLayer(self.hidden_size)

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

    def forward(self, instances, config):
        if usecuda:
            Context_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxSentenceLength, 2 * self.word_emb_size).type(torch.cuda.FloatTensor))
            Query_Char_Word_list = Variable(
                torch.zeros(len(instances), self.MaxQuestionLength, 2 * self.word_emb_size).type(torch.cuda.FloatTensor))
        else:
            Context_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxSentenceLength, 2 * self.word_emb_size).type(torch.FloatTensor))
            Query_Char_Word_list = Variable(
                torch.zeros(len(instances), self.MaxQuestionLength, 2 * self.word_emb_size).type(torch.FloatTensor))

        count = 0
        # Context_Char_Word_list = []
        # Query_Char_Word_list = []
        for instance in instances:
            Context = instance[0]
            Query = instance[1]
            ContextChar = []
            QueryChar = []

            if self.use_char_emb:
                ContextChar = instance[8]
                QueryChar = instance[9]

                #ContextChar = self.padVectors(ContextChar, config.MaxSentenceLength, self.max_word_size)
                #QueryChar = self.padVectors(QueryChar, self.MaxQuestionLength, self.max_word_size)

                if usecuda:
                    ContextChar_beforeCNN = self.char_embed(Variable(torch.cuda.LongTensor(ContextChar)))
                else:
                    ContextChar_beforeCNN = self.char_embed(Variable(LongTensor(ContextChar)))

                ContextChar_beforeCNN = ContextChar_beforeCNN.unsqueeze(0)
                if usecuda:
                    QueryChar_beforeCNN = self.char_embed(Variable(torch.cuda.LongTensor(QueryChar)))
                else:
                    QueryChar_beforeCNN = self.char_embed(Variable(LongTensor(QueryChar)))
                QueryChar_beforeCNN = QueryChar_beforeCNN.unsqueeze(0)

                ContextChar_CNN = self.cnn_x(ContextChar_beforeCNN, self.filter_sizes, self.heights, self.padding)
                QueryChar_CNN = self.cnn_q(QueryChar_beforeCNN, self.filter_sizes, self.heights, self.padding)

                ContextChar_CNN = ContextChar_CNN.permute(0, 2, 1)
                QueryChar_CNN = QueryChar_CNN.permute(0, 2, 1)


            length =  config.MaxSentenceLength - ContextChar_CNN.size()[1]
            if length > 0:
                if usecuda:
                    ContextCharPadding = Variable(torch.cuda.LongTensor(length, self.word_emb_size).zero_().unsqueeze(0))
                    ContextChar_CNN = torch.cat((ContextChar_CNN.type(LongTensor), ContextCharPadding), 1)
                else:
                    ContextCharPadding = Variable(torch.LongTensor(length, self.word_emb_size).zero_().unsqueeze(0))
                    ContextChar_CNN = torch.cat((ContextChar_CNN.type(LongTensor), ContextCharPadding), 1)

            length = config.MaxQuestionLength - QueryChar_CNN.size()[1]
            if length > 0:
                if usecuda:
                    QueryCharPadding = Variable(torch.cuda.LongTensor(length, self.word_emb_size).zero_().unsqueeze(0))
                    QueryChar_CNN = torch.cat((QueryChar_CNN.type(LongTensor), QueryCharPadding), 1)
                else:
                    QueryCharPadding = Variable(torch.LongTensor(length, self.word_emb_size).zero_().unsqueeze(0))
                    QueryChar_CNN = torch.cat((QueryChar_CNN.type(LongTensor), QueryCharPadding), 1)

            #print(Cx.size())
            #print(Cq.size())

            ContextWord = self.loadSentVectors(Context)
            QueryWord = self.loadSentVectors(Query)
            ContextWord = np.array(self.padVectors(ContextWord, config.MaxSentenceLength, self.word_emb_size))
            QueryWord = np.array(self.padVectors(QueryWord, self.MaxQuestionLength, self.word_emb_size))

            if usecuda:
                ContextWord_tensor = Variable(torch.cuda.FloatTensor([ContextWord]))
                QueryWord_tensor = Variable(torch.cuda.FloatTensor([QueryWord]))
            else:
                ContextWord_tensor = Variable(FloatTensor([ContextWord]))
                QueryWord_tensor = Variable(FloatTensor([QueryWord]))

            #print(Ax_tensor.size())
            #print(Aq_tensor.size())

            Context_Char_Word = torch.cat((ContextWord_tensor, ContextChar_CNN), 2)
            Query_Char_Word = torch.cat((QueryWord_tensor, QueryChar_CNN), 2)
            Context_Char_Word_list[count] = Context_Char_Word
            Query_Char_Word_list[count] = Query_Char_Word
            count = count + 1
        #xx = xx.unsqueeze(0)
        #qq = qq.unsqueeze(0)
        # Context_Char_Word = Variable(LongTensor(Context_Char_Word_list))
        # Query_Char_Word = Variable(LongTensor(Query_Char_Word_list))
        Context_Char_Word = self.hw_1(Context_Char_Word_list, self.is_train)
        Query_Char_Word = self.hw_1(Query_Char_Word_list, self.is_train)

        h = self.lstm_x(Context_Char_Word)# add dimension for batch
        h = h.unsqueeze(1)
        u = self.lstm_q(Query_Char_Word)

        Context_Char_Word = Context_Char_Word.unsqueeze(1)
        Query_Char_Word = Query_Char_Word.unsqueeze(1)

        JX = Context_Char_Word.size()[2]
        M = Context_Char_Word.size()[1]
        N = Context_Char_Word.size()[0]

        attentionOutput = self.biattention(h, u, self.is_train, config)

        m1 = self.m1_bilstm(attentionOutput.view(N, M * JX, -1))
        m2 = self.m2_bilstm(m1.contiguous().view(N, M * JX, -1))
        # m1 = self.m1_bilstm(attentionOutput)
        # m2 = self.m2_bilstm(m1)

        o1 = self.o1_output((m2.contiguous(), attentionOutput))  # Colin: what is x_mask

        a1i = selection(m2.view(N, M * JX, 2 * self.hidden_size), o1.view(N, M * JX))
        a1i = a1i.unsqueeze(1).unsqueeze(1).repeat(1, M, JX, 1)

        m2 = m2.view(N, M, JX, -1)
        o2 = self.o2_bilstm(torch.cat([attentionOutput, m2, a1i, m2 * a1i], 3).squeeze(1))  # we removed the number of sentences
        o3 = self.o3_output((o2.contiguous(), attentionOutput))

        flat_o1 = o1.view(-1, M * JX)
        flat_start = F.softmax(flat_o1)
        flat_o3 = o3.view(-1, M * JX)
        flat_end = F.softmax(flat_o3)

        start = flat_start.view(-1, M, JX)
        end = flat_end.view(-1, M, JX)

        return start, end, o1, o3

    def getLoss(self,predict, true):
        #target = Variable(torch.LongTensor([int(true)]))
        output = self.loss(predict, true)
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
