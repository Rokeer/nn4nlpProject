
from collections import defaultdict
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Embedding
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, ByteTensor

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

        self.CNNEmbeddingSize = config.CNNEmbeddingSize

        self.hw_1 = HighwayNetwork(config.numOfHighwayLayers, self.input_size)

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

    def padTensors(self, T, max_l, embed_size):
        l = max_l - T.size()[1]
        if l > 0:
            if usecuda:
                pad_T = Variable(torch.cuda.FloatTensor([[0] * embed_size]*l)).unsqueeze(0)
                #print(T.size(),pad_T.size())
                T = torch.cat((T, pad_T), 1).cuda()
            else:
                pad_T = Variable(torch.zeros(l, embed_size)).unsqueeze(0)
                T = torch.cat((T.type(FloatTensor), pad_T), 1)
        return T

    def getMask(self, instances, config):
        contextMask = [[1 for i in range(len(instance[0]))] + [0 for j in range(config.MaxSentenceLength - len(instance[0]))] for instance in instances]
        questionMask = [[1 for i in range(len(instance[1]))] + [0 for j in range(config.MaxQuestionLength - len(instance[1]))] for instance in instances]

        if usecuda:
            contextMask_Tensor = torch.cuda.ByteTensor(contextMask)
            questionMask_Tensor = torch.cuda.ByteTensor(questionMask)
        else:
            contextMask_Tensor = torch.ByteTensor(contextMask)#,requires_grad=False)
            questionMask_Tensor = torch.ByteTensor(questionMask)#,requires_grad=False)

        contextMask_Tensor = contextMask_Tensor.unsqueeze(1)
        return contextMask_Tensor, questionMask_Tensor

    def forward(self, instances, config):
        if usecuda:
            Context_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxSentenceLength, self.input_size).type(torch.cuda.FloatTensor))
            Query_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxQuestionLength, self.input_size).type(torch.cuda.FloatTensor))
        else:
            Context_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxSentenceLength, self.input_size).type(torch.FloatTensor))
            Query_Char_Word_list = Variable(
                torch.zeros(len(instances), config.MaxQuestionLength, self.input_size).type(torch.FloatTensor))

        contextMask, QuestionMask = self.getMask(instances, config)
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
                    #ContextChar_beforeCNN = Variable(torch.cuda.LongTensor(self.char_embed(ContextChar)))
                else:
                    ContextChar_beforeCNN = self.char_embed(Variable(LongTensor(ContextChar)))

                ContextChar_beforeCNN = ContextChar_beforeCNN.unsqueeze(0)
                if usecuda:
                    QueryChar_beforeCNN = self.char_embed(Variable(torch.cuda.LongTensor(QueryChar)))
                    #QueryChar_beforeCNN = Variable(torch.cuda.LongTensor(self.char_embed(QueryChar)))
                else:
                    QueryChar_beforeCNN = self.char_embed(Variable(LongTensor(QueryChar)))
                QueryChar_beforeCNN = QueryChar_beforeCNN.unsqueeze(0)

                # Output from CNN is a FloatTensor
                ContextChar_CNN = self.cnn_x(ContextChar_beforeCNN, self.filter_sizes, self.heights, self.padding)
                QueryChar_CNN = self.cnn_q(QueryChar_beforeCNN, self.filter_sizes, self.heights, self.padding)

                ContextChar_CNN = ContextChar_CNN.permute(0, 2, 1)
                QueryChar_CNN = QueryChar_CNN.permute(0, 2, 1)

            # pad word tensors from CNN output
           #  ContextChar_CNN_ = self.padTensors(ContextChar_CNN, config.MaxSentenceLength, self.word_emb_size)
            # QueryChar_CNN_ = self.padTensors(QueryChar_CNN, config.MaxQuestionLength, self.word_emb_size)

            # Word Embedding: Load glove vectors for sentence and pad with 0
            ContextWord = self.loadSentVectors(Context)
            QueryWord = self.loadSentVectors(Query)
            ContextWord = np.array(self.padVectors(ContextWord, config.MaxSentenceLength, self.word_emb_size))
            QueryWord = np.array(self.padVectors(QueryWord, config.MaxQuestionLength, self.word_emb_size))

            if usecuda:
                ContextWord_tensor = Variable(torch.cuda.FloatTensor([ContextWord]))
                QueryWord_tensor = Variable(torch.cuda.FloatTensor([QueryWord]))
            else:
                ContextWord_tensor = Variable(FloatTensor([ContextWord]))
                QueryWord_tensor = Variable(FloatTensor([QueryWord]))

            # Concatenate word vectors from word and character embeddings
            if self.use_char_emb:
                ContextChar_CNN_ = self.padTensors(ContextChar_CNN, config.MaxSentenceLength, self.word_emb_size)
                QueryChar_CNN_ = self.padTensors(QueryChar_CNN, config.MaxQuestionLength, self.word_emb_size)
                Context_Char_Word = torch.cat((ContextWord_tensor, ContextChar_CNN_), 2)
                Query_Char_Word = torch.cat((QueryWord_tensor, QueryChar_CNN_), 2)

            else:
                Context_Char_Word = ContextWord_tensor
                Query_Char_Word = QueryWord_tensor
            Context_Char_Word_list[count] = Context_Char_Word
            Query_Char_Word_list[count] = Query_Char_Word
            count = count + 1
        #xx = xx.unsqueeze(0)
        #qq = qq.unsqueeze(0)
        # Context_Char_Word = Variable(LongTensor(Context_Char_Word_list))
        # Query_Char_Word = Variable(LongTensor(Query_Char_Word_list))
        Context_Char_Word = self.hw_1(Context_Char_Word_list, self.is_train)
        Query_Char_Word = self.hw_1(Query_Char_Word_list, self.is_train)

        contextEmbeddingLayerOut = self.lstm_x(Context_Char_Word)# add dimension for batch
        contextEmbeddingLayerOut = contextEmbeddingLayerOut.unsqueeze(1)
        questionEmbeddingLayerOut = self.lstm_q(Query_Char_Word)

        Context_Char_Word = Context_Char_Word.unsqueeze(1)
        Query_Char_Word = Query_Char_Word.unsqueeze(1)

        JX = Context_Char_Word.size()[2]
        M = Context_Char_Word.size()[1]
        N = Context_Char_Word.size()[0]

        attentionOutput = self.biattention(contextEmbeddingLayerOut, questionEmbeddingLayerOut, self.is_train, config, contextMask, QuestionMask)

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

        #Mask Logits before the loss
        contextMask = contextMask.squeeze(1)
        o1.data = torch.add(o1.data, (~contextMask).float() * -1e20)
        o3.data = torch.add(o3.data, (~contextMask).float() * -1e20)

        return start, end, o1, o3

    def getLoss(self,predict, true):
        #target = Variable(torch.LongTensor([int(true)]))
        output = self.loss(predict, true)
        return output