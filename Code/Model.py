
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

max_length = 0
max_Query_Length = 0

def read_train(fname_src):
    global max_length
    global max_Query_Length
    lineindex = 0
    with open(fname_src, "r") as f_src:
        for line_src in f_src:
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            [ID, context, question, answer, start, end] = line_src.split('\t')
            sent_context = [w2i[x] for x in context.strip().split()]
            sent_answers = [w2i[x] for x in answer.strip().split()]
            sent_question = [w2i[x] for x in question.strip().split()]
            max_length = max(max_length, len(sent_context))
            max_Query_Length = max(max_Query_Length,len(sent_question))
            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end)
            if lineindex >= 1000:
                break

def read_dev(fname_src):
    global max_length
    lineindex = 0
    question = context = ''
    answers, sent_answers, sent_context, sent_question = ([] for i in range(4))
    with open(fname_src, "r") as f_src:
        for line_src in f_src:
            line_src = line_src.replace('\n', '').replace('\r', '').strip()
            if line_src == "":
                continue
            lineindex += 1
            if context == line_src.split('\t')[1] and question == line_src.split('\t')[2]:
                answer = line_src.split('\t')[3]
                answers.append(answer)
                sent_answers.append([w2i[x] for x in answer.strip().split()])
            else:
                if context != '' or question != '':
                    yield (sent_context, sent_question, sent_answers, context, question, answers, start, end)
                context = line_src.split('\t')[1]
                max_length = max(max_length, len(context))

                question = line_src.split('\t')[2]
                start = int(line_src.split('\t')[4])
                end = int(line_src.split('\t')[5])
                sent_context = [w2i[x] for x in context.strip().split()]
                sent_question = [w2i[x] for x in question.strip().split()]
                answers = []
                sent_answers = []

                answer = line_src.split('\t')[3]
                answers.append(answer)
                sent_answers.append([w2i[x] for x in answer.strip().split()])
                if lineindex >= 1000:
                    break

# This function uses the global w2i dictionary and gets the glove vector for each word as a dictionary
def get_GLOVE_word2vec():
    glove_file = "{}glove.6B.{}d.txt".format(glove_path, word_emb_size)
    total = int(4e5) #6B
    word2vec_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as gf:
        for line in gf:
            emb = line.lstrip().rstrip().split(" ")
            word = emb[0]
            vector = list(map(float, emb[1:]))
            if word in w2i:
                word2vec_dict[word] = vector
            elif word.capitalize() in w2i:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in w2i:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in w2i:
                word2vec_dict[word.upper()] = vector

    return word2vec_dict


def loadSentVectors(sentence):
    M = []
    for i in sentence:
        vec = emb_mat[i]
        M.append(vec)
    return M

def calLoss(predict, true):
    return -1.0 * math.log(predict[true])


if __name__ == '__main__':
    ModelConfiguration = Configuration()
    # format of files: each line is "word1 word2 ..." aligned line-by-line
    train_src_file = ModelConfiguration.train_src_file#"../data/train_lines"
    dev_src_file = ModelConfiguration.dev_src_file #"../data/dev_lines"
    glove_path = ModelConfiguration.glove_path# "../data/glove/"
    BATCH_SIZE = ModelConfiguration.BatchSize
    word_emb_size = ModelConfiguration.word_emb_size
    input_size = ModelConfiguration.word_emb_size
    hidden_size = ModelConfiguration.hidden_size
    lstm_layers = ModelConfiguration.numOfLSTMLayers
    is_train = ModelConfiguration.is_train
    EPOCHS = ModelConfiguration.EPOCHS
    outputDropout = ModelConfiguration.outputDropout
    maxSentenceLength = ModelConfiguration.MaxSentenceLength
    maxquestionLength = ModelConfiguration.MaxQuestionLength
    maxNumberofSentence = ModelConfiguration.MaxNumberOfSentences
    maxNumberofSentence = 1
    maxquestionLength = max_Query_Length
    maxSentenceLength = max_length
    #global max_length
    #global max_Query_Length
    # the max length of context in train
    #max_length = 0
    #max_Query_Length = 0
    #word_emb_size = 100
    #char_vocab_size = 100
    #char_emb_size = 8
    #max_num_sents = 10
    #EPOCHS = 1

    #LSTM initializer
    #input_size =  100
    #hidden_size = 100
    #lstm_layers = 1
    #is_train = True
    #outputDropout = 0.2

    # Initialize dictionary
    w2i = defaultdict(lambda: len(w2i))

    train = list(read_train(train_src_file))
    unk_src = w2i["<unk>"]
    w2i = defaultdict(lambda: unk_src, w2i)
    word_vocab_size = len(w2i)
    dev = list(read_dev(dev_src_file))

    word2vec_dict = get_GLOVE_word2vec()
    widx2vec_dict = {w2i[word]: vec for word, vec in word2vec_dict.items() if word in w2i}
    emb_mat = np.array([widx2vec_dict[wid] if wid in widx2vec_dict
                        else np.random.multivariate_normal(np.zeros(word_emb_size), np.eye(word_emb_size))
                        for wid in range(word_vocab_size)])


    maxquestionLength = max_Query_Length
    maxSentenceLength = max_length

    # char_embedding = Embedding(char_vocab_size, char_emb_size)
    #word_embed = Embedding(word_vocab_size, word_emb_size)

    """if use char emb: inputsize = char out size + word emb size else: input size = word emb size"""

    lstm_x = BiLSTM(input_size, hidden_size, lstm_layers)
    lstm_q = BiLSTM(input_size, hidden_size, lstm_layers)
    biattention = atLayer(maxSentenceLength, maxNumberofSentence, maxquestionLength,hidden_size)

    m1_bilstm = BiModeling(8 * hidden_size, hidden_size, lstm_layers)  # input_size = 100, hidden_size = 100, lstm_layers = 1 , dropout = 0.2
    m2_bilstm = BiModeling(2 * hidden_size, hidden_size, lstm_layers)  # Colin: why 8 times and 2 times

    o1_output = Outputs(is_train, 10 * hidden_size, outputDropout)  # is_train, input_size = 100, dropout=0.0, output_size=1
    o2_bilstm = BiModeling(14 * hidden_size, hidden_size, lstm_layers)
    o3_output = Outputs(is_train, 10 * hidden_size, outputDropout)

    for epoch in range(0, EPOCHS):
        loss = 0
        #need to implement BATCH
        for instance in train:
            x = instance[0]
            q = instance[1]

            Ax = np.array(loadSentVectors(x))
            Aq = np.array(loadSentVectors(q))

            Ax_tensor = Variable(FloatTensor([Ax]))
            Aq_tensor = Variable(FloatTensor([Aq]))
            p2d = (2,100)
            F.pad(Ax_tensor,p2d,mode = 'constant', value = 0)


            h = lstm_x(Ax_tensor) # add dimension for batch
            h = h.unsqueeze(0)
            u = lstm_q(Aq_tensor)

            Ax_tensor = Ax_tensor.unsqueeze(0)
            Aq_tensor = Aq_tensor.unsqueeze(0)

            JX = Ax_tensor.size()[2]
            M = Ax_tensor.size()[1]
            N = Ax_tensor.size()[0]

            #print(h.size())
            #print(u.size())
            attentionOutput = biattention(h,u, True)
            #print(attentionOutput.size())

            m1 = m1_bilstm(attentionOutput.view(N, M * JX, -1))
            m2 = m2_bilstm(m1.view(N, M * JX, -1))

            o1 = o1_output((m2, attentionOutput))  # Colin: what is x_mask

            a1i = selection(m2.view(N, M * JX, 2 * hidden_size), o1.view(N, M * JX))
            a1i = a1i.unsqueeze(1).unsqueeze(1).repeat(1, M, JX, 1)

            m2 = m2.view(N, M, JX, -1)
            o2 = o2_bilstm(torch.cat([attentionOutput, m2, a1i, m2 * a1i], 3).squeeze())
            o3 = o3_output((o2, attentionOutput))

            flat_o1 = o1.view(-1, M * JX)
            flat_start = F.softmax(flat_o1)
            flat_o2 = o2.view(-1, M * JX)
            flat_end = F.softmax(flat_o2)

            start = flat_start.view(-1, M, JX)
            end = flat_end.view(-1, M, JX)

            loss += calLoss(start, instance[6])
            loss += calLoss(end, instance[7])


            break
            # Attention layer starts

        loss /= len(train)
        print(loss)
        loss.backward()
