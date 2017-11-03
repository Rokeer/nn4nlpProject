
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.autograd import Variable
from torch import LongTensor, FloatTensor



def read_train(fname_src):
    global max_length
    lineindex = 0
    with open(fname_src, "r") as f_src:
        for line_src in f_src:
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            [ID, context, question, answer, start, end] = line_src.split('\t')
            max_length = max(max_length,len(context))
            sent_context = [w2i[x] for x in context.strip().split()]
            sent_answers = [w2i[x] for x in answer.strip().split()]
            sent_question = [w2i[x] for x in question.strip().split()]

            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end)
            if lineindex >= 2:
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
                if lineindex >= 2:
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


class BiLSTM(nn.Module):
    def __init__(self,input_size = 100, hidden_size = 100, lstm_layers = 1):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=lstm_layers,bidirectional=True)

    def forward(self, input):
        h_0 = Variable(torch.zeros(lstm_layers, 1, hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(lstm_layers, 1, hidden_size), requires_grad=False)
        outputs, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        return outputs

def loadSentVectors(sentence):
    M = []
    for i in sentence:
        vec = emb_mat[i]
        M.append(vec)
    return M

if __name__ == '__main__':
    # format of files: each line is "word1 word2 ..." aligned line-by-line
    train_src_file = "../data/train_lines"
    dev_src_file = "../data/dev_lines"
    glove_path = "../data/glove/"

    BATCH_SIZE = 20

    # the max length of context in train
    max_length = 0
    word_emb_size = 100
    char_vocab_size = 100
    char_emb_size = 8
    max_num_sents = 10
    EPOCHS = 1

    #LSTM initializer
    input_size =  100
    hidden_size = 100
    lstm_layers = 1

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

    # char_embedding = Embedding(char_vocab_size, char_emb_size)
    #word_embed = Embedding(word_vocab_size, word_emb_size)

    """if use char emb: inputsize = char out size + word emb size else: input size = word emb size"""

    lstm_x = BiLSTM(input_size, hidden_size, lstm_layers)
    lstm_q = BiLSTM(input_size, hidden_size, lstm_layers)

    for epoch in range(0, EPOCHS):
        #need to implement BATCH
        for instance in train:
            x = instance[0]
            q = instance[1]

            Ax = loadSentVectors(x)
            Aq = loadSentVectors(q)

            Ax_tensor = Variable(FloatTensor(Ax))
            Aq_tensor = Variable(FloatTensor(Aq))

            h = lstm_x.forward(Ax_tensor)
            u = lstm_q.forward(Aq_tensor)

            print(h)
            break
            # Attention layer starts