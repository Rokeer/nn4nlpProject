from collections import defaultdict, Counter
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from ConfigFile import Configuration
from Model import BiDAFModel
from trainer import Trainer
import time

def read_train(configuration):
    max_length = 0
    max_Query_Length = 0
    lineindex = 0
    with open(config.train_src_file, "r") as f_src:
        for line_src in f_src:
            cx = []
            cq = []
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            [ID, context, question, answer, start, end] = line_src.split('\t')

            sent_context = [w2i[x] for x in context.strip().split()]
            for w in context.strip().split():
                for c in w:
                    char_counter[c] += 1
                cxi = [c2i[c] for c in list(w)]
                cx.append((cxi + config.max_word_size * [0])[:config.max_word_size])

            sent_answers = [w2i[x] for x in answer.strip().split()]
            for w in answer.strip().split():
                for c in w:
                    char_counter[c] += 1
                cqi = [c2i[c] for c in list(w)]
                cq.append((cqi + config.max_word_size * [0])[:config.max_word_size])

            sent_question = [w2i[x] for x in question.strip().split()]
            for q in question.strip().split():
                for c in q:
                    char_counter[c] += 1
            max_length = max(max_length, len(sent_context))
            max_Query_Length = max(max_Query_Length,len(sent_question))
            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end, cx, cq)
            # if lineindex >= 20:
            #     break
    config.MaxSentenceLength = max_length
    config.MaxQuestionLength = max_Query_Length

def read_dev(fname_src):
    #global max_length
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
                #max_length = max(max_length, len(context))

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
                if lineindex >= 100:
                    break

# This function uses the global w2i dictionary and gets the glove vector for each word as a dictionary
def get_GLOVE_word2vec(glove_path, word_emb_size):
    glove_file = "{}glove.6B.{}d.txt".format(glove_path, word_emb_size)
    total = int(4e5) #6B
    word2vec_dict = {}
    with open(glove_file, 'r') as gf:
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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


config = Configuration()
w2i = defaultdict(lambda: len(w2i))
c2i = defaultdict(lambda: len(c2i))
char_counter = Counter()

unk_char_src = c2i["<unk>"]
c2i = defaultdict(lambda: unk_char_src, c2i)

train = list(read_train(config))
unk_src = w2i["<unk>"]
w2i = defaultdict(lambda: unk_src, w2i)
word_vocab_size = len(w2i)
dev = list(read_dev(config.dev_src_file))

word2vec_dict = get_GLOVE_word2vec(config.glove_path, config.GloveEmbeddingSize)
widx2vec_dict = {w2i[word]: vec for word, vec in word2vec_dict.items() if word in w2i}
emb_mat = np.array([widx2vec_dict[wid] if wid in widx2vec_dict
                    else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                    for wid in range(word_vocab_size)])
config.emb_mat = emb_mat

BiDAF_Model = BiDAFModel(config)
BiDAFTrainer = Trainer(config,BiDAF_Model)

for epoch in range(0, config.EPOCHS):
    loss = 0
    # need to implement BATCH
    numOfSamples = 0
    start = time.time()
    for instance in train:
        # for sid in range(0, len(train), config.BatchSize):
        #     instances = train[sid:sid + config.BatchSize]

        sampleLoss = BiDAFTrainer.step(instance, config.is_train)
        loss += sampleLoss
        numOfSamples+=1
        if numOfSamples%2 == 0:
            print (str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(train)) + " , Current loss : "+str(loss / numOfSamples))
            # end = time.time()
            # print("run time = " + str(end - start))
            # start = time.time()
            print('%s (%d %d%%) %.4f' % (timeSince(start, numOfSamples / (len(train) * 1.0)),
                                         numOfSamples, numOfSamples / len(train) * 100, loss / numOfSamples))

    loss /= len(train)
    print(loss)
    torch.save(BiDAF_Model.state_dict(), '../models/epoch.pkl')