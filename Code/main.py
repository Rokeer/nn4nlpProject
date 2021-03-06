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
import os
import pdb
import codecs
import pickle
import evaluate
import json
import cloudpickle


if torch.cuda.is_available():
    usecuda = True
else:
    usecuda = False

LoadFiles = False
reverse = True
if usecuda:
    print("Using Cuda")

def read_train(configuration):
    max_length = 0
    max_Query_Length = 0
    lineindex = 0
    # , encoding='utf-8'
    with codecs.open(config.train_src_file, "r",encoding='utf-8') as f_src:

        for line_src in f_src:
            cx = []
            cq = []
            #line_src = line_src.decode("utf-8")
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            [ID, context, question, answer, start, end] = line_src.split('\t')
            # if ID != '5726a975708984140094cd38':
            #     continue
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


            sent_question = [w2i[x] for x in question.strip().split()]
            for q in question.strip().split():
                for c in q:
                    char_counter[c] += 1
                cqi = [c2i[c] for c in list(w)]
                cq.append((cqi + config.max_word_size * [0])[:config.max_word_size])
            max_length = max(max_length, len(sent_context))
            max_Query_Length = max(max_Query_Length,len(sent_question))
            try:
                int_val = int(start)
                int_val = int(end)
            except ValueError:
                # pdb.set_trace()
                print("Failure w/ value " + ID)
            if int(end) >= len(sent_context):
                print ("Wrong:" + ID)
            if ID == '56cec3e8aab44d1400b88a02':
                continue
            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end, cx, cq, ID)
            # if lineindex >= 100:
            #     break
    config.MaxSentenceLength = max_length
    config.MaxQuestionLength = max_Query_Length


def read_dev(configuration):
    #max_length = 0
    #max_Query_Length = 0
    lineindex = 0
    # , encoding='utf-8'
    with codecs.open(config.dev_src_file, "r",encoding='utf-8') as f_src:

        for line_src in f_src:
            cx = []
            cq = []
            #line_src = line_src.decode("utf-8")
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex += 1
            [ID, context, question, answer, start, end] = line_src.split('\t')
            # if ID != '5726a975708984140094cd38':
            #     continue
            sent_context = [w2i[x] for x in context.strip().split()]
            for w in context.strip().split():
                cxi = [c2i[c] for c in list(w)]
                cx.append((cxi + config.max_word_size * [0])[:config.max_word_size])

            sent_answers = [w2i[x] for x in answer.strip().split()]


            sent_question = [w2i[x] for x in question.strip().split()]
            for q in question.strip().split():
                cqi = [c2i[c] for c in list(w)]
                cq.append((cqi + config.max_word_size * [0])[:config.max_word_size])
            #max_length = max(max_length, len(sent_context))
            #max_Query_Length = max(max_Query_Length,len(sent_question))
            try:
                int_val = int(start)
                int_val = int(end)
            except ValueError:
                # pdb.set_trace()
                print("Failure w/ value " + ID)
            if int(end) >= len(sent_context):
                print ("Wrong:" + ID)
            if ID == '56cec3e8aab44d1400b88a02':
                continue
            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end, cx, cq, ID)
            # if lineindex >= 50:
            #     break
    #config.MaxSentenceLength = max_length
    #config.MaxQuestionLength = max_Query_Length



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


# if LoadFiles == True:
#     ######################## Loading Everything###############################33
#     with open('../objects/emb_mat', 'rb') as f:
#         emb_mat = pickle.load(f)
#     with open('../objects/w2i', 'rb') as f:
#         w2i = pickle.load(f)
#     with open('../objects/c2i', 'rb') as f:
#         c2i = pickle.load(f)
#     with open('../objects/dev', 'rb') as f:
#         dev = pickle.load(f)
#     with open('./objects/train', 'rb') as f:
#         train = pickle.load(f)
#     with open('../objects/word2vec_dict', 'rb') as f:
#         word2vec_dict = pickle.load(f)
#     with open('../objects/widx2vec_dict', 'rb') as f:
#         widx2vec_dict = pickle.load(f)
#     with open('../objects/config', 'rb') as f:
#         config = pickle.load(f)

#else:
config = Configuration()
w2i = defaultdict(lambda: len(w2i))
c2i = defaultdict(lambda: len(c2i) + 1)
char_counter = Counter()


train = list(read_train(config))
if reverse:
    train.sort(key=lambda x: len(x[0]), reverse=reverse)
else:
    train.sort(key=lambda x: len(x[0]))

# train = train[0:0 + config.BatchSize]
print(str(config.MaxSentenceLength))
unk_src = w2i["<unk>"]
w2i = defaultdict(lambda: unk_src, w2i)
unk_char_src = c2i["<unk>"]
c2i = defaultdict(lambda: unk_char_src, c2i)

word_vocab_size = len(w2i)
print(len(c2i))
config.char_vocab_size = len(c2i) + 1
dev = list(read_dev(config.dev_src_file))
if reverse:
    dev.sort(key=lambda x: len(x[0]), reverse=reverse)
else:
    dev.sort(key=lambda x: len(x[0]))

word2vec_dict = get_GLOVE_word2vec(config.glove_path, config.GloveEmbeddingSize)
widx2vec_dict = {w2i[word]: vec for word, vec in word2vec_dict.items() if word in w2i}

if LoadFiles == True:
    with open('../objects/emb_mat', 'rb') as f:
        emb_mat = pickle.load(f)
else:
    emb_mat = np.array([widx2vec_dict[wid] if wid in widx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for wid in range(word_vocab_size)])
config.emb_mat = emb_mat


    ######################## Saving Everything so far###############################
    # with open('../objects/emb_mat', 'wb') as f:
    #     cloudpickle.dump(emb_mat, f)
    # with open('../objects/config', 'wb') as f:
    #    cloudpickle.dump(config, f)
    # with open('../objects/dev', 'wb') as f:
    #     cloudpickle.dump(dev, f)
    # with open('../objects/train', 'wb') as f:
    #     cloudpickle.dump(train, f)
    # with open('../objects/w2i', 'wb') as f:
    #     cloudpickle.dump(w2i, f)
    # with open('../objects/c2i', 'wb') as f:
    #     cloudpickle.dump(c2i, f)
    # with open('../objects/word2vec_dict', 'wb') as f:
    #     cloudpickle.dump(word2vec_dict, f)
    # with open('../objects/widx2vec_dict', 'wb') as f:
    #     cloudpickle.dump(widx2vec_dict, f)


BiDAF_Model = BiDAFModel(config)
if os.path.isfile('../models/model_nov19.pkl'):
    BiDAF_Model.load_state_dict(torch.load('../models/model_nov19.pkl', map_location=lambda storage, loc: storage))
    print('Loading model...')
if usecuda:
    BiDAF_Model.cuda()
else:
    BiDAF_Model.cpu()

BiDAFTrainer = Trainer(config,BiDAF_Model)

for epoch in range(1, config.EPOCHS):
    ####################################################Train#########################################3
    # writeResult = codecs.open(str(epoch) + 'prediction_results.txt', 'w', encoding='utf-8')
    config.is_train = True
    BiDAF_Model.train()
    loss = 0
    numOfSamples = 0
    numOfBatch = 0
    start = time.time()

    if epoch % 4 == 0:
        dict = {}
    # for instance in train:
    print("Start Training:" + str(epoch))
    for sid in range(0, len(train), config.BatchSize):
        instances = train[sid:sid + config.BatchSize]
        if reverse:
            config.MaxSentenceLength = len(instances[0][0])
        else:
            config.MaxSentenceLength = len(instances[len(instances)-1][0])
        config.MaxQuestionLength = max([len(instance[1]) for instance in instances])
        # print(config.MaxSentenceLength)

        if epoch % 4 == 0:
            sampleLoss,answers = BiDAFTrainer.step(instances, config, config.is_train, isSearching = True)
            for i in range(len(instances)):
                text = ''
                cnt = instances[i][3].split()
                for j in range(answers[i][0], answers[i][1] + 1):
                    text += cnt[j] + " "
                dict[instances[i][10]] = text.strip()
                dict[instances[i][10]] = text
        else:
            sampleLoss = BiDAFTrainer.step(instances, config, config.is_train)

        sampleLoss = sampleLoss * len(instances)
        loss += sampleLoss

        numOfBatch += 1
        numOfSamples += len(instances)
        if numOfBatch % 1000 == 0:
            end = time.time()
            print(str(epoch) + " , " + str(numOfSamples) + ' / ' + str(len(train)) + " , Current loss : " + str(
                loss / numOfSamples)+", run time = " + str(end - start))
            start = time.time()

    loss /= numOfSamples
    print(str(loss))
    torch.save(BiDAF_Model.state_dict(), '../models/'+str(epoch)+'_nov19.pkl')
    with open('../objects/emb_mat', 'wb') as f:
        cloudpickle.dump(emb_mat, f)
    if epoch % 4 == 0:
        with codecs.open(str(epoch) + '_train_result.txt', 'w', encoding='utf-8') as outfile:
            json.dump(dict, outfile)

    #############################DEV##############################3
    # Start Dev
    if epoch % 3 == 0:
        dict = {}
        config.is_train = False
        BiDAF_Model.eval()
        loss = 0
        numOfSamples = 0
        numOfBatch = 0
        start = time.time()
        print("Start Dev:")
        for sid in range(0, len(dev), config.DevBatchSize):

            instances = dev[sid:sid + config.DevBatchSize]
            #print(instances[0][10])
            if reverse:
                config.MaxSentenceLength = len(instances[0][0])
            else:
                config.MaxSentenceLength = len(instances[len(instances) - 1][0])
            config.MaxQuestionLength = max([len(instance[1]) for instance in instances])
            # print(config.MaxSentenceLength)
            sampleLoss, answers = BiDAFTrainer.step(instances, config, config.is_train, isSearching = True)
            for i in range(len(instances)):
                text = ''
                cnt = instances[i][3].split()
                for j in range(answers[i][0], answers[i][1] + 1):
                    text += cnt[j] + " "
                dict[instances[i][10]] = text.strip()
                dict[instances[i][10]] = text
            sampleLoss = sampleLoss * len(instances)
            loss += sampleLoss
            numOfBatch += 1
            numOfSamples += len(instances)
            if numOfSamples % 5000 == 0:
                end = time.time()
                print("Dev: " + str(numOfSamples) + ' / ' + str(len(dev)) + " , Current loss : " + str(
                    loss / numOfSamples) + ", run time = " + str(end - start))
                start = time.time()

        with codecs.open(str(epoch) + '_Dev_result.txt', 'w', encoding='utf-8') as outfile:
            json.dump(dict, outfile)
        loss /= numOfSamples
        print(str(epoch) + ' Dev Loss: ' + str(loss))


