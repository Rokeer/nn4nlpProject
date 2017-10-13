from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np

# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/train_lines"
dev_src_file = "../data/dev_lines"

w2i = defaultdict(lambda: len(w2i))

SentRepdictionary = {}
max_length = 0
def read(fname_src):
    global max_length
    lineindex = 0
    with open(fname_src, "r") as f_src:
        for line_src in f_src:
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            context = line_src.split('\t')[1]
            if len(context) > max_length:
                max_length = len(context)
            question = line_src.split('\t')[2]
            answer = line_src.split('\t')[3]
            start = int(line_src.split('\t')[4])
            end = int(line_src.split('\t')[5])
            sent_context = [w2i[x] for x in context.strip().split()]
            sent_answers = [w2i[x] for x in answer.strip().split()]
            sent_question = [w2i[x] for x in question.strip().split()]
            yield (sent_context, sent_question, sent_answers, context, question, answer, start, end)
            if lineindex >= 2000:
                break

# Read the data
train = list(read(train_src_file))
unk_src = w2i["<unk>"]
w2i = defaultdict(lambda: unk_src, w2i)
nwords_src = len(w2i)
dev = list(read(dev_src_file))
print (max_length)
# DyNet Starts
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 512

# Lookup parameters for word embeddings
LOOKUP_SRC = model.add_lookup_parameters((nwords_src, EMBED_SIZE))

# Word-level BiLSTMs
LSTM_context = dy.BiRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model, dy.GRUBuilder)
LSTM_answer = dy.BiRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model, dy.GRUBuilder)
LSTM_question = dy.BiRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model, dy.GRUBuilder)

# Word-level softmax
W_sm_start = model.add_parameters((max_length+1, 2*HIDDEN_SIZE))
b_sm_start = model.add_parameters(max_length+1)

W_sm_end = model.add_parameters((max_length+1, 2*HIDDEN_SIZE))
b_sm_end = model.add_parameters(max_length+1)

# A function to calculate scores for one value
def calc_scores(sent):
    dy.renew_cg()

    # context = sent[3]
    # question = sent[4]
    # answer = sent[5]
    # if context in SentRepdictionary.keys():
    #    contextReps = SentRepdictionary[context]
    # else:
    contextReps = LSTM_context.transduce([LOOKUP_SRC[x] for x in sent[0]])[-1]
    #    SentRepdictionary[context] = contextReps

    # if question in SentRepdictionary.keys():
    #    questionReps = SentRepdictionary[question]
    # else:
    questionReps = LSTM_question.transduce([LOOKUP_SRC[x] for x in sent[1]])[-1]
    #    SentRepdictionary[question] = questionReps

    # if answer in SentRepdictionary.keys():
    #    answerReps = SentRepdictionary[answer]
    # else:
    #answerReps = LSTM_answer.transduce([LOOKUP_SRC[x] for x in sent[2]])[-1]
    #    SentRepdictionary[answer] = answerReps

    W_sm_exp_start = dy.parameter(W_sm_start)
    b_sm_exp_start = dy.parameter(b_sm_start)
    W_sm_exp_end = dy.parameter(W_sm_end)
    b_sm_exp_end = dy.parameter(b_sm_end)
    return [dy.affine_transform([b_sm_exp_start, W_sm_exp_start, dy.concatenate([contextReps, questionReps])]),
            dy.affine_transform([b_sm_exp_end, W_sm_exp_end, dy.concatenate([contextReps, questionReps])])]
    #return W_sm_exp * dy.concatenate([contextReps, questionReps]) + b_sm_exp

# Calculate loss for one mini-batch
def calc_start_loss(sents):
    dy.renew_cg()
    # Transduce all batch elements with an LSTM
    sent_reps = [(LSTM_context.transduce([LOOKUP_SRC[x] for x in sent[0]])[-1],
                  LSTM_question.transduce([LOOKUP_SRC[z] for z in sent[1]])[-1],
                  LSTM_answer.transduce([LOOKUP_SRC[y] for y in sent[2]])[-1]) for sent  in sents]

    W_sm_exp_start = dy.parameter(W_sm_start)
    b_sm_exp_start = dy.parameter(b_sm_start)

    loss_start = [dy.affine_transform([b_sm_exp_start, W_sm_exp_start, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    start_loss = [dy.pickneglogsoftmax(x, y[6]) for x,y in zip(loss_start,sents)]
     #dy.sum_batches(dy.esum(losses))
    return dy.esum(start_loss)

def calc_end_loss(sents):
    dy.renew_cg()
    # Transduce all batch elements with an LSTM
    sent_reps = [(LSTM_context.transduce([LOOKUP_SRC[x] for x in sent[0]])[-1],
                  LSTM_question.transduce([LOOKUP_SRC[z] for z in sent[1]])[-1],
                  LSTM_answer.transduce([LOOKUP_SRC[y] for y in sent[2]])[-1]) for sent  in sents]

    W_sm_exp_end = dy.parameter(W_sm_end)
    b_sm_exp_end = dy.parameter(b_sm_end)

    loss_end = [dy.affine_transform([b_sm_exp_end, W_sm_exp_end, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    end_loss = [dy.pickneglogsoftmax(x, y[7]) for x, y in zip(loss_end, sents)]
    #dy.sum_batches(dy.esum(losses))
    return dy.esum(end_loss)



# for ITER in range(5):
#     # Perform training
#     random.shuffle(train)
#     train_loss = 0.0
#     start = time.time()
#     sentIndex = 0
#     for sent in train:
#         my_loss = dy.pickneglogsoftmax(calc_scores(sent), sent[2][0])
#         train_loss += my_loss.value()
#         my_loss.backward()
#         trainer.update()
#         sentIndex+=1
#         if sentIndex % 10 == 0:
#             print('Train ' + str(sentIndex) + ' / ' + str(len(train)) + ' ,Iter ' + str(ITER))
#     print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
#     # Perform training
#     sentIndex = 0
#     test_correct = 0.0
#     for sent in dev:
#         scores = calc_scores(sent).npvalue()
#         predict = np.argmax(scores)
#         if predict == sent[2][0]:
#             test_correct += 1
#         sentIndex+=1
#         if sentIndex % 10 == 0:
#             print('Dev ' + str(sentIndex) + ' / ' + str(len(dev)) + ' ,Iter ' + str(ITER))
#     print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))

start = time.time()
train_mbs = all_time = dev_time = all_tagged = this_sents = this_loss = 0
for ITER in range(2):
    train_loss = 0
    random.shuffle(train)
    for sid in range(0, len(train), BATCH_SIZE):
        my_size = min(BATCH_SIZE, len(train)-sid)
        train_mbs += 1
        if train_mbs % int(1000/BATCH_SIZE) == 0:
            trainer.status()
            #print("loss/sent=%.4f, sent/sec=%.4f" % (this_loss / this_sents, (train_mbs * BATCH_SIZE) / (time.time() - start - dev_time)), file=sys.stderr)
            this_loss = this_sents = 0
        # train on the minibatch
        loss_exp = calc_start_loss(train[sid:sid+BATCH_SIZE])
        this_loss += loss_exp.scalar_value()
        train_loss += loss_exp.scalar_value()
        loss_exp = calc_end_loss(train[sid:sid + BATCH_SIZE])
        this_loss += loss_exp.scalar_value()
        train_loss += loss_exp.scalar_value()
        this_sents += BATCH_SIZE
        loss_exp.backward()
        trainer.update()
        print('Train ' + str(sid) + ' / ' + str(len(train)) + ' ,Iter ' + str(ITER))
    #print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
    sentIndex = 0
    test_correct = 0.0
    for sent in dev:
        scores = calc_scores(sent)
        scores_start = scores[0].npvalue()
        scores_end = scores[1].npvalue()
        predict_start = np.argmax(scores_start)
        print (predict_start)
        print (np.argmax(scores_end))
        if predict_start == sent[6]:
            test_correct += 1
        sentIndex +=1
        if sentIndex % 500 == 0:
            print('Dev ' + str(sentIndex) + ' / ' + str(len(dev)) + ' ,Iter ' + str(ITER))
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))
    end = time.time()
    print("run time %.4f" % end - start)
