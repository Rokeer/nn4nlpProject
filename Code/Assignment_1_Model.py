from __future__ import print_function
import time
import readJSON
from collections import Counter
from collections import defaultdict

import random
import math
import sys
import argparse
import string
import re

import dynet as dy
import numpy as np

# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/train_lines"
dev_src_file = "../data/dev_lines"

readJSON.ConvertJSON('../data/train-v1.1.json', train_src_file)
readJSON.ConvertJSON('../data/dev-v1.1.json', dev_src_file)

w2i = defaultdict(lambda: len(w2i))

SentRepdictionary = {}
max_length = 0

def read_train(fname_src):
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
            #if lineindex >= 2000:
            #    break

def read_dev(fname_src):
    global max_length
    lineindex = 0
    question = context = ''
    answers = []
    sent_answers = []
    sent_context = []
    sent_question = []
    with open(fname_src, "r") as f_src:
        for line_src in f_src:
            line_src = line_src.replace('\n','').replace('\r','').strip()
            if line_src == "":
                continue
            lineindex+=1
            if context == line_src.split('\t')[1] and question == line_src.split('\t')[2]:
                answer = line_src.split('\t')[3]
                answers.append(answer)
                sent_answers.append([w2i[x] for x in answer.strip().split()])
            else:
                if context != '' or question != '':
                    yield (sent_context, sent_question, sent_answers, context, question, answers, start, end)
                context = line_src.split('\t')[1]
                if len(context) > max_length:
                    max_length = len(context)
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
            #if lineindex >= 50:
            #    break


# Read the data
train = list(read_train(train_src_file))
unk_src = w2i["<unk>"]
w2i = defaultdict(lambda: unk_src, w2i)
nwords_src = len(w2i)
dev = list(read_dev(dev_src_file))
# print (max_length)
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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(qas):
    f1 = exact_match = total = 0
    for i in range(len(qas)):
        total += 1
        ground_truths = qas[i][0]
        prediction = qas[i][1]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths) 

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return [exact_match, f1]

# A function to calculate scores for one value
def calc_scores(sent):
    dy.renew_cg()

    contextReps = LSTM_context.transduce([LOOKUP_SRC[x] for x in sent[0]])[-1]
    questionReps = LSTM_question.transduce([LOOKUP_SRC[x] for x in sent[1]])[-1]

    W_sm_exp_start = dy.parameter(W_sm_start)
    b_sm_exp_start = dy.parameter(b_sm_start)
    W_sm_exp_end = dy.parameter(W_sm_end)
    b_sm_exp_end = dy.parameter(b_sm_end)

    loss_start = dy.affine_transform([b_sm_exp_start, W_sm_exp_start, dy.concatenate([contextReps, questionReps])])
    loss_end = dy.affine_transform([b_sm_exp_end, W_sm_exp_end, dy.concatenate([contextReps, questionReps])])

    return [dy.log_softmax(loss_start), dy.log_softmax(loss_end)]
    #return W_sm_exp * dy.concatenate([contextReps, questionReps]) + b_sm_exp

# Calculate loss for one mini-batch
def getRepresentation(sents):
    dy.renew_cg()
    # Transduce all batch elements with an LSTM
    sent_reps = [(LSTM_context.transduce([LOOKUP_SRC[x] for x in sent[0]])[-1],
                  LSTM_question.transduce([LOOKUP_SRC[z] for z in sent[1]])[-1],
                  LSTM_answer.transduce([LOOKUP_SRC[y] for y in sent[2]])[-1]) for sent in sents]
    return sent_reps

def calc_loss(sent_reps, sents):


    W_sm_exp_start = dy.parameter(W_sm_start)
    b_sm_exp_start = dy.parameter(b_sm_start)

    loss_start = [dy.affine_transform([b_sm_exp_start, W_sm_exp_start, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    start_loss = [dy.pickneglogsoftmax(x, y[6]) for x,y in zip(loss_start,sents)]
     #dy.sum_batches(dy.esum(losses))


    W_sm_exp_end = dy.parameter(W_sm_end)
    b_sm_exp_end = dy.parameter(b_sm_end)

    loss_end = [dy.affine_transform([b_sm_exp_end, W_sm_exp_end, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    end_loss = [dy.pickneglogsoftmax(x, y[7]) for x, y in zip(loss_end, sents)]

    return dy.esum(start_loss + end_loss)

def calc_start_loss(sent_reps, sents):


    W_sm_exp_start = dy.parameter(W_sm_start)
    b_sm_exp_start = dy.parameter(b_sm_start)

    loss_start = [dy.affine_transform([b_sm_exp_start, W_sm_exp_start, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    start_loss = [dy.pickneglogsoftmax(x, y[6]) for x,y in zip(loss_start,sents)]
     #dy.sum_batches(dy.esum(losses))
    return dy.esum(start_loss)

def calc_end_loss(sent_reps, sents):

    W_sm_exp_end = dy.parameter(W_sm_end)
    b_sm_exp_end = dy.parameter(b_sm_end)

    loss_end = [dy.affine_transform([b_sm_exp_end, W_sm_exp_end, dy.concatenate([x[0], x[1]])]) for x in sent_reps]

    #loss = W_sm_exp * mtxCombined + b_sm_exp
    end_loss = [dy.pickneglogsoftmax(x, y[7]) for x, y in zip(loss_end, sents)]
    #dy.sum_batches(dy.esum(losses))
    return dy.esum(end_loss)



start = time.time()
train_mbs = all_time = dev_time = all_tagged = this_sents = this_loss = 0
for ITER in range(1,201):
    start_itr = time.time()
    train_loss = 0
    random.shuffle(train)
    for sid in range(0, len(train), BATCH_SIZE):
        my_size = min(BATCH_SIZE, len(train)-sid)
        train_mbs += 1
        if train_mbs % int(10240/BATCH_SIZE) == 0:
            trainer.status()
            print('Train ' + str(sid) + ' / ' + str(len(train)) + ' ,Iter ' + str(ITER) + ', loss/sent=' + str(
                this_loss / this_sents))
            #print("loss/sent=%.4f, sent/sec=%.4f" % (this_loss / this_sents, (train_mbs * BATCH_SIZE) / (time.time() - start - dev_time)), file=sys.stderr)
            this_loss = this_sents = 0

        # train on the minibatch
        sents = train[sid:sid+BATCH_SIZE]
        reps = getRepresentation(sents)
        flag = False
        if flag:
            loss_exp = calc_loss(reps, sents)
            this_loss += loss_exp.scalar_value()
            train_loss += loss_exp.scalar_value()
            this_sents += BATCH_SIZE
            loss_exp.backward()
            trainer.update()
        else:
            loss_Start = calc_start_loss(reps, sents)
            this_loss += loss_Start.scalar_value()/float(2)
            train_loss += loss_Start.scalar_value()/float(2)
            loss_Start.backward()
            trainer.update()

            loss_End = calc_end_loss(reps, sents)
            this_loss += loss_End.scalar_value()/float(2)
            train_loss += loss_End.scalar_value()/float(2)
            this_sents += BATCH_SIZE
            loss_End.backward()
            trainer.update()

    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start_itr))



    if ITER % 10 == 0:
        #model.save('a.model')
        start_itr = time.time()
        sentIndex = 0
        test_correct = 0.0
        qas = []
        for sent in dev:
            scores = calc_scores(sent)
            scores_start = scores[0].npvalue()
            scores_end = scores[1].npvalue()

            predict_start = 0
            predict_end = 1
            max_score = scores_start[0] + scores_end[1]
            for i in range(len(sent[3])):
                for j in range(i + 1, len(sent[3]) + 1):
                    if scores_start[i] + scores_end[j] > max_score:
                        max_score = scores_start[i] + scores_end[j]
                        predict_start = i
                        predict_end = j
            # print (predict_start)
            # print (predict_end)
            answer = ''
            for i in range(predict_start, predict_end):
                answer = answer + str(sent[3][i])
            qas.append([sent[5], answer])
            # print (answer)

            # if predict_start == sent[6]:
            #    test_correct += 1
            # sentIndex +=1
            # if sentIndex % 500 == 0:
            #    print('Dev ' + str(sentIndex) + ' / ' + str(len(dev)) + ' ,Iter ' + str(ITER))
        result = evaluate(qas)
        print("iter %r: test total=%r, EM=%.4f, F1=%.4f time=%.2fs" % (
            ITER, len(dev), result[0], result[1], time.time() - start_itr))
        model.save('../models/' + str(ITER) + '.model')
end = time.time()
print("run time = " + str(end - start))
        

