import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from functools import reduce
from operator import mul
#import code

def selection(target, logits):
    flat_logits = flatten(logits, 1)
    flat_out = F.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)
    out = out.unsqueeze(len(out.size())).mul(target).sum(len(target.size())-2)
    return out

def flatten(tensor, keep):
    fixed_shape = list(tensor.size())
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] for i in range(start, len(fixed_shape))]
    flat = tensor.view(out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = list(ref.size())
    tensor_shape = list(tensor.size())
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    output = tensor.view(target_shape)
    return output

EMBED_SIZE = 64
HIDDEN_SIZE = 128
#BATCH_SIZE = 512

contextLength = 50
queryLength = 10

ContextMatrix = Variable(torch.rand(1,1,contextLength, 2*EMBED_SIZE))
QueryMatrix = Variable(torch.rand(1,queryLength, 2*EMBED_SIZE))

class BiModeling(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 100, lstm_layers = 1 , dropout = 0.2):
        super(BiModeling, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, dropout=dropout, bidirectional=True)

    def forward(self, inputs):
        seq_len, batch_size, feature_size = inputs.size()
        h_0 = Variable(torch.zeros(2, batch_size, 100), requires_grad=False)
        c_0 = Variable(torch.zeros(2, batch_size, 100), requires_grad=False)
        outputs, (h_n, c_n) = self.bilstm(inputs, (h_0, c_0))
        return outputs

def linear_logits(args, is_train, dropout, linear, mask):
    flatedargs = [F.dropout(flatten(arg, 1), training=is_train, p=dropout) for arg in args]
    flated = linear(torch.cat(flatedargs, 1))
    out = reconstruct(flated, args[0], 1)
    output = out.squeeze(len(list(args[0].size())) - 1)
    if mask is not None:
        if torch.cuda.is_available():
            output = torch.add(output.data, (torch.ones(mask.size()) - mask.type(torch.cuda.FloatTensor)) * 1e30 * -1)
        else:
            output = torch.add(output.data, (torch.ones(mask.size()) - mask.type(torch.FloatTensor)) * 1e30 * -1)
    return output


class Outputs(nn.Module):
    def __init__(self, is_train, input_size = 100, dropout=0.2, output_size=1):
        super(Outputs, self).__init__()

        self.dropout = dropout
        self.is_train = is_train
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, args, mask=None):
        outputs = linear_logits(args, self.is_train, self.dropout, self.linear, mask)

        return outputs



# h is context Vector with length t
# u is query vector with length j

class attentionLayer(nn.Module):
    def __init__(self, contextLength, numofSentences, QueryLength, embedSize):
        super(attentionLayer, self).__init__()

        self.S_weightVector = torch.nn.Linear(6 * embedSize, 1)
        self.sentenceLength = contextLength # sentence length in context
        self.num_sentences = numofSentences # number of sentences in context
        self.questionLength = QueryLength # question length

    def forward(self, h_vector, u_vector, is_train):
        # Add new dimension and repeat h vector multiple times in that dimension to ease the multiplication with the query.
        h_vector_expanded = h_vector.unsqueeze(3).repeat(1, 1, 1, self.questionLength, 1)
        # Add two dimensions one for sentences and one sentence length and repeat u vector multiple times in these dimension.
        u_vector_expanded = u_vector.unsqueeze(1).unsqueeze(1).repeat(1, self.num_sentences, self.sentenceLength, 1, 1)

        h_u_mul = h_vector_expanded * u_vector_expanded
        allVectors = [h_vector_expanded, u_vector_expanded, h_u_mul]

        flat_vectors = [F.dropout(flatten(arg, 1), training=is_train) for arg in allVectors]
        flat_outs = self.S_weightVector(torch.cat(flat_vectors, 1))
        out = reconstruct(flat_outs, allVectors[0], 1)
        sValues = out.squeeze(len(list(allVectors[0].size())) - 1)

        # query attention
        flat_sValues = flatten(sValues, 1)
        flat_out = F.softmax(flat_sValues)
        out = reconstruct(flat_out, sValues, 1)
        u_vector_attention = out.unsqueeze(len(out.size())).mul(u_vector_expanded).sum(len(u_vector_expanded.size()) - 2)

        # Context attention
        max_sValues = torch.max(sValues, 3)[0]
        flat_max_sValues = flatten(max_sValues, 1)
        flat_out = F.softmax(flat_max_sValues)
        out = reconstruct(flat_out, max_sValues, 1)
        h_vector_attention = out.unsqueeze(len(out.size())).mul(h_vector).sum(len(h_vector.size()) - 2)
        h_vector_attention = h_vector_attention.unsqueeze(2).repeat(1, 1, self.sentenceLength, 1)

        attentionLayerOutput = torch.cat([h_vector, u_vector_attention, h_vector * u_vector_attention, h_vector * h_vector_attention], 3)

        return attentionLayerOutput
        #if not config.c2q_att:
        #    u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        #if config.q2c_att:
        #    p0 = tf.concat(3, [h, u_a, h * u_a, h * h_a])
        #else:
        #    p0 = tf.concat(3, [h, u_a, h * u_a])

class HighwayLayer(nn.Module):
    def __init__(self, size):
        super(HighwayLayer, self).__init__()

        self.reluNonlinearity = torch.nn.ReLU(inplace=True)
        self.sigmoidNonlinearity = F.sigmoid
        self.LinearTransform = torch.nn.Linear(size, size)
        self.gate_LinearTransform = torch.nn.Linear(size, size)
        # self.gate_lin.bias.data.fill_(bias_init)

    def getLinearTransofrmation(self, input, linearModel, is_train):
        if linearModel == 'Linear':
            flat_vectors = [F.dropout(flatten(arg, 1), training=is_train) for arg in input]
            flat_outs = self.LinearTransform(torch.cat(flat_vectors, 1))
            out = reconstruct(flat_outs, input[0], 1)
            finalValue = out.squeeze(len(list(input[0].size())) - 1)
            return finalValue
        elif linearModel == 'gate':
            flat_vectors = [F.dropout(flatten(arg, 1), training=is_train) for arg in input]
            flat_outs = self.gate_LinearTransform(torch.cat(flat_vectors, 1))
            out = reconstruct(flat_outs, input[0], 1)
            finalValue = out.squeeze(len(list(input[0].size())) - 1)
            return finalValue

    def forward(self, inputVector, is_train):
        transformation = self.getLinearTransofrmation([inputVector], 'Linear', is_train)
        transformation = self.reluNonlinearity(transformation)

        gate = self.getLinearTransofrmation([inputVector], 'gate', is_train)
        gate = self.sigmoidNonlinearity(gate)
        return torch.add(torch.mul(gate, transformation), torch.mul((1 - gate), inputVector))

class HighwayNetwork(nn.Module):
    def __init__(self, _numLayers, inputSize):
        super(HighwayNetwork, self).__init__()
        self.numLayers = _numLayers
        self.layers = [HighwayLayer(inputSize) for i in range(_numLayers)]

    def forward(self, input, isTrain):
        currentValue = input
        for index in range(self.numLayers):
            # getLinearTransofrmation([currentValue], )
            output = self.layers[index](currentValue, isTrain)
            # output = highway_layer(currentValue, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
            #                        input_keep_prob=input_keep_prob, is_train=is_train)

            currentValue = output
        return currentValue

class BiLSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, lstm_layers=1):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True)

    def forward(self, input):
        h_0 = Variable(torch.zeros(2, 1, hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(2, 1, hidden_size), requires_grad=False)
        outputs, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        return outputs
#aLayer = attentionLayer(contextLength, 1, queryLength)
#attentionLayerOutput = aLayer.forward(ContextMatrix, QueryMatrix, is_train=True)
#print ('Done')