import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from operator import mul
#import code

if torch.cuda.is_available():
    usecuda = True
else:
    usecuda = False
# usecuda = False

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
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
    def forward(self, inputs):
        batch_size, seq_len, feature_size = inputs.size()
        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size), requires_grad=False)
        outputs, (h_n, c_n) = self.bilstm(inputs, (h_0, c_0))
        return outputs

def linear_logits(args, is_train, dropout, linear, mask):
    flatedargs = [F.dropout(flatten(arg, 1), training=is_train, p=dropout) for arg in args]
    flated = linear(torch.cat(flatedargs, 1))
    out = reconstruct(flated, args[0], 1)
    output = out.squeeze(len(list(args[0].size())) - 1)
    if mask is not None:
        if usecuda:
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
    def __init__(self, embedSize):
        super(attentionLayer, self).__init__()

        self.S_weightVector = torch.nn.Linear(6 * embedSize, 1)
        # self.sentenceLength = contextLength # sentence length in context
        # self.num_sentences = numofSentences # number of sentences in context
        # self.questionLength = QueryLength # question length

    def forward(self, h_vector, u_vector, is_train, config):
        # Add new dimension and repeat h vector multiple times in that dimension to ease the multiplication with the query.
        h_vector_expanded = h_vector.unsqueeze(3).repeat(1, 1, 1, config.MaxQuestionLength, 1)
        # Add two dimensions one for sentences and one sentence length and repeat u vector multiple times in these dimension.
        u_vector_expanded = u_vector.unsqueeze(1).unsqueeze(1).repeat(1, config.MaxNumberOfSentences, config.MaxSentenceLength, 1, 1)

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
        h_vector_attention = h_vector_attention.unsqueeze(2).repeat(1, 1, config.MaxSentenceLength, 1)

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
            flat_outs = self.LinearTransform(torch.cat(flat_vectors, 1).cuda())
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
        self.hidden_size = hidden_size

    def forward(self, input):
        h_0 = Variable(torch.zeros(2, 1, self.hidden_size), requires_grad=False)
        c_0 = Variable(torch.zeros(2, 1, self.hidden_size), requires_grad=False)
        outputs, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        return outputs
#aLayer = attentionLayer(contextLength, 1, queryLength)
#attentionLayerOutput = aLayer.forward(ContextMatrix, QueryMatrix, is_train=True)
#print ('Done')

# Covolutional NN for char based word embedding
class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_height, filter_width, is_train=None, keep_prob=1.0, padding=0):
        super(Conv1D, self).__init__()

        self.is_train = is_train
        self.dropout = nn.Dropout(1. - keep_prob)
        self.keep_prob = keep_prob
        kernel_size = (filter_height, filter_width)
        self.conv2d_ = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True, padding=padding)

    def forward(self, inputs):
        if self.is_train is not None and self.keep_prob < 1.0:
            self.dropout(inputs)
        '''
        tf: input tensor of shape [batch, in_height, in_width, in_channels]
        pt: input tensor of shape [batch, in_channels, in_height, in_width]
        '''
        t_in = inputs.permute(0, 3, 1, 2)
        xxc = self.conv2d_(t_in)
        out, argmax_out = torch.max(F.relu(xxc), -1)
        return out

class Multi_Conv1D(nn.Module):
    def __init__(self, is_train, cnn_dropout_keep_prob):
        super(Multi_Conv1D, self).__init__()
        self.is_train = is_train
        self.cnn_dropout_keep_prob = cnn_dropout_keep_prob

    def forward(self, inputs, filter_sizes, heights, padding, share_variable=False):
        assert len(filter_sizes) == len(heights)
        outs = []
        # for different filter sizes and height pairs
        for idx, (filter, height) in enumerate(zip(filter_sizes, heights)):
            if filter == 0:
                continue
            # input shape: size of batch, input height, input width, input channels
            batch_size, in_height, in_width, in_channels = inputs.size()
            filter_height = 1
            filter_width = height
            out_channels = filter
            '''
            Comment: Pytorch doesn't support reusable variables. However, we can reuse these
            variables by passing data through the same layers.
            '''
            conv1d = Conv1D(in_channels, out_channels, filter_height, filter_width,is_train=self.is_train, keep_prob=self.cnn_dropout_keep_prob, padding=padding)
            if (usecuda):
                conv1d = conv1d.cuda()
            out = conv1d(inputs)
            outs.append(out)

        concat_out = torch.cat(outs, 2)
        return concat_out
