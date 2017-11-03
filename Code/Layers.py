import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from functools import reduce
from operator import mul
import code



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

        outputs = linear_logits(args)

class Outputs(nn.Module):
    def __init__(self, is_train, input_size = 100, dropout=0.0, output_size=1):
        super(Outputs, self).__init__()

        self.dropout = dropout
        self.is_train = is_train
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, args, mask=None):
        outputs = linear_logits(args, self.is_train, self.dropout, self.linear, mask)

        return outputs

class BiAttentionLogits(nn.Module):
    def __init__(self, is_train, input_size = 100, dropout=0.0, output_size=1):
        super(Outputs, self).__init__()

        self.dropout = dropout
        self.is_train = is_train
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, args, mask=None):
        new_arg = torch.mul(args[0], args[1])
        output_args = [args[0], args[1], new_arg]
        outputs = linear_logits(output_args, self.is_train, self.dropout, self.linear, mask)

        return outputs

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