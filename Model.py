import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import Constants

class Encoder(nn.Module):
	"""
	Encoder:
	Args:
	Input: seq_len, batch_size
	return:
	"""
	def __init__(self, opt):
		super(Encoder, self).__init__()
        self.vocab_size = opt.vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        self.in_channels = opt.in_channels
        self.out_channels = opt.out_channels
		self.kernel_size = opt.kernel_size
		self.stride = opt.stride
		self.padding = opt.kernel_size / 2
		self.layers = 1

		self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
		self.affine = nn.Linear(self.embedding_size, 2*self.hidden_size)
		self.softmax = nn.Softmax()

		self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,self.padding)

	def forward(self, input):
        inputs = self.embedding(input)
        outputs = self.affine(inputs)

        for i in self.layers:
            outputs = self.conv(outputs)         

        A, B = outputs.split(self.embedding_size)
        attn = torch.mul(a, self.softmax(B))
        attn_inputs = attn + inputs

        return attn, attn_inputs


class Decoder(nn.Module):
    """
    	Encoder:
    	Args:
    	Input: seq_len, batch_size
    	return:
    	"""

    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.vocab_size = opt.vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size


        self.in_channels = opt.in_channels
        self.out_channels = opt.out_channels
        self.kernel_size = opt.kernel_size
        self.stride = opt.stride
        self.padding = opt.kernel_size - 1
        self.layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2 * self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.mapping = nn.Linear(self.hidden_size, 2*self.hidden_size)

    # attn_src: src_seq_len, hidden_size
	def forward(self, source, attn_src, target):
	    inputs = self.embedding(target)
	    outputs = self.affine(inputs)

	    for i in torch.range(self.layers):
			conv_out = self.conv(outputs)
		    conv_out = conv_out[:self.kernel_size - 1]  # remove the last k - 1 elements

		    A, B = conv_out.split(self.hidden_size)
		    conv_out = torch.mul(A, self.softmax(B))

        	# This is the residual connection, 
        	# for the output of the conv will add kernel_size/2 elements 
        	# before and after the origin input
        	if i > 0:
	            conv_out = conv_out + outputs

			outputs = self.mapping(conv_out) + outputs

	        attn = torch.mm(outputs, attn_src.t()) # trt_seq_len, src_seq_len

	        attn = nn.Softmax(attn) # trt_seq_len, src_seq_len;

	        source_attn = self.mapping(attn_src) + source # src_seq_len, 2 * hidden_size

	        c = torch.mm(attn, source_attn) # trt_seq_len, 2 * hidden_size

	        outputs = outputs + c

        # leave the generator outside the model
	    # outputs = nn.Softmax(self.generator(outputs))
	    return outputs


class NMTModel(nn.Module):
	"""
	NMTModel:
	Input:
	    encoder:
	    decoder:
	    attention:
	    generator:
	return:
	"""
	def __init__(self, encoder, decocer):
		super(Model, self).__init__()
		self.encoder = encoder
		self.decocer = decocer
	
	def forward(self, source, target):
		attn_src, attn_inputs = encoder(source) # src_seq_len, embedding_size
		
		out = decocer(source, attn_src, target) # trt_seq_len, embedding_size

		return out

