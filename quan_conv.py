import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
	

class ScaleSigner(Function):
	"""take a real value x, output sign(x)*E(|x|)"""
	@staticmethod
	def forward(ctx, input):
		return torch.sign(input) * torch.mean(torch.abs(input))
	@staticmethod
	def backward(ctx, grad_output):
		return grad_output

def scale_sign(input):
	return ScaleSigner.apply(input)

class Quantizer(Function):
	@staticmethod
	def forward(ctx, input, nbit):
		scale = 2 ** nbit -1
		return torch.round(input * scale) / scale

	@staticmethod 
	def backward(ctx, grad_output):
		return grad_output, None

def quantize(input, nbit):
	return Quantizer.apply(input, nbit)

def dorefa_w(w, nbit_w):
	if nbit_w == 1:
		w = scale_sign(w)
	else:
		w = torch.tanh(w)
		w = w / (2 * torch.max(torc.abs(w))) + 0.5
		w = 2 * quantize(w, nbit_w) - 1

	return w

def dorefa_a(input, nbit_a):
	return quantize(torch.clamp(input, 0, 1), nbit_a)


class QuanConv(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(QuanConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.nbit_w=nbit_w
		self.nbit_a=nbit_a
		name_w_dict={'dorefa':dorefa_w}
		name_a_dict={'dorefa':dorefa_a}
		self.quan_w = name_w_dict[quan_name_w]
		self.quan_a = name_a_dict[quan_name_a]

	def forward(self, input):
		if self.nbit_w<32:
			w = self.quan_w(self.weight, self.nbit_w)
		else:
			w = self.weight

		if self.nbit_a<32:
			x = self.quan_a(input, self.nbit_a)
		else:
			x = F.relu(input)

		output = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
		return output
