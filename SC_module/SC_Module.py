
import torch
import math
import os
import sys

# sys.path.append(os.getcwd()+"/SC_module/SC_utils")
# sys.path.append(os.getcwd()+"/SC_module/SC_Mul")

from SC_module.SC_Linear import *
from  SC_module.SC_utils import conv2d_output_shape, num2tuple
from torch.nn.modules.utils import _single, _pair, _triple
from SC_module.SC_Mul import *
from models.cifar.bit import *
from torch.cuda.amp import autocast

class Bit_SC_Conv2d(Bit_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', Nbits=8, bin=False):
        
        self.total_weight = (in_channels//groups)*out_channels*kernel_size*kernel_size
        self.total_bias = out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Bit_SC_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, Nbits, bin)

    def quant(self, maxbit=10):
    # For binary model
        ## Re-quantize the binary part of the weight, keep scale unchanged
        
        dev = self.pweight.device
        ## Quantize weight
        weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
        weight = torch.sum(weight,dim=4)
        inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
        inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
        step = 2 ** (self.Nbits)-1
        Rp = torch.round(inip * step)
        Rn = torch.round(inin * step)
        
        ## Increase layer precision if the binary part is out of range
        if self.Nbits>=maxbit:
            Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
            Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
        elif torch.max(torch.cat((Rp,Rn),0))>step:
            self.Nbits = self.Nbits+1
            self.pweight = Parameter(self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
            self.nweight = Parameter(self.nweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
            N = self.Nbits 
            ex = np.arange(N-1, -1, -1)
            self.exps = torch.Tensor((2**ex)/(2**(N)-1)).float()
            self.scale.data = self.scale.data*(2**(N)-1)/(2**(N-1)-1)
        
        for i in range(self.Nbits):
            ex = 2**(self.Nbits-i-1)
            self.pweight.data[...,i] = torch.floor(Rp/ex)
            self.nweight.data[...,i] = torch.floor(Rn/ex)
            Rp = Rp-torch.floor(Rp/ex)*ex
            Rn = Rn-torch.floor(Rn/ex)*ex
        ## Quantize bias
        if self.pbias is not None:
            weight = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
            weight = torch.sum(weight,dim=1)
            inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
            inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
            step = 2 ** (self.bNbits)-1
            Rp = torch.round(inip * step)
            Rn = torch.round(inin * step)
            
            ## Increase layer precision if the binary part is out of range
            if self.bNbits>=maxbit:
                Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
                Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
            elif torch.max(torch.cat((Rp,Rn),0))>step:
                self.Nbits = self.Nbits+1
                self.pbias = Parameter(self.pbias.data.new_zeros(self.out_channels, self.Nbits))
                self.nbias = Parameter(self.nbias.data.new_zeros(self.out_channels, self.Nbits))
                N = self.bNbits 
                ex = np.arange(N-1, -1, -1)
                self.bexps = torch.Tensor((2**ex)/(2**(N)-1)).float()
                self.biasscale.data = self.biasscale.data*(2**(N)-1)/(2**(N-1)-1)
            
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
     
    def conv2d_forward(self, input, weight, bias):
        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])
        output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size,
                                              dilation=self.dilation, pad=self.padding, stride=self.stride)
        originWeight = weight
        weight = weight.view(weight.size()[0], -1)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            input_reshape = F.pad(input, expanded_padding, mode='circular')
        
        Nbits_no_grad = self.Nbits
        mm_out = SC_LinearFunction.forward(input= input_reshape, weight = weight,bias = bias, Nbits = Nbits_no_grad)
        # assert torch.isnan(mm_out).any() ==False
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))
        
        realOutput = F.conv2d(input,  originWeight , self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])
        
    def forward(self, input):
        if self.bin:
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            weight = bit_STE.apply(torch.sum(weight,dim=4), self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias,dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        else:
            return self.conv2d_forward(input, self.weight, self.bias)
            #return self.conv2d_forward(input, STE.apply(self.weight, self.Nbits), self.bias)
        
    def L1reg(self,reg):
        param = torch.cat((self.pweight,self.nweight),0)
        total_weight = self.total_weight*self.Nbits
        reg += torch.sum(torch.sqrt(1e-8+torch.sum(param**2,(0,1,2,3))))*total_weight
        if self.pbias is not None:
            param = torch.cat((self.pbias,self.nbias),0)
            reg += torch.sum(torch.sqrt(1e-8+torch.sum(param**2,0)))
        return reg
         