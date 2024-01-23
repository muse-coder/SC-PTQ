import torch
from SC_module.SC_Mul import *
# from SC_Mul import *

class SC_LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward( input, weight, bias=None,
            Nbits = 8):
        # ctx.save_for_backward(input, weight, bias)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        input_round = torch.empty(0, device=input.device)
        device = input.device
        scale = (2**(Nbits)-1)
        torch.round(input.detach()* scale , out=input_round)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        wght_round = torch.empty(0,device=input.device)
        torch.round(weight.detach()*scale , out=wght_round)

        # output = torch.empty(0, device=weight.device)
        # torch.matmul(input_round, wght_round.transpose(1, 2), out=output)

        sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29,
                   13, 9, 25, 17, 1]

        # sobol_1 = [0, 8, 12, 4, 6, 14, 10, 2, 3, 11, 15, 7, 5, 13, 9, 1]
        sobolTensor = torch.tensor(sobol_1).to(device)

        # approximateResult = matrixMulSC(tensorData_1=input_round.squeeze(1), tensorData_2=(wght_round.squeeze(0)).transpose(0,1), rngSeq=sobolTensor, dataWidth=8,
        #                                 device=device)

        approximateResult = matrixMulSeriesSC_new(tensorData_1=input_round, tensorData_2=(wght_round).transpose(0,1) , rngSeq=sobolTensor, dataWidth=8,
                                        device=device)
        # relativeError = abs(1-approximateResult/(output.squeeze(1)))
        output = torch.empty(0, device=weight.device)
        torch.matmul(input_round, wght_round.transpose(0, 1), out=output)
        relativeError = abs(1-approximateResult/(output.squeeze(1)))
        # if (approximateResult.abs().max().log2() != output.abs().max().log2().round()):
        #     print("error")
        # assert approximateResult.abs().max().log2().round() == output.abs().max().log2().round()
        approximateResult = approximateResult/(scale * scale )
        assert torch.isnan(approximateResult).any() ==False
        if bias is not None:
            approximateResult += bias.unsqueeze(0).expand_as(approximateResult)

        return approximateResult

    # # This function has only a single output, so it gets only one gradient
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # This is a pattern that is very convenient - at the top of backward
    #     # unpack saved_tensors and initialize all gradients w.r.t. inputs to
    #     # None. Thanks to the fact that additional trailing Nones are
    #     # ignored, the return statement is simple even when the function has
    #     # optional inputs.
    #     input, weight, bias = ctx.saved_tensors
    #     grad_input = grad_weight = grad_bias = None

    #     # These needs_input_grad checks are optional and there only to
    #     # improve efficiency. If you want to make your code simpler, you can
    #     # skip them. Returning gradients for inputs that don't require it is
    #     # not an error.
    #     if ctx.needs_input_grad[0]:
    #         grad_input = grad_output.matmul(weight)
    #     if ctx.needs_input_grad[1]:
    #         grad_weight = grad_output.t().matmul(input)
    #     if bias is not None and ctx.needs_input_grad[2]:
    #         grad_bias = grad_output.sum(0)

    #     return grad_input, grad_weight, grad_bias, None, None, None, None, None
