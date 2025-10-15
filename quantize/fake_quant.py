import torch
from torch import nn
from functools import partial
from awq.quantize.genCodeBook import codeBookQuant
import json
import copy
from hadamard_transform import hadamard_transform
from torch.distributed.distributed_c10d import _get_group_size

def read_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)

def write_json(fp, obj):
    with open(fp, "w") as f:
        json.dump(obj, f, indent=4)


import torch

def quantize_weight_absmax(w, n_bits=8 , group_size=0, codeBookQuantInd=True, debugPath=[], debug=False):
    # w = torch.fft.fft(w , dim=-1, norm='ortho')
    # w = hadamard_transform( w.to(torch.float32) )
    # w[ w == 0 ] = 5.96*10**-8
    # scalesLog = w.abs().max(dim=-1, keepdim=True)[0]
    # w.div_(scalesLog)
    # w[ w == 0 ] = 5.96*10**-8
    # signMat = torch.sign(w)
    # w = signMat * torch.log2( w * signMat )

    # w: (out_features, in_features)
    org_w_shape = w.shape
    if group_size > 0:
        while(org_w_shape[-1] % group_size != 0):
            # print("orig_weight_row: ", org_w_shape[-1])
            # print("group_size: ", group_size)
            group_size = group_size - 32

        assert org_w_shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)

    # ad hoc dithering
    # tempMat = torch.randn( w.shape ).cuda()
    # tempMat: Any = tempMat * scales
    # w: Any = w + tempMat/20

    if codeBookQuantInd:
        w2 = copy.deepcopy(w)
        w3 = copy.deepcopy(w)

        w.div_(scales)
        w , klDiv = codeBookQuant(w, w_bit=n_bits, numCodebooks=4, numCentroids=8, numBins_hist=65, debugPath=debugPath[0], debug=debug)
        w.mul_(scales)
        
        # frobenium norm logging
        obj = read_json( debugPath[1] )
        obj.update({'Codebook '+debugPath[0]: torch.sum( torch.abs( w2 - w )).cpu().numpy().item() })        
        write_json( debugPath[1] , obj)
        obj.update({'KL Div '+debugPath[0]: klDiv.numpy().item() })        
        write_json( debugPath[1] , obj)
        w3.div_(scales).round_().mul_(scales)
        obj.update({'AWQ '+debugPath[0]: torch.sum( torch.abs( w2 - w3)).cpu().numpy().item() })        
        write_json( debugPath[1] , obj)

    else:
        w.div_(scales).round_().mul_(scales)

    assert torch.isnan(w).sum() == 0
    w = w.reshape(org_w_shape).to(torch.float16)

    # signMat = -torch.sign(w)
    # w = signMat * torch.special.exp2( w * signMat )
    # w.mul_(scalesLog)
    # w = torch.fft.ifft(w , dim=-1, norm='ortho')
    # w = hadamard_transform( w ).to(torch.float16)


    return w
    
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w_q = w.clone()
    scales = w_q.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w_q.div_(scales).round_().mul_(scales)
    return w_q.to(w.dtype)


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w_q = w.clone()
    orig_dtype = w.dtype
    scales = w_q.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w_q.div_(scales).round_().mul_(scales)
    return w_q.to(orig_dtype)


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_q = t.clone()
    t_shape = t.shape
    orig_dtype = t.dtype
    t_q = t_q.contiguous().view(-1, t_shape[-1])
    scales = t_q.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t_q.div_(scales).round_().mul_(scales)
    return t_q.view(t_shape).to(orig_dtype)

# Channel


@torch.no_grad()
def quantize_activation_per_channel_absmax(t, n_bits=8):
    t_q = t.clone()
    orig_dtype = t.dtype
    scales = torch.amax(t_q.abs(), dim=(2, 3), keepdim=True)
    q_max = 2**(n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t_q.div_(scales).round_().mul_(scales)
    return t_q.to(orig_dtype)

@torch.no_grad()
def quantize_activation_per_channel_group_absmax(t, group_size = 128, n_bits=8):
    t_q = t.clone()
    orig_dtype = t.dtype
    N, C, H, W = t.shape
    
    while(t.shape[2] % group_size != 0 or t.shape[3]%group_size != 0):
        group_size = group_size - 2
    
    print(f"Used group_size = {group_size}")
    patches = t_q.unfold(2, group_size, group_size).unfold(3, group_size, group_size)
    scales = torch.amax(patches.abs(), dim=(4, 5), keepdim=True)
    q_max = 2**(n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    quantized_patches = patches.div(scales).round().mul(scales)
    quantized_patches = quantized_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    try:
        reconstructed_t = quantized_patches.view(N, C, H, W)
    except:
        print(f"Failed to reconstruct, Input was {t.shape} and output shape is {quantized_patches.shape}, group_size is {group_size}")
    return reconstructed_t



@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_q = t.clone()
    t_shape = t.shape
    orig_dtype = t.dtype
    t_q.contiguous().view(-1, t_shape[-1])
    scales = t_q.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t_q.div_(scales).round_().mul_(scales)
    return t_q.view(t_shape).to(orig_dtype)


class WxAxLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_quant='per_channel', act_quant='per_token', quantize_output=False, n_bits_A=16, q_act = False):
        super().__init__()
        self.scales = []
        self.inputs = [] #Store as tuples (t_step, input)
        self.quantize_act = q_act
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        self.weight_quant_name = weight_quant
        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=n_bits_A)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=n_bits_A)
        
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WxAxLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if 'to_q' in str(self):
            print(f"Inside forward of {self}, self.weight.shape = {self.weight.shape}")

        if self.quantize_act == False:
            q_x = x
        else:
            q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y.to(x.dtype)

    @classmethod
    def from_linear(cls, module, init_only=False, weight_quant='per_channel', act_quant='per_token', quantize_output=False, n_bits_W=8, n_bits_A=16, group_size_W=0):
        assert isinstance(module, torch.nn.Linear)
        new_module = cls(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output, n_bits_A=n_bits_A)
        return new_module

    @staticmethod
    def from_float(module, init_only=False, weight_quant='per_channel', act_quant='per_token', quantize_output=False, n_bits_W=8, n_bits_A=16, group_size_W=0, codeBookQuantInd=False, debugPath=[], debug=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = WxAxLinear(
            module.in_features, module.out_features, module.bias is not None, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output, n_bits_A=n_bits_A)

        if init_only:
            return new_module

        if weight_quant == 'per_channel':
            new_module.weight.data.copy_(quantize_weight_per_channel_absmax(
                module.weight, n_bits_W))  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight.data.copy_(quantize_weight_per_tensor_absmax(
                module.weight, n_bits_W))
        elif weight_quant == 'group':
            new_module.weight.data.copy_(quantize_weight_absmax(
                module.weight, n_bits_W, group_size_W, codeBookQuantInd=codeBookQuantInd, debugPath=debugPath, debug=debug))
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant

        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.to(new_module.weight.dtype))
        return new_module

    def __repr__(self):
        return f'WxAxLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'

class WxAxConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        act_group_size = 1,
        weight_quant='per_tensor',
        act_quant='per_token',
        quantize_output=False,
        n_bits_A=16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.a_gs = act_group_size
        self.quantise_act = quantize_output

        assert self.in_channels % self.groups == 0

        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        self.register_buffer('weight', torch.randn(weight_shape, dtype=torch.float16, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros(self.out_channels, dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        self.weight_quant_name = weight_quant
        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=n_bits_A)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=n_bits_A)
        elif act_quant == 'per_channel':
            self.act_quant_name = 'per_channel'
            self.act_quant = partial(
                quantize_activation_per_channel_absmax, n_bits=n_bits_A)
        elif act_quant == "per_group":
            self.act_quant_name = 'per_group'
            self.act_quant = partial(
                quantize_activation_per_channel_group_absmax, n_bits=n_bits_A, group_size = self.a_gs)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WxAxConv2d, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if(self.quantise_act == True):
            q_x = self.act_quant(x)
        else:
            q_x = x
        y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        q_y = self.output_quant(y)
        return q_y.to(x.dtype)

    @classmethod
    def from_float(cls, module, init_only=False, weight_quant='per_tensor', act_quant='per_tensor',act_group_size = 1 ,quantize_output=False, n_bits_W=8, n_bits_A=16, group_size_W=0, codeBookQuantInd=False, debugPath=[], debug=False):
        assert isinstance(module, torch.nn.Conv2d)
        
        new_module = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            n_bits_A=n_bits_A,
            act_group_size = act_group_size
        )

        if init_only:
            return new_module

        if weight_quant == 'per_channel':
            new_module.weight.data.copy_(quantize_weight_per_channel_absmax(
                module.weight, n_bits_W))
        elif weight_quant == 'per_tensor':
            new_module.weight.data.copy_(quantize_weight_per_tensor_absmax(module.weight, n_bits_W))
        elif weight_quant == 'group':
            new_module.weight.data.copy_(quantize_weight_absmax(
                module.weight, n_bits_W, group_size_W, 
                codeBookQuantInd=codeBookQuantInd, debugPath=debugPath, debug=debug
            ))
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
            
        new_module.weight_quant_name = weight_quant
        
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.to(new_module.weight.dtype))            
        return new_module

    def __repr__(self):
        s = f'WxAxConv2d({self.in_channels}, {self.out_channels}, '
        s += f'kernel_size={self.kernel_size}, stride={self.stride}'
        if self.padding != (0, 0):
            s += f', padding={self.padding}'
        if self.dilation != (1, 1):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        s += f', weight_quant={self.weight_quant_name}'
        s += f', act_quant={self.act_quant_name}'
        s += f', output_quant={self.output_quant_name})'
        return s