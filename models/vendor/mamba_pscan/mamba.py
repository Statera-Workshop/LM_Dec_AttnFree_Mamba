# Vendored from RichardMinsooGo/LM_2023_Mamba_transformer-pytorch @ e8d6c81, file 2023_mamba_model.py lines 1-503
# (the LightningModule wrapper MambaLLM after line 503 is not used here; see models/registry.py)
'''
    Implementation of the parallel scan algorithm in PyTorch.
    This version is basically ported as-is from the codes in:
    - https://github.com/alxndrTL/mamba.py/blob/main/pscan.py
    - https://github.com/kyegomez/zeta/blob/be1c7e14d6c5a78f7d558ad919ec774a5f018042/zeta/nn/modules/p_scan.py
    to which all the credit goes.
'''

import math
import torch

from torch import Tensor
from einops import rearrange

from typing import Callable, Tuple

from torch.autograd import Function

class PScan(Function):
    '''
    Implementation of the parallel scan algorithm in PyTorch for
    the particular case of the cumulative filtering needed by the
    mamba architecture in its SSM stage.
    '''
    
    @staticmethod
    def forward(
        ctx,
        A_inp: Tensor,
        X_inp: Tensor,
    ) -> Tensor:
        '''Forward pass of the pscan module.

        This method performs the forward pass of the pscan module.
        It takes in two input tensors, A and X, and returns a tensor
        as output containing the result of the following operation:
        
        Y[t] = A[t] * Y[t - 1] + X[t]

        Args:
            ctx (_type_): The context object.
            A (Tensor): The input tensor A of expected shape:
                (seq_len, batch_size, d_model, d_state).
            X (Tensor): The input tensor X of expected shape:
                (seq_len, batch_size, d_model, d_state).

        Returns:
            Tensor: The result of the parallel scan.
        '''
        
        # Clone the tensors because we will modify them in-place
        A = A_inp.clone()
        X = X_inp.clone()
        
        A = rearrange(A, 'l b d s -> b d l s')
        X = rearrange(X, 'l b d s -> b d l s')
        
        # Perform the parallel scan, which modifies the input tensors in-place
        # [Unified] 2026-10-19 bug fix: save A before the in-place scan. The original saved the scanned
        # (cumulative-product) A, which made every gradient wrong.
        A_orig = A.clone()
        PScan._forward(A, X)
        
        ctx.save_for_backward(A_orig, X)
        
        return rearrange(X, 'b d l s -> b l d s')

    @staticmethod
    # TODO: Understand the implementation of the backward pass
    def backward(ctx, grad_inp: Tensor) -> Tuple[Tensor, Tensor]:
        '''Implements the backward pass for the pscan module.
        Tells the gradient how to propagate through the pscan module.

        Args:
            ctx (A, X): Saved tensors from the forward pass.
                A_in: The input tensor A of expected shape:
                    (seq_len, batch_size, d_model, d_state).
                X: The input tensor X of expected shape:
                    (seq_len, batch_size, d_model, d_state).
            grad_outputs (Tensor): The incoming gradients

        Returns:
            Tuple of Tensor: Gradients with respect to the A and X tensors.
                grad_A: The gradient with respect to A.
                grad_X: The gradient with respect to X.
                both tensor have the same shape as the input tensors.
        '''
        
        A, X = ctx.saved_tensors
        
        # Reverse both the A and grad tensor along the sequence dim
        # NOTE: Apparently A needs to be "shifted by one" to the right
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        
        grad_out = rearrange(grad_inp, 'b l d s -> b d l s')
        
        # Perform the reverse parallel scan
        grad_out = grad_out.flip(2)
        PScan._forward(A, grad_out)
        grad_out = grad_out.flip(2)
        
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_out[:, :, 1:])
        
        # [Unified] 2026-10-19 bug fix: inputs arrive as (l b d s), so gradients must be returned in that
        # layout. The original returned (b l d s), which fails whenever batch != seq_len.
        Q = rearrange(Q, 'b d l s -> l b d s')
        grad_out = rearrange(grad_out, 'b d l s -> l b d s')

        return Q, grad_out
        
    
    @staticmethod
    def _forward(A: Tensor, X: Tensor) -> None:
        '''Perform the forward pass of the parallel scan algorithm.
        Modify the input tensors in-place.

        Args:
            A (Tensor): Tensor of expected shape (batch_size, d_model, seq_len, d_state).
            X (Tensor): Tensor of expected shape (batch_size, d_model, seq_len, d_state).
        '''
        
        # Get the dimensions of the input tensors
        b, d, l, s = A.shape
        
        num_steps = int(math.log2(l))
        
        # * Upsweep phase of the scan (going up the three)
        Av = A
        Xv = X
        for _ in range(num_steps):
            T = Xv.size(2)
            
            Av = Av[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv = Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)
            
            Xv[:, :, :, 1].add_(Av[:, :, :, 1].mul(Xv[:, :, :, 0]))
            Av[:, :, :, 1].mul_(Av[:, :, :, 0])
            
            Av = Av[:, :, :, 1]
            Xv = Xv[:, :, :, 1]
            
        # * Downsweep phase of the scan (going down the three)
        for k in range(num_steps - 1, -1, -1):
            Av = A[:, :, 2**k - 1 : l : 2**k]
            Xv = X[:, :, 2**k - 1 : l : 2**k]
            
            T = 2 * (Xv.size(2) // 2)

            if T < Xv.size(2):
                Xv[:, :, -1].add_(Av[:, :, -1].mul(Xv[:, :, -2]))
                Av[:, :, -1].mul_(Av[:, :, -2])

            Av = Av[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv = Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)

            Xv[:, :, 1:, 0].add_(Av[:, :, 1:, 0].mul(Xv[:, :, :-1, 1]))
            Av[:, :, 1:, 0].mul_(Av[:, :, :-1, 1])
    
def pscan(A: Tensor, X: Tensor) -> Tensor:
    """[Unified] 2026-10-19: the scan only supports power-of-two lengths, so pad A with ones and X with
    zeros up to the next power of two (does not change earlier outputs) and cut the result back."""
    L = A.shape[0]
    n = 1 << (L - 1).bit_length()
    if n != L:
        pad = (0, 0, 0, 0, 0, 0, 0, n - L)
        A = torch.nn.functional.pad(A, pad, value=1.0)
        X = torch.nn.functional.pad(X, pad, value=0.0)
    return PScan.apply(A, X)[:, :L]

# Utils ----------------------------------------

import torch
import torch.nn as nn
from torch import Tensor
from typing import TypeVar, Tuple

from torch.utils.data import get_worker_info

T = TypeVar('T')
D = TypeVar('D')

Cache = Tuple[Tensor, Tensor] | None

def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def default_iterdata_worker_init(worker_id : int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()
    
    if worker_info is None: return
    
    dataset = worker_info.dataset
    glob_start = dataset._start # type: ignore
    glob_end   = dataset._end   # type: ignore
    
    per_worker = int((glob_end - glob_start) / worker_info.num_workers)
    worker_id = worker_info.id
    
    dataset._start = glob_start + worker_id * per_worker        # type: ignore 
    dataset._end   = min(dataset._start + per_worker, glob_end) # type: ignore

# This implementation of RMSNorm is taken directly from:
# https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    '''
    A module that performs RMS normalization on the input tensor.

    Args:
        d_model (int): The size of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-8.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (torch.Tensor): A learnable parameter used to scale the normalized input tensor.

    Methods:
        forward(x): Performs RMS normalization on the input tensor.

    Example:
        >>> rms_norm = RMSNorm(d_model=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output_tensor = rms_norm(input_tensor)
    '''
    
    def __init__(
        self,
        d_model : int,
        eps : float = 1e-8,
    ) -> None:
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x : Tensor) -> Tensor:        
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

# mamba -----------------------------------------------------------

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einops import rearrange

from typing import Tuple
from torch.nn.functional import silu
from torch.nn.functional import softplus

# from .utils import default
# from .utils import RMSNorm
# from .utils import Cache
# from .pscan import pscan

class Mamba(nn.Module):
    '''
    Class representing the Mamba model as introduced in Gu & Dao (2023)
    (see paper: https://arxiv.org/abs/2312.00752). It is a State Space
    Model with context-dependent capability that matches the performances
    of the strongest Transformer competitor (albeit only tested for small
    scales) while being much more compute efficient.
    '''
    
    def __init__(
        self,
        num_layers : int,
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
    ) -> None:
        super().__init__()
        
        mamba_par = {
            'd_input' : d_input,
            'd_model' : d_model,
            'd_state' : d_state,
            'd_discr' : d_discr,
            'ker_size': ker_size,
            'parallel': parallel,
        }
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.layers = nn.ModuleList([
            nn.ModuleList(
                [
                    MambaBlock(**mamba_par),
                    RMSNorm(d_input)
                ]
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, seq : Tensor, cache : Cache = None) -> Tuple[Tensor, Cache]:
        '''
        Forward pass of the Mamba model.
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, d_seq).
            
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, d_seq).
        '''
        
        for mamba, norm in self.layers: # type: ignore
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        return seq, cache
        
class MambaBlock(nn.Module):
    '''
    Class representing the MambaBlock as introduced in Gu & Dao (2023).
    '''
    
    def __init__(
        self, 
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
    ) -> None:
        '''Initialize the Mamba model.

        Args:
            d_input (int): The dimension of the input sequence.
            d_model (int): The dimension of the model state space.
            d_state (int, optional): The dimension of the state space in the SSM stage. Defaults to 16.
            d_discr (int | None, optional): The dimension of the discrete space in the SSM stage. Defaults to None.
            ker_size (int, optional): The kernel size for the convolutional layer. Defaults to 4.
            parallel (bool, optional): Whether to use parallel scan for the SSM stage. Defaults to False.
        '''
        super().__init__()
        
        d_discr = default(d_discr, d_model // 16)
        
        # Projection matrices from the input sequence space to the
        # model state space (of dimension d_model) and back.
        # NOTE: The in_proj matrix has a factor of 2 because it is
        #       used to split the input sequence into two branches
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        
        # Projection matrices for endowing the SSM stage with
        # context-dependent capability (i.e. input dependence)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False), # Fixing matrix rank to d_disc
            nn.Linear(d_discr, d_model, bias=False),
        )
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        )
        
        # Parameters for the SSM. Follows the S4 initialization
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float))
        
        # Whether to use or not the parallel scan for the SSM
        self.parallel = parallel
        
    def forward(self, seq : Tensor, cache : Cache = None) -> Tuple[Tensor, Cache]:
        '''
        Forward pass of the MambaBlock.
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, d_seq).
            
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, d_seq).
        '''
        b, l, d = seq.shape
        
        (prev_hid, prev_inp) = default(cache, (None, None))
        
        # Project the input sequence from d_seq to d_model and into two
        # distinct branches, one for the SSM and the residual branch
        # (see Fig. 3 of the Mamba paper). The resulting shapes are:
        # a: (batch_size, seq_len, d_model), b: (batch_size, seq_len, d_model)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        
        # * The SSM branch
        # Apply the convolutional layer to the SSM branch
        # NOTE: We need to move the channel dimension to the second dimension
        #       for the convolution to work properly, hence the rearrange
        x = rearrange(a, 'b l d -> b d l')

        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l] # Crop the output to the original length
        a = rearrange(a, 'b d l -> b l d')
        
        # Apply the SSM
        a = silu(a)
        a, hid = self.ssm(a, prev_hid=prev_hid) 
        
        # * The residual branch
        b = silu(b)
        
        # Combine the two branches
        out = a * b
        out =  self.out_proj(out)
        
        # Update the cache for next call if provided
        if cache:
            # Drop the first element of the hidden input states and attach
            # the newly computed results from the convolutions
            cache = (hid.squeeze(), x[..., 1:]) # type: ignore
        
        return out, cache
    
    def ssm(self, seq : Tensor, prev_hid : Tensor | None) -> Tuple[Tensor, Tensor]:
        '''
        State Space Model (SSM) of the MambaBlock.
        
        Args:
            seq (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        '''
        
        # Compute the context-dependent projections
        A = -self.A # shape: (d_model, d_state)
        D = +self.D # shape: (d_model, )
        
        B = self.s_B(seq)               # shape: (batch_size, seq_len, d_state)
        C = self.s_C(seq)               # shape: (batch_size, seq_len, d_state)
        Δ = softplus(D + self.s_D(seq)) # shape: (batch_size, seq_len, d_model)
        
        # Discretize the A and B parameters using Δ
        A_bar = einsum(torch.exp(A), Δ, 'd s,   b l d -> b l d s')
        B_bar = einsum(          B,  Δ, 'b l s, b l d -> b l d s')
        
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        
        # Compute the state space hidden states
        # NOTE: This can be done either sequentially (slow) or with
        # a parallel scan (fast)
        hid = self._hid_states(
            A_bar,
            X_bar,
            parallel=self.parallel,
            prev_hid=prev_hid,    
        )
        
        # Compute the output based on the hidden states
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
    
        out = out + D * seq
        
        return out, hid
    
    def _hid_states(
        self,
        A : Tensor,
        X : Tensor,
        parallel : bool = False,
        prev_hid : Tensor | None = None,
    ) -> Tensor:
        '''
        Calculate the hidden states of the SSM.

        Args:
            A (Tensor): The tensor representing A_bar.
            X (Tensor): The tensor representing X.
            parallel (bool): Whether to use parallel scan or 
                sequential computation (slower).

        Returns:
            Tensor: The tensor representing the hidden states.
        '''
        b, l, d, s = A.shape
        
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        
        if prev_hid is not None:
            # If we have a previous hidden state it means we are running the
            # efficient auto-regressive inference, so we expect both A and X
            # to have a trivial length of 1, we just drop it when returning
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
        
        h = None if parallel else torch.zeros(b, d, s, device=self.device)
        
        return pscan(A, X) if parallel else torch.stack([
            h := A_t * h + X_t
            for A_t, X_t in zip(A, X)
        ], dim=1)

    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        '''
        return next(self.parameters()).device

