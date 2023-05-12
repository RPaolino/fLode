import lib.integrate
import numpy as np
import torch
import torch.nn.init as I


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS=1e-10


def compute_num_params(model):
  r'''
  Compute the number of learnable parameters of "model".
  '''
  num_params=0
  for _, value in model.named_parameters():
    if value.requires_grad:
      num_params +=value.numel()
  return num_params

class Dropout(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.p = p

  def forward(self, x):
    if self.training:
        sample = torch.distributions.bernoulli.Bernoulli(1-self.p).sample(x.size()).to(device)
        x = x * sample
    return x 

    
class FGNODE(torch.nn.Module):
    r'''
    Create a Fractional Laplacian Graph Neural ODE.
    '''
    def __init__(
      self, 
      in_channels: int, 
      hidden_channels: int, 
      out_channels: int, 
      num_layers: int, 
      dtype="float", 
      eq: str="-h",
      spectral_shift="learnable", 
      exponent="learnable", 
      method: str="heun_euler", 
      adaptive: bool=False,
      step_size="learnable", 
      channel_mixing="f", 
      input_dropout: float=0, 
      decoder_dropout: float=0, 
      init: str="normal", 
      encoder_layers = 1,
      decoder_layers = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dtype = torch.cfloat if dtype=="cfloat" else torch.float
        self.eq = eq
        if self.eq=="h":
          self.iu = torch.ones(1, device=device)
        elif self.eq=="-h":
          self.iu = -torch.ones(1, device=device)
        elif self.eq=="s":
          self.iu = torch.complex(torch.zeros(1, device=device), torch.ones(1, device=device))
        elif self.eq=="-s":
          self.iu = torch.complex(torch.zeros(1, device=device), -torch.ones(1, device=device))
        self.method=method
        self.butcher_tableau = getattr(lib.integrate, self.method)
        self.adaptive=adaptive
        
        if exponent=="learnable":
          self.exponent = torch.nn.Parameter(
            torch.ones(1, device=device)
          )
        else:
          self.exponent=torch.tensor(exponent, device=device)
          
        if spectral_shift=="learnable":
          self.spectral_shift = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=self.dtype)
          )
        else:
          self.spectral_shift=torch.tensor(spectral_shift, device=device, dtype=self.dtype)
          
        if step_size=="learnable":
          self.step_size = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=self.dtype)
          )
        else:
          self.step_size = torch.ones(1, device=device, dtype=self.dtype)*step_size
          
        implemented_channel_mixing = ["d", "s", "f"]
        assert channel_mixing in implemented_channel_mixing, f'Channel mixing {channel_mixing} not implemented. Choose between {", ".join(implemented_channel_mixing)}.'
        self.channel_mixing=channel_mixing
        self.W = torch.nn.Parameter(
          torch.zeros((hidden_channels, hidden_channels), device=device, dtype=self.dtype)
        )
        
        self.dropout=[Dropout(p=d) for d in [input_dropout, decoder_dropout]]
                
        self.encoder = MLP(
          nin = self.in_channels, 
          nout = self.hidden_channels, 
          n_hid = self.hidden_channels, 
          nlayer=encoder_layers, 
          with_norm=None, 
          with_final_activation=False, 
          dtype=self.dtype, 
          dp=self.dropout[1]
        )
        
        self.decoder = MLP(
          nin = self.hidden_channels, 
          nout = out_channels, 
          n_hid = self.hidden_channels, 
          nlayer=decoder_layers, 
          with_norm=None, 
          with_final_activation=False, 
          dtype=self.dtype, 
          dp=self.dropout[1]
        )
        
        self.output_decoder = MLP(
            hidden_channels, 
            out_channels, 
            nlayer=2, 
            with_norm=None, 
            with_final_activation=False, 
            dtype=self.dtype
        )
        
        # Initialize the parameters
        self.reset_parameters(init)

    def reset_parameters(self, init):
      r'''
      Useful to initialize the parameters. The channel_mixing matrix is initialized as specified in "init",
      while the other linear layers using "kaiming_uniform" from torch.nn.init.
      '''
      init=init if init[-1]=="_" else init+'_'
      if self.dtype is torch.cfloat:
        getattr(torch.nn.init, init)(self.W.real)
        getattr(torch.nn.init, init)(self.W.imag)
      else:
        getattr(torch.nn.init, init)(self.W)
      for name, param in self.named_parameters():
        if param.requires_grad and (name!="W") and isinstance(param, torch.nn.Linear):
          fan_in, _ = torch.init._calculate_fan_in_and_fan_out(param.weight)
          bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
          if self.dtype is torch.cfloat:
            I.kaiming_uniform_(param.weight.real)
            I.kaiming_uniform_(param.weight.imag)
            if param.bias is not None:
              I.uniform_(param.bias.real, -bound, bound)
              I.uniform_(param.bias.imag, -bound, bound)
          else:
            I.kaiming_uniform_(param.weight)
            if param.bias is not None:
              I.uniform_(param.bias, -bound, bound)
    
    def parametrize_channel_mixing(self):
      r'''
      Useful to parametrize the "channel_mixing" matrix as
      - "d" diagonal matrix,
      - "s" symmetric matrix,
      - "f" full matrix.
      '''
      if self.channel_mixing=="d":
        return self.W.diag().diag()
      elif self.channel_mixing=="s":
        return self.W+self.W.transpose(0, 1)
      elif self.channel_mixing=="f":
        return self.W
      
    def LxW(self, data, x):
      r'''
      Compute the right-hand side of the equation.
      '''
      new_S = (self.spectral_shift+data.S).unsqueeze(dim=1).pow(self.exponent).to(self.dtype)  
      return data.U.to(self.dtype) @ ((new_S * (data.Vh.to(self.dtype) @ x.to(self.dtype))) @ self.parametrize_channel_mixing())
     
    def integration_step(self, data, previous_x): 
      LxW = self.LxW(data=data, x=previous_x).unsqueeze(dim=2)
      K = self.iu*LxW
      for i in range(1, len(self.butcher_tableau["b"])):
        ki = self.iu*(LxW+self.step_size*torch.cat([self.butcher_tableau["A"][i, k]*self.LxW(data=data, x=K[:, :, k]).unsqueeze(dim=2) for k in range(i)], dim=2).sum(dim=2, keepdim=True))
        K = torch.cat([K, ki], dim=2)
      current_x = previous_x + self.step_size * torch.cat([b * K[:, :, i].unsqueeze(dim=2) for i, b in enumerate(self.butcher_tableau["b"])], dim=2).sum(dim=2)
      return current_x

    def dirichlet_energy(self, data, x): 
      x=x.to(self.dtype)
      x=x/x.norm()
      if self.dtype == torch.cfloat:
        x_herm  = torch.complex( x.real.transpose(0,1), -x.imag.transpose(0,1) )
      else:
        x_herm = x.transpose(0, 1)
      energy = .5 * torch.trace(x_herm @ torch.sparse.mm(data.snl.to(self.dtype), x))
      return energy
    
     
    def forward(self, data):
        energy = torch.zeros(self.num_layers+1, dtype=self.dtype)
        
        x = self.dropout[0](data.x.to(dtype=self.dtype))
        x = self.encoder(x)
        
        energy[0]=self.dirichlet_energy(data=data, x=x)
        
        for nl in np.arange(1, self.num_layers+1):
          x = self.integration_step(data=data, previous_x=x)
          energy[nl] = self.dirichlet_energy(data=data, x=x)
          
        x = self.dropout[1](x)
        x = self.decoder(x)
        if self.dtype is torch.cfloat:
          x = x.abs()
        return x, energy
    


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

BN = True
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class LeakyReLU(torch.nn.Module):
  def __init__(self, negative_slope=0.01):
    super().__init__()
    self.act = nn.LeakyReLU(negative_slope)
    
  def forward(self, x, dtype):
    if dtype == torch.cfloat:
      re, im = x.real, x.imag
      re = self.act(re)
      im = self.act(im)
      return torch.complex(re, im)
    else: 
      return self.act(x)
class MLP(nn.Module):
    def __init__(
      self, 
      nin, 
      nout, 
      nlayer=2, 
      with_final_activation=True, 
      with_norm=BN, bias=True, 
      dtype=torch.cfloat, 
      n_hid = 0, 
      dp = None
    ):
        super().__init__()
        if n_hid == 0:
          n_hid = nin
        self.dropout = dp
        self.layers = nn.ModuleList([
          nn.Linear(nin if i == 0 else n_hid,
          n_hid if i < nlayer-1 else nout,
          # TODO: revise later
                    bias=True if (i == nlayer-1 and not with_final_activation and bias)
                    or (not with_norm) else False, dtype = dtype)  # set bias=False for BN
          for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this
        self.dtype = dtype

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = LeakyReLU()(x, dtype=self.dtype)
                x = self.dropout(x)
        return x