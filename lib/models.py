import numpy as np
import torch
import torch.nn.init as I
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_num_params(model):
  r'''
  Compute the number of learnable parameters of "model".
  '''
  num_params=0
  for name, value in model.named_parameters():
    if value.requires_grad:
      # The channel mixing matrix has different number of learnable params
      # depending on the parametrization
      if name=="W":
        if model.channel_mixing == "d":
          for v in value:
            num_params += v.diag().numel()
        elif model.channel_mixing == "s":
          for v in value:
            tmp = v.diag().numel()
            num_params += int(tmp * (tmp-1) /2)
        elif model.channel_mixing == "f":
          num_params += value.numel()
      else:
        num_params += value.numel()
  return num_params
  
class fLode(torch.nn.Module):
    r'''
    '''
    def __init__(
      self, 
      in_channels: int, 
      hidden_channels: int, 
      out_channels: int, 
      num_layers: int,
      dtype=torch.cfloat, 
      eq: str="ms",
      spectral_shift=0., 
      exponent="learnable", 
      step_size="learnable",   
      channel_mixing="d", 
      input_dropout: float=0, 
      decoder_dropout: float=0, 
      init: str="normal", 
      encoder_layers: int = 1,
      decoder_layers: int = 2,
      gcn: bool=False,
      no_sharing=False,
      layer_norm=False
    ):
        super().__init__()
        
        self.no_sharing=no_sharing
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.layer_norm = layer_norm
        self.dtype = dtype
        self.eq = eq
        self.gcn=gcn
        if self.eq=="h":
          self.iu = torch.ones(1, device=device)
        elif self.eq=="mh":
          self.iu = -torch.ones(1, device=device)
        elif self.eq=="s":
          self.iu = torch.complex(torch.zeros(1, device=device), torch.ones(1, device=device))
        elif self.eq=="ms":
          self.iu = torch.complex(torch.zeros(1, device=device), -torch.ones(1, device=device))
        
        if exponent=="learnable":
          self.exponent = torch.nn.Parameter(
            torch.ones(num_layers if no_sharing else 1, device=device)
          )
        else:
          self.exponent = torch.tensor(exponent, device=device)
          
        if spectral_shift=="learnable":
          self.spectral_shift = torch.nn.Parameter(
            torch.zeros(1, device=device, dtype=self.dtype)
          )
        else:
          self.spectral_shift = torch.tensor(spectral_shift, device=device, dtype=self.dtype)
          
        if step_size=="learnable":
          self.step_size = torch.nn.Parameter(
            torch.ones(num_layers if no_sharing else 1, device=device, dtype=self.dtype)
          )
        else:
          self.step_size = torch.ones(num_layers if no_sharing else 1, device=device, dtype=self.dtype)*step_size
          
        assert channel_mixing in ["d", "s", "f"], f'Channel mixing {channel_mixing} not implemented.'
        self.channel_mixing = channel_mixing
        self.W = torch.nn.Parameter(
          torch.zeros((num_layers if no_sharing else 1, hidden_channels, hidden_channels), device=device, dtype=self.dtype)
        )
        self.dropout=[Dropout(p=d) for d in [input_dropout, decoder_dropout]]
                      
        self.encoder = MLP(
          in_channels = self.in_channels, 
          hidden_channels = self.hidden_channels, 
          out_channels = self.hidden_channels, 
          num_layers = self.encoder_layers, 
          bias = True,
          with_norm = None, 
          with_final_activation = False, 
          dtype = self.dtype, 
          dropout = self.dropout[1]
        )
        
        self.decoder = MLP(
          in_channels = self.hidden_channels, 
          hidden_channels = self.hidden_channels, 
          out_channels = self.out_channels, 
          num_layers = self.decoder_layers, 
          bias = True,
          with_norm = None, 
          with_final_activation = False, 
          dtype = self.dtype, 
          dropout = self.dropout[1]
        )
        
        # Initialize the parameters
        self.init = init if init[-1]=="_" else init+'_'
        self.reset_parameters()

    def reset_parameters(self):
      r'''
      Useful to initialize the parameters. The channel_mixing matrix is initialized as specified in "init",
      while the other linear layers as default, i.e., using kaiming_uniform from torch.nn.init.
      '''
      if self.dtype is torch.cfloat:
        getattr(I, self.init)(self.W.real)
        getattr(I, self.init)(self.W.imag)
      else:
        getattr(torch.nn.init, self.init)(self.W)
      #Default behaviour for other linear layers
      for name, param in self.named_parameters():
        if param.requires_grad and (name!="W") and isinstance(param, torch.nn.Linear):
          fan_in, _ = torch.init._calculate_fan_in_and_fan_out(param.weight)
          bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
          if self.dtype is torch.cfloat:
            I.kaiming_uniform_(param.weight.real, a=np.sqrt(5))
            I.kaiming_uniform_(param.weight.imag, a=np.sqrt(5))
            if param.bias is not None:
              I.uniform_(param.bias.real, -bound, bound)
              I.uniform_(param.bias.imag, -bound, bound)
          else:
            I.kaiming_uniform_(param.weight, a=np.sqrt(5))
            if param.bias is not None:
              I.uniform_(param.bias, -bound, bound)
    
    def parametrize_channel_mixing(self, W):
      r'''
      Useful to parametrize the "channel_mixing" matrix as
      - "d" diagonal matrix,
      - "s" symmetric matrix,
      - "f" full matrix.
      '''
      if self.channel_mixing=="d":
        return (W.diag().diag())
      elif self.channel_mixing=="s":
        return (W+W.transpose(0, 1))
      elif self.channel_mixing=="f":
        return (W)
      
    def LxW(self, U, S, Vh, x, W, exponent):
      r'''
      Compute the right-hand side of the equation.
      '''
      #new_S = DifferentiableSum.apply(S, self.M, self.N).unsqueeze(dim=1)
      new_S = (self.spectral_shift + S).pow(exponent).unsqueeze(dim=1)
      LxW = U @ ((new_S * (Vh @ x)) @ self.parametrize_channel_mixing(W))
      return LxW
     
    def forward_euler_step(self, U, S, Vh, x, W, exponent, step_size): 
      r'''
      Compute the forward euler step x[t+1] = x[t] + h \iu L^\alpha x[t] W.
      '''
      x = x + self.iu * step_size * self.LxW(U=U, S=S, Vh=Vh, x=x, W=W, exponent=exponent)
      return x

    def gcn_step(self, U, S, Vh, x, W, exponent):
      r'''
      Compute the update step x[t+1] = = \iu L^\alpha x[t] W, i.e., a normal gcn step with the fractional Laplacian
      '''
      x = self.iu * self.LxW(U=U, S=S, Vh=Vh, x=x, W=W, exponent=exponent)
      return x

    def dirichlet_energy(self, snl, x): 
      x=x/x.norm()
      if self.dtype == torch.cfloat:
        x_herm  = torch.complex( 
          x.real.transpose(0,1), 
          -x.imag.transpose(0,1) 
        )
      else:
        x_herm = x.transpose(0, 1)
      energy = .5 * torch.trace(x_herm @ torch.sparse.mm(snl, x))
      return energy
    
     
    def forward(self, data):
      
      x=data.x.to(self.dtype)
      U, S, Vh = data.U.to(self.dtype), data.S, data.Vh.to(self.dtype)
      snl=data.snl.to(self.dtype)
      
      energy = torch.zeros(self.num_layers+1, dtype=self.dtype)
      
      x = self.dropout[0](x)
      x = self.encoder(x)
      
      energy[0]=self.dirichlet_energy(
        snl=snl, 
        x=x
      )
      
      for nl in np.arange(1, self.num_layers+1):
        if self.gcn:
          x = self.gcn_step(
            U=U,
            S=S,
            Vh=Vh, 
            x=x,
            W=self.W[nl-1] if self.no_sharing else self.W[0],
            exponent = self.exponent[nl-1] if self.no_sharing else self.exponent
          )
        else:
          x = self.forward_euler_step(
            U=U,
            S=S,
            Vh=Vh, 
            x=x,
            W=self.W[nl-1] if self.no_sharing else self.W[0],
            exponent = self.exponent[nl-1] if self.no_sharing else self.exponent,
            step_size = self.step_size[nl-1] if self.no_sharing else self.step_size
          )
        energy[nl] = self.dirichlet_energy(
          snl=snl,
          x=x
        )
        if self.layer_norm and (nl != self.num_layers):
          x = F.normalize(x, p=2, dim=1)

      x = self.dropout[1](x)
      x = self.decoder(x)
      
      if self.dtype is torch.cfloat:
        x = x.abs()
      return x, energy
    
class Dropout(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.p = p

  def forward(self, x):
    if self.training:
        sample = torch.distributions.bernoulli.Bernoulli(1-self.p).sample(x.size()).to(device)
        x = x * sample
    return x 
class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def reset_parameters(self):
        pass

class LeakyReLU(torch.nn.Module):
  r'''
  Overwriting the LeakyReLU for (possibly) complex inputs.
  '''
  def __init__(self, negative_slope=0.01, dtype=torch.cfloat):
    super().__init__()
    self.activation = torch.nn.LeakyReLU(negative_slope)
    self.dtype = dtype
    
  def forward(self, x):
    if self.dtype == torch.cfloat:
      x = torch.complex(
        self.activation(x.real), 
        self.activation(x.imag)
      )  
    else:
      x = self.activation(x) 
    return x
class MLP(torch.nn.Module):
    def __init__(
      self, 
      in_channels,
      hidden_channels, 
      out_channels, 
      num_layers=2, 
      with_final_activation=True, 
      with_norm=True, 
      bias=True, 
      dtype=torch.cfloat, 
      dropout=None
    ):
      super().__init__()
      if hidden_channels == 0:
        hidden_channels = in_channels
      self.num_layers = num_layers
      self.with_final_activation = with_final_activation
      self.dtype = dtype
      
      self.dropout = dropout
      
      self.layers = torch.nn.ModuleList([
        torch.nn.Linear(
          in_channels if i == 0 else hidden_channels,
          hidden_channels if i < num_layers-1 else out_channels,
          bias=True if (i == num_layers-1 and not with_final_activation and bias) or (not with_norm) else False, 
          dtype = dtype
        )
        for i in range(num_layers)
      ])
      self.norms = torch.nn.ModuleList([
        torch.nn.BatchNorm1d(
          hidden_channels if i < num_layers-1 else out_channels
        ) if with_norm else Identity()
        for i in range(num_layers)
      ])
        
    def reset_parameters(self):
      for layer, norm in zip(self.layers, self.norms):
        layer.reset_parameters()
        norm.reset_parameters()

    def forward(self, x):
      for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
        x = layer(x)
        if (i < self.num_layers-1) or (self.with_final_activation):
          x = norm(x)
          x = LeakyReLU(dtype=self.dtype)(x)
          x = self.dropout(x)
      return x