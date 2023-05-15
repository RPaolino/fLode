import numpy as np
import torch
import torch.nn.init as I


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_num_params(model):
  r'''
  Compute the number of learnable parameters of "model".
  '''
  num_params=0
  for _, value in model.named_parameters():
    if value.requires_grad:
      num_params +=value.numel()
  return num_params
   
class fLode(torch.nn.Module):
    r'''
    Create a Fractional Laplacian Graph Neural ODE.
    '''
    def __init__(
      self, 
      in_channels: int, 
      hidden_channels: int, 
      out_channels: int, 
      num_layers: int, 
      dtype=torch.cfloat, 
      eq: str="-s",
      spectral_shift=0., 
      exponent="learnable", 
      step_size="learnable", 
      channel_mixing="d", 
      input_dropout: float=0, 
      decoder_dropout: float=0, 
      init: str="normal", 
      encoder_layers: int = 1,
      decoder_layers: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dtype = dtype
        self.eq = eq
        if self.eq=="h":
          self.iu = torch.ones(1, device=device)
        elif self.eq=="-h":
          self.iu = -torch.ones(1, device=device)
        elif self.eq=="s":
          self.iu = torch.complex(torch.zeros(1, device=device), torch.ones(1, device=device))
        elif self.eq=="-s":
          self.iu = torch.complex(torch.zeros(1, device=device), -torch.ones(1, device=device))
        
        if exponent=="learnable":
          self.exponent = torch.nn.Parameter(
            torch.ones(1, device=device)
          )
        else:
          self.exponent = torch.tensor(exponent, device=device)
          
        if spectral_shift=="learnable":
          self.spectral_shift = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=self.dtype)
          )
        else:
          self.spectral_shift = torch.tensor(spectral_shift, device=device, dtype=self.dtype)
          
        if step_size=="learnable":
          self.step_size = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=self.dtype)
          )
        else:
          self.step_size = torch.ones(1, device=device, dtype=self.dtype)*step_size
          
        assert channel_mixing in ["d", "s", "f"], f'Channel mixing {channel_mixing} not implemented.'
        self.channel_mixing = channel_mixing
        self.W = torch.nn.Parameter(
          torch.zeros((hidden_channels, hidden_channels), device=device, dtype=self.dtype)
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
      torch.manual_seed(1000)
      if self.dtype is torch.cfloat:
        getattr(torch.nn.init, self.init)(self.W.real)
        getattr(torch.nn.init, self.init)(self.W.imag)
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
    
    def parametrize_channel_mixing(self):
      r'''
      Useful to parametrize the "channel_mixing" matrix as
      - "d" diagonal matrix,
      - "s" symmetric matrix,
      - "f" full matrix.
      '''
      if self.channel_mixing=="d":
        return (self.W.diag().diag())
      elif self.channel_mixing=="s":
        return (self.W+self.W.transpose(0, 1))
      elif self.channel_mixing=="f":
        return (self.W)
      
    def LxW(self, U, S, Vh, x):
      r'''
      Compute the right-hand side of the equation.
      '''
      new_S = (self.spectral_shift+S).unsqueeze(dim=1).pow(self.exponent)
      return U @ ((new_S * (Vh @ x)) @ self.parametrize_channel_mixing())
     
    def forward_euler_step(self, U, S, Vh, x): 
      r'''
      Compute the forward euler step x'(t) \approx (x(t+h)-x(t))/h.
      '''
      x = x + self.iu * self.step_size * self.LxW(U=U, S=S, Vh=Vh, x=x)
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
        x = self.forward_euler_step(
          U=U,
          S=S,
          Vh=Vh, 
          x=x
        )
        energy[nl] = self.dirichlet_energy(
          snl=snl,
          x=x
        )
        
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