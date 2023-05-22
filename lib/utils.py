import numpy as np
import random
import torch

class bcolors:
  r'''
  Useful to print colored in the consol
  '''
  h = '\033[95m' #header
  b = '\033[94m' #blue
  c = '\033[96m' #cyan
  g = '\033[92m' #green
  r = '\033[91m' #red
  w = '\033[93m' #warning
  f = '\033[91m' #fail
  endc = '\033[0m' 
  bf = '\033[1m' #bold
  u = '\033[4m' #underline

def colortext(text, color):
  return f'{getattr(bcolors, color)}{text}{bcolors.endc}'


def seed_all(seed=1000):
  r'''
  Manually set the seeds for torch, numpy
  '''
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
def collapse(dict2collapse={}, collapsed_key="", collapsed_dict={}, verbose=False):
  r'''
  Given a nested dict of dicts, the function returns a collapsed dict
  dict whose nested keys are joined with a '.'.
  EXAMPLE
  -------
  dict2expand = {
    'a': {
      'b': 3
      'c': 5
    }
  }
  collapsedDict = {
    'a.b': 3,
    'a.c': 5
  }
  '''
  dict2collapse=dict2collapse.copy()
  collapsed_dict=collapsed_dict.copy()
  for key, value in dict2collapse.items():
    if type(value) is dict:
      r'''
      If the value is a dictionary, this must be collapsed as well. Hence, we get here a recursion.
      The key of the collapsed dict is then the concatenation of the collapsed key and the actual key.
      '''
      if collapsed_key=='':
        collapsed_dict = collapse(value.copy(), key, collapsed_dict)
      else:
        collapsed_dict = collapse(value.copy(), collapsed_key+f'.{key}', collapsed_dict)
    else:
      r'''
      If the value is not a dictionary, we can set the value.
      '''
      if collapsed_key=='':
        collapsed_dict[key]=value
      else:
        collapsed_dict[collapsed_key+f'.{key}']=value
  return collapsed_dict

def float_or_learnable(value):
  r'''
  Useful for parsing variables that are either "learnable" or float.
  '''
  if value=="learnable":
    return value
  else:
    return float(value)
  
def norm_ord_type(value):
  r'''
  Useful for parsing the argument of norm_ord.
  '''
  if value in ["fro", "nuc"]:
    return value
  elif value in ["inf", "-inf"]:
    return float(value)
  else:
    return int(value)