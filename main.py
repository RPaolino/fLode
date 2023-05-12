from lib.best import *
from lib.transforms import *
from lib.models import *
from lib.utils import *
from lib.dataset import *
import lib.dataset
import torch

import tqdm
import argparse

from sklearnex import patch_sklearn
patch_sklearn()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {colortext(device, "c")}.')

def training_step(model, optimizer, criterion, data, train_mask):
  model.train()
  optimizer.zero_grad()  
  out, dirichlet_energy = model(data)  # Perform a single forward pass.
  out = out-out.max(dim=1)[0].unsqueeze(dim=1)
  loss = criterion(out[train_mask], data.y[train_mask]) 
  loss.backward()   
  optimizer.step()  
  return loss
  

def evaluate(model, criterion, data, train_mask, val_mask, test_mask):
  model.eval()
  metrics = {
    "loss":{},
    "acc":{},
    "dirichlet_energy": {"real": {},
                          "imag": {}}
  }
  out, dirichlet_energy = model(data)
  metrics["dirichlet_energy"]["real"] = dirichlet_energy.real
  if type(dirichlet_energy)==torch.cfloat:
    metrics["dirichlet_energy"]["imag"] = dirichlet_energy.imag  
  pred_class = out.argmax(dim=1)  # Use the class with highest probability.
  for split, mask in zip(["train", "val", "test"], [train_mask, val_mask, test_mask]):
    metrics["loss"][split] = criterion(out[mask], data.y[mask]).item()
    correct = pred_class[mask] == data.y[mask]  # Check against ground-truth labels.
    metrics["acc"][split] = int(correct.sum()) / int(mask.sum()) # Derive ratio of correct predictions.
  return metrics


def main(options):
  seed_all()
  transform = build_transform(
    normalize_features=options["normalize_features"],
    norm_ord=options["norm_ord"], 
    norm_dim=options["norm_dim"],
    undirected=options["undirected"],
    add_self_loops=options["add_self_loops"],
    lcc=options["lcc"],
    sparsity=options["sparsity"],
    sklearn=options["sklearn"],
    verbose=options["verbose"]  
  )
  dataset, data = build_dataset(
    dataset_name=options["dataset"], 
    transform=transform,
    verbose=options["verbose"]
  )
  
  criterion = torch.nn.CrossEntropyLoss()
  num_splits=len(options["num_split"])
  best={
    "loss.train": np.zeros(num_splits),
    "loss.val": np.zeros(num_splits),
    "loss.test": np.zeros(num_splits),
    "acc.train": np.zeros(num_splits),
    "acc.val": np.zeros(num_splits),
    "acc.test": np.zeros(num_splits),
    "epoch": np.zeros(num_splits),
    "exponent": np.zeros(num_splits),
    "step_size": np.zeros(num_splits),
  }
  
  # Training
  for n, nsplit in enumerate(options["num_split"]):
    model = FGNODE(
      in_channels=dataset.num_features,
      out_channels=dataset.num_classes,
      hidden_channels=options["hidden_channels"], 
      num_layers=options["num_layers"], 
      method=options["method"], 
      exponent=options["exponent"], 
      spectral_shift=options["spectral_shift"], 
      step_size=options["step_size"], 
      channel_mixing=options["channel_mixing"], 
      input_dropout=options["input_dropout"], 
      decoder_dropout=options["decoder_dropout"], 
      init=options["init"], 
      dtype=options["dtype"], 
      eq=options["equation"],
      adaptive=options["adaptive"],
      encoder_layers=options["encoder_layers"],
      decoder_layers=options["decoder_layers"]
    ).to(device)
    
    if options["verbose"]:
      print("Model")
      print(f'| num. params: {colortext(compute_num_params(model), "c")}')
      
    optimizer = getattr(torch.optim, options["optimizer"])(
      model.parameters(),  
      lr=options["learning_rate"],
      weight_decay=options["weight_decay"]    
    )
    train_mask = data.train_mask[:, nsplit].to(bool)
    val_mask = data.val_mask[:, nsplit].to(bool)
    test_mask = data.test_mask[:, nsplit].to(bool)
    with tqdm.trange(1, options["num_epochs"]+1) as progress:
      early_stopping_counter = 0 #counter for early stopping
      for epoch in progress:
        loss = training_step(
          data=data,
          model=model, 
          optimizer=optimizer, 
          criterion=criterion, 
          train_mask=train_mask, 
        )
        with torch.no_grad():
          evaluation_metrics = collapse(
            evaluate(
              model=model, 
              criterion=criterion, 
              data=data, 
              train_mask=train_mask,
              val_mask=val_mask,
              test_mask=test_mask
            )
          )
          if evaluation_metrics["acc.val"] > best["acc.val"][n]: 
            for k, v in evaluation_metrics.items():
              if k in best.keys():
                best[k][n] = v
            best["exponent"][n] = model.exponent.item()
            best["epoch"][n] = epoch
            best["step_size"][n] = model.step_size.abs().item()
            early_stopping_counter = 0
          else: 
            early_stopping_counter += 1
        
        description = f'Epoch: {epoch:03d}, early stopping: {early_stopping_counter:3d}, loss.train: {evaluation_metrics["loss.train"]:.4f}'
        progress.set_description(description)
        
        if early_stopping_counter >= options["patience"]:
          break
        
      if options["verbose"]:
            print(f'Best')
            for k in best.keys():
              print(f'| {k}: {best[k][n]}')
              
  print(f'Overall performances (mean, std)')
  for k in best.keys():
    print(f'| {k}: ({best[k].mean():.5f}, {best[k].std():.5f})')
        
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true', help='Flag to print useful information.')
    parser.add_argument('-b', '--best', dest="best", action='store_true', help='Flag to use the hyperparams from "lib.best".')
    #Dataset
    parser.add_argument('--dataset', default='chameleon', type=str, help='Which dataset to use.') 
    # Transforms
    parser.add_argument('-n', '--normalize_features', dest="normalize_features", action='store_true', help='Normalizes features.')
    parser.add_argument('--norm_ord', default=2, help='p-norm w.r.t. which normalize the features.') 
    parser.add_argument('--norm_dim', type=int, default=0, help='Dimension w.r.t. which normalize the features.') 
    parser.add_argument('-u', '--undirected', dest="undirected", action='store_true', help='Make the graph undirected.')
    parser.add_argument('-s', '--add_self_loops', dest="add_self_loops", action='store_true', help='Add self loops.')
    parser.add_argument('-l', '--lcc', dest="lcc", action='store_true', help='Consider only the largest connected component.') 
    parser.add_argument('--sparsity', default=0.0, help='(1-sparsity)*num_nodes singular values will be considered.') 
    parser.add_argument('--sklearn', dest="sklearn", action='store_true', help='Use the scikit-learn-intelex.extmath library to compute the svd.') 

    # Model
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels (default 64).') 
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers (default 3).') 
    parser.add_argument('--method', type=str, default="forward_euler", help='Method of integration (defaul "forward_euler"). Check the ones implemented in lib.integrate.')
    parser.add_argument('--exponent', default="learnable", help='Which exponent to use (float or "learnable", default "learnable").') 
    parser.add_argument('--spectral_shift', default=1e-16, help='Which spectral_shift to use (float or "learnable", default 1e-16).') 
    parser.add_argument('--step_size', default="learnable", help='Which step_size to use (float or "learnable", default "learnable").') 
    parser.add_argument('--channel_mixing', type=str, default="d", help='Which parametrization of channel_mixing to use (defaul "d"): "d" for diagonal, "s" for symmetric, "f" for full.') 
    parser.add_argument('--init', type=str, default="normal", help='Which initialization to use for channel_mixing (default "normal"). Check the ones implemented in torch.nn.init.') 
    parser.add_argument('--dtype', type=str, default="cfloat", help='(default "cfloat")"float" for real neural network, "cfloat" for complex neural network.') 
    parser.add_argument('--equation', type=str, default="-s", help='(default "-s") "h" for heat eq., "-h" for minus heat eq., "s" for Schroedinger eq., "-s" "s" for minus Schroedinger eq.') 
    parser.add_argument('--adaptive', dest="adaptive", action='store_true', help='Implement adaptive step size.') 
    parser.add_argument('--encoder_layers', type=int, default=1, help='Number of encoding layers before the neural ODE (default 1).') 
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoding layers after the neural ODE (default 1).') 
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Dropout of the first encoding layer (default 0.).') 
    parser.add_argument('--decoder_dropout', type=float, default=0.0, help='Dropout of the last decoding layer (default 0.).')
    # Optimizer 
    parser.add_argument('--optimizer', type=str, default="Adam", help='Which optimizer to use (default "Adam").')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate (default 1e-2).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default 5e-4).')
    # Training
    parser.add_argument('--num_epochs', type=int, default=1000, help='Maximal number of epochs (default 1000).')
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping (default 200): stops after "patience" consecutive epochs in which the validation accuracy did not increase.')
    # Num split
    parser.add_argument('--num_split', default=range(10), help='Which split to consider (default range(10))')

    options = vars(parser.parse_args())
    
    if options["best"]:
      best_hyperparams = lib.best.best_hyperparams[options["dataset"]]
      if ("directed" in best_hyperparams.keys()):
        choice = "undirected" if options['undirected'] else "directed"
        best_hyperparams=best_hyperparams[choice]
      options={
        **options,
        **best_hyperparams
      }
    
    main(options)
