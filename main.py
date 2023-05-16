import json
from lib.best import *
from lib.transforms import *
from lib.models import *
from lib.utils import *
from lib.dataset import *
import os
import shutil
import torch

import tqdm
import argparse

from sklearnex import patch_sklearn

RESULTS_FOLDER = '.results'
if not os.path.exists(RESULTS_FOLDER):
  os.makedirs(RESULTS_FOLDER)


def training_step(model, optimizer, criterion, data, train_mask):
  model.train()
  optimizer.zero_grad()  
  out, dirichlet_energy = model(data)  # Perform a single forward pass.
  #out = out-out.max(dim=1)[0].unsqueeze(dim=1)
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
  metrics["dirichlet_energy"]["real"] = dirichlet_energy[-1].real
  if type(dirichlet_energy)==torch.cfloat:
    metrics["dirichlet_energy"]["imag"] = dirichlet_energy[-1].imag  
  pred_class = out.argmax(dim=1)  # Use the class with highest probability.
  for split, mask in zip(["train", "val", "test"], [train_mask, val_mask, test_mask]):
    metrics["loss"][split] = criterion(out[mask], data.y[mask]).item()
    correct = (pred_class[mask] == data.y[mask])  # Check against ground-truth labels.
    metrics["acc"][split] = int(correct.sum()) / int(mask.sum()) # Derive ratio of correct predictions.
  return metrics


def main(options):
  patch_sklearn()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'device: {colortext(device, "c")}.')
  transform = build_transform(
    normalize_features=options["normalize_features"],
    norm_ord=options["norm_ord"], 
    norm_dim=options["norm_dim"],
    undirected=options["undirected"],
    self_loops=options["self_loops"],
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
    "dirichlet_energy.real": np.zeros(num_splits),
    "dirichlet_energy.imag": np.zeros(num_splits)
  }
  
  # Training
  for n, nsplit in enumerate(options["num_split"]):
    seed_all()
    model = fLode(
      in_channels=dataset.num_features,
      out_channels=dataset.num_classes,
      hidden_channels=options["hidden_channels"], 
      num_layers=options["num_layers"], 
      exponent=options["exponent"], 
      spectral_shift=options["spectral_shift"], 
      step_size=options["step_size"], 
      channel_mixing=options["channel_mixing"], 
      input_dropout=options["input_dropout"], 
      decoder_dropout=options["decoder_dropout"], 
      init=options["init"], 
      dtype=torch.float if options["real"] else torch.cfloat, 
      eq=options["equation"],
      encoder_layers=options["encoder_layers"],
      decoder_layers=options["decoder_layers"],
      gcn=options["gcn"]
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
        
        description = f'Accuracy (train, val, test): ({best["acc.train"][n]*100:.2f}%, {best["acc.val"][n]*100:.2f}%, {best["acc.test"][n]*100:.2f}%)'
        progress.set_description(description)
        
        if early_stopping_counter >= options["patience"]:
          break
        
      if options["verbose"]:
            print(f'Best')
            for k in best.keys():
              print(f'| {k}: {best[k][n]}')
              
  print(f'Overall performances (mean, std)')
  avg_best = best.copy()
  for k in best.keys():
    mean = best[k].mean()
    std = best[k].std()
    avg_best[k] = [mean, std]
    print(f'| {k}: ({mean:.5f}, {std:.5f})')
  
  #Saving results
  filename=f'{RESULTS_FOLDER}/{options["dataset"]}'
  if options["dataset"] in ["film", "chameleon", "squirrel"]:
    filename += f'_undirected' if options["undirected"] else f'_directed'
  with open(filename+f'.json', 'w', encoding='utf-8') as f:
    json.dump(avg_best, f, ensure_ascii=False, indent=4)
  
  #Delete processed file
  try:
    shutil.rmtree(f'.data/{options["dataset"]}/geom_gcn/processed')
  except:
    shutil.rmtree(f'.data/{options["dataset"]}/geom-gcn/processed')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='fLode: fractional Laplacian graph neural ode')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true', help='Flag to print useful information.')
    parser.add_argument('-b', '--best', dest="best", action='store_true', help='Flag to use the hyperparams from "lib.best".')
    #Dataset
    parser.add_argument('--dataset', dest="dataset", default='chameleon', type=str, help='Which dataset to use (default chameleon).') 
    # Transforms
    parser.add_argument('-n', '--normalize_features', dest="normalize_features", action='store_true', help='Normalizes features.')
    parser.add_argument('--norm_ord', default=2, help='p-norm w.r.t. which normalize the features (default 2). Check torch.linalg.norm for the possible values. Note that we allow norm_ord="sum" to retrieve the behaviour of NormalizeFeatures() from torch_geometric.transforms.NormalizeFeatures().') 
    parser.add_argument('--norm_dim', type=int, default=0, help='Dimension w.r.t. which normalize the features (default 0).') 
    parser.add_argument('-u', '--undirected', dest="undirected", action='store_true', help='Make the graph undirected.')
    parser.add_argument('--self_loops', type=float, default=0., help='Value for the self loops (default 0.0).')
    parser.add_argument('-l', '--lcc', dest="lcc", action='store_true', help='Consider only the largest connected component.') 
    parser.add_argument('--sparsity', default=0.0, help='(1-sparsity)*num_nodes singular values will be computed and stored.') 
    parser.add_argument('--sklearn', dest="sklearn", action='store_true', help='Use the scikit-learn-intelex.extmath library to compute the svd. Useful when the graph is too large, i.e., when the torch.linalg.svd() would cause an out-of-memory error.') 
    # Model
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels (default 64).') 
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers (default 3).') 
    parser.add_argument('--exponent', default="learnable", help='Value of \alpha (float or "learnable", default "learnable").') 
    parser.add_argument('--spectral_shift', default=0.0, help='Value of spectral_shift (float or "learnable", default 0.0).') 
    parser.add_argument('--step_size', default="learnable", help='Value of step_size (float or "learnable", default "learnable").') 
    parser.add_argument('--channel_mixing', type=str, default="d", help='Which parametrization of channel_mixing to use: "d" for diagonal, "s" for symmetric, "f" for full. (defaul "d")') 
    parser.add_argument('--init', type=str, default="normal", help='Which initialization to use for channel_mixing. Check the ones implemented in torch.nn.init. (default "normal")') 
    parser.add_argument('-r', '--real', dest="real", action='store_true', help='The dtype of learnable parameters will be real.') 
    parser.add_argument('--equation', type=str, default="-s", help='Equation to solve: "h" for heat eq., "-h" for minus heat eq., "s" for Schroedinger eq., "-s" "s" for minus Schroedinger eq. (default "-s")') 
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
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping: stops after "patience" consecutive epochs in which the validation accuracy did not increase. (default 200)')
    # Num split
    parser.add_argument('--num_split', default=range(10), help='Which splits to consider (default range(10))')
    #Ablation
    parser.add_argument('--gcn', dest="gcn", action='store_true', help='The model is converted to a gcn implementing the (possibly) fractional sna.') 

    options = vars(parser.parse_args())
    
    if options["best"]:
      best_hyperparams = best_hyperparams[options["dataset"]]
      if ("directed" in best_hyperparams.keys()):
        choice = "undirected" if options['undirected'] else "directed"
        best_hyperparams=best_hyperparams[choice]
      options={
        **options,
        **best_hyperparams
      }
    
    print(f'Options')
    if options["verbose"]:
      for k in sorted(options.keys()):
        print(f'| {k}: {options[k]}')
        
    main(options)
