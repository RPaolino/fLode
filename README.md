# fLode: <ins>f</ins>ractional <ins>L</ins>aplacian graph neural <ins>ode</ins>
+ [Introduction](#introduction)
+ [Fractional heat and Schrödinger equations](#fractional-heat-and-schrödinger-equations)
+ [Experiments](#experiments)
    - [How to run the code](#how-to-run-the-code)
    - [Summary of the results](#summary-of-the-results)
+ [Cite Us](#cite-us)

## Introduction
In this work, we introduce the symmetrically normalized Laplacian $\mathbf{L}$ for directed graphs. We show theoretically that it generalizes well-known properties of the ones for undirected graphs. 

We then define the $\alpha$-fractional Laplacian $\mathbf{L}^\alpha$ using singular value calculus. The key insight is that singular values are always positive, hence, their fractional powers are always well-defined. Moreover, the SVD is more stable and accurate than the Jordan decomposition, which would be required with the usual functional calculus.


<img style="float: center;" src="imgs/fractional_edges.svg"/>

As shown in the picture, the $\alpha$-fractional Laplacian admits both positive and negative entries, and it is able to connect long distant nodes. Intuitively, this is very useful for heterophilic graphs, i.e., when nodes are more likely to be connected to nodes belonging to a different class.


## Fractional heat and Schrödinger equations
We consider the fractional Laplacian heat equation $\mathbf{x}'(t) = -\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ and the fractional Laplacian Schrödinger equation $\mathbf{x}'(t) = -i\mkern1mu\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ with initial condition $\mathbf{x}(0)=\mathbf{x}_0$, where $\mathbf{x}_0\in\mathbb{R}^{\lvert \mathcal{V}\rvert \times F}$ are the node features and $\mathbf{W}\in\mathbb{C}^{F \times F}$ are the learnable parameters. We theoretically show that tuning $\alpha$ we can converge to any frequency $\lambda$, making our method flexible towards different graph homophily levels.

<img style="float: center;" img src="imgs/C8_eigs.svg">

Real-world graphs are not purely homophilic nor purely heterophilic, but lie somewhere in between. Hence, the ability to converge to different frequencies $\lambda$ is important to enhance performances.


# Experiments
## How to run the code
Clone the repository:
```
git clone git@github.com:RPaolino/fLode.git
```
Please check the dependencies and the required packages, or create a new environment from the `environment.yml` file
```
conda env create -f environment.yml
conda activate fLode
```
To run the experiments, for example, on `chameleon`, type:
```
python main.py --dataset chameleon
```
If you want to use the best hyperparams we found, you can use the flag `-b`: this will overwrite the default values with the values saved in `lib.best`. You can specify your own configuration via command line. For a complete list of all arguments and their explanation, type:
```
python main.py -h
```
## Summary of the results
The test accuracy of our method on real-world graphs is shown below.
|               | film | squirrel | chameleon | Citeseer | Pubmed | Cora|
| :---------------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| Undirected        |      | 62.96±0.79 | 73.44±1.45 | 77.32±2.28 | | 86.52±0.64|
| Directed          |      | 73.47±2.07 | 77.61±1.86 | - | - | - |

In the paper, one can find a comparison with other models.

In order to give an idea of the computational time, we report some statistics for the undirected case. The GPU is a `NVIDIA TITANRTX`.
|               | film | squirrel | chameleon | Citeseer | Pubmed | Cora|
| :---------------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SVD [mm:ss] |   02:55   | 01:30 | 00:03 | 00:03 | | 00:04 |
| Training [iters/sec] | 5 | 4 | 10 | 8 | | 15|
# Cite Us
If you find this work interesting, please cite us.

