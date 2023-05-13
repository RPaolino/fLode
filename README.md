# Fractional Laplacian Graph Neural ODE

## Introduction
In this work, we introduce the symmetrically normalized Laplacian $\mathbf{L}$ for directed graphs. We show theoretically that it generalizes well-known properties of the ones for undirected graphs. 

We then define the $\alpha$-fractional Laplacian $\mathbf{L}^\alpha$ using singular value calculus. The key insight is that singular values are always positive, hence, their fractional powers are always well-defined. Moreover, the SVD is more stable and accurate than the Jordan decomposition, which would be required with the usual functional calculus.


<img style="float: left;" src="imgs/fractional_edges.svg"/>

As shown, the $\alpha$-fractional Laplacian admits both positive and negative entries, and it is able to connect long distant nodes. Intuitively, this is very useful when the graph is heterophilic, i.e., the nodes are more likely to be connected to nodes of different classes.


## Graph Neural ODE
We consider the fractional Laplacian heat equation $\mathbf{x}'(t) = -\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ and the fractional Laplacian Schrödinger equation $\mathbf{x}'(t) = -i\mkern1mu\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ with initial condition $\mathbf{x}(0)=\mathbf{x}_0$, where $\mathbf{x}_0\in\mathbb{R}^{\lvert \mathcal{V}\rvert \times F}$ are the node features and $\mathbf{W}\in\mathbb{C}^{F \times F}$ are the learnable parameters. We theoretically show that changing $\alpha$ we can converge to any frequency, making our method flexible towards different graph homophily levels.
<div style="text-align:center">
    <img src="imgs/C8_eigs.svg">
</div>


