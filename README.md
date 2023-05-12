# Fractional Laplacian Graph Neural ODE

## Introduction
Consider a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ with adjacency matrix $\mathbf{A}\in\mathbb{R}^{\lvert \mathcal{V} \rvert\times \lvert \mathcal{V} \rvert}$ such that $a_{i, j}=1$ iff $j\to i$.
Define the in- and out-degree matrices as $\mathbf{D}_1 \coloneqq \mathrm{diag}(\mathbf{A}\mathbf{1})$ and $\mathbf{D}_0 \coloneqq \mathrm{diag}(\mathbf{A}^\top\mathbf{1})$ respectively. Define the simetrically normalized adjacency as $\mathbf{L}\coloneqq \mathbf{D}_1^{-\frac{1}{2}}\mathbf{A}\mathbf{D}_0^{-\frac{1}{2}}$. Given the svd-decomposition $\mathbf{L} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\mathrm{H}$, we define the $\alpha$-fractional Laplacian as $\mathbf{L}^\alpha\coloneqq\mathbf{U}\mathbf{\Sigma}^\alpha\mathbf{V}^\mathrm{H}$.

<div style="text-align:center">
    <img src="imgs/fractional_edges.svg">
</div>

As shown, the $\alpha$-fractional Laplacian admits both positive and negative entries, and it builds long range connections.

## Graph Neural ODE
We consider the fractional heat equation $\mathbf{x}'(t) = -\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ and the fractional Schrödinger equation $\mathbf{x}'(t) = -i\mkern1mu\mathbf{L}^\alpha \mathbf{x}(t) \mathbf{W}$ with initial condition $\mathbf{x}(0)=\mathbf{x}_0$, where $\mathbf{x}_0\in\mathbb{R}^{\lvert \mathcal{V}\rvert \times F}$ are the node features and $\mathbf{W}\in\mathbb{C}^{F \times F}$ are the learnable parameters. We theoretically show that changing $\alpha$ we can converge to any frequency, making our method flexible towards different graph homophily levels.
<div style="text-align:center">
    <img src="imgs/C8_eigs.svg">
</div>


