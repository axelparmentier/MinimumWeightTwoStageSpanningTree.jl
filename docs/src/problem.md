# Problem statement

Let ``G = (V,E)`` be an undirected graph, and ``S`` be a set of scenario. 
For each edge ``e`` in ``E``, we suppose to have a first stage cost ``c_e`` in ``\mathbb{R}``. And for each ``e`` in ``E`` and ``s`` in ``S``, we suppose to have a second stage cost ``d_{es}``.

Let ``\mathcal{P}`` be the spanning tree polytope in ``\mathbb{R}^E``. The two stage spanning tree problem can be formulated as follows,

```math
\begin{array}{ll}
\min\, & \displaystyle \sum_{e\in E}c_e x_e + \dfrac{1}{|S|}\sum_{e \in E} \sum_{s \in S}d_{es}y_{es} \\
\mathrm{s.t.}\, & \mathbf{x} + \mathbf{y}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$} 
\end{array}
```

where ``x_e`` is a binary variable indicating if ``e`` is in the first stage solution, ``y_{es}`` is a binary variable indicating if ``e`` is in the second stage solution for scenario ``s``, ``\mathbf{x} = (x_e)_{e \in E}``, and ``\mathbf{y}_s = (y_{es})_{e \in E}``.
