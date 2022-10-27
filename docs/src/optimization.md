
## Spanning tree polytope and branch and cut formulation

### Spanning tree polytope

The spanning tree polytope ``\mathcal{P}``

```math
    \begin{array}{ll}
        \sum_{e \in E} x_e = |V|-1 \\
        \sum_{e \in E(X)} x_e \leq |X| - 1, \qquad \text{for all } \emptyset \subsetneq X \subsetneq V
    \end{array}
```
The forest polytope is obtained when we remove the first constraint.

### Naive MILP for the separation

```math
    \begin{array}{rll}
        \min_{y,z}\, & \sum_{v\in V}z_v - \sum_{e \in E} y_e x_e -1 \\
        \mathrm{s.t.}\, & y_{e} \leq z_u \text{ and } y_e \leq z_v \qquad & \text{for all }e = (u,v) \\
        & y,z \in \{0,1\}
    \end{array}
```

### Min cut MILP for the separation problem

The separation problem 

```math
    \min  |X| - 1 - \sum_{e \in E(X)} x_e \quad \text{subject to} \quad \emptyset \subsetneq X \subsetneq V
```

is equivalent to

```math
    \min  |X| + \sum_{e \notin E(X)} x_e - |V| \quad \text{subject to} \quad \emptyset \subsetneq X \subsetneq V.
```

Let us define the digraph ``\mathcal{D} = (\mathcal{V},\mathcal{A})`` with vertex set ``\mathcal{V} = \{s,t\} \cup V \cup E`` and the following arcs.

| Arc ``a`` | Capacity ``u_a`` | 
| ------ | ----- |
| ``(s,e)`` for ``e \in E`` | ``x_e`` |
| ``(e,u)`` and ``(e,v)`` for ``e = (u,v) \in E`` | ``\infty``|
| ``(v,t)`` for ``v \in V`` | ``1`` |


The separation problem is equivalent to finding a non-empty minimum a minimum-capacity ``s``-``t`` cut ``X`` in ``\mathcal{D}``. This can be done with the following MILP.

```math
    \begin{array}{rll}
        \min \, & \sum_{a \in \mathcal{A}} u_a z_a \\
        \mathrm{s.t.} \, & y_s - y_t \geq 1 \\
        & z_a \geq y_u - y_v & \text{ for all } a= (u,v) \in \mathcal{A} \\
        & \sum_{v \in V} y_v \geq 1 \\
        & y,z \in \{0,1\} 
    \end{array}
```

### Branch and cut formulation for the minimum spanning tree problem

Using the spanning tree polytope, we can reformulate the minimum spanning tree problem as

```math
    \begin{array}{rll}
        \min\,& \displaystyle\sum_{e \in E}c_e x_e \\
        \mathrm{s.t.}\,&\displaystyle\sum_{e \in E} x_e = |V|-1 \\
        &\displaystyle\sum_{e \in E(X)} x_e \leq |X| - 1, \qquad \text{for all } \emptyset \subsetneq X \subsetneq V
    \end{array}
```

A callback based version of the Branch-and-Cut algorithm is implemented in the function `minimum_spanning_tree_MILP!`.
Its keyword argument `separate_constraint_function` enables to select the optimization algorithm used to separate constraints:

- `separate_forest_polytope_constraint_vertex_set_using_simple_MILP_formulation!` uses the naive formulation for the cut separation problem
- `separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!` uses the cut-based formulation.

### Branch and cut formulation for the two stage minimum spanning tree problem

In the same vein, we can propose the following formulation of the two stage spanning tree problem

```math
    \begin{array}{rll}
        \min\,& \displaystyle\sum_{e \in E}c_e x_e + \frac{1}{|S|}\sum_{e \in E}\sum_{s \in S}d_{es}y_{es}\\
        \mathrm{s.t.}\,&\displaystyle\sum_{e \in E} x_e + y_{es} = |V|-1 & \text{for all }s \in S \\
        &\displaystyle\sum_{e \in E(X)} x_e + y_{es} \leq |X| - 1, \qquad &\text{for all } \emptyset \subsetneq X \subsetneq E \text{ and } s \in S \\
        & x,y \in \{0,1\}
    \end{array}
```

Again, the keyword argument `separate_constraint_function` enables to select the optimization algorithm used to separate constraints among the two mentioned above.

## Column generation formulation

The cut generation previously mentioned is quite slow. 
Since minimum spanning tree can be solved efficiently, it is natural to perform and Dantzig-Wolfe reformulation of the problem previously introduced.

It leads to the following formulation.

```math
    \begin{array}{rll}
        \min\,& \displaystyle\sum_{e \in E}c_e x_e +  \frac{1}{|S|}\sum_{e \in E}\sum_{s \in S}d_{es}y_{es}\\
        \mathrm{s.t.} \,& x_e + y_{e,s} = \displaystyle\sum_{T \in \mathcal{T}\colon e \in T} \lambda_{Ts} & \text{for all $e\in E$ and $s \in S$} \\
        & \displaystyle\sum_{T \in \mathcal{T}} \lambda_{Ts} = 1 & \text{for all }s \in S \\
        & x,y\in \mathbb{Z}_+ \\
        &\lambda \geq 0
    \end{array}
```

The linear relaxation of this problem can be solved by column generation, and the problem itself can be solved using a Branch-and-Price. 
The column generation is implemented in the function `column_generation_two_stage_spanning_tree`.
Practically, we have coded directly the constraint generation on the dual using a callback mechanism.

To avoid the Branch-and-Price, we instead use a Benders decomposition of this problem, which enables to rely on the callback-mechanism of the solve, and avoid coding the branching scheme.

## Benders decomposition

Let us now perform a Benders decomposition of the column generation formulation provided above.

### Dual of the second stage problem

If we fix ``x``, the second stage problem for scenario ``s`` becomes
```math
    \begin{array}{rll}
        \min_{y,\lambda}\,& \displaystyle \frac{1}{|S|}\sum_{e \in E}d_{es}y_{es}\\
        \mathrm{s.t.} \,& y_{e,s} = \displaystyle\sum_{T \in \mathcal{T}\colon e \in T} \lambda_{Ts} - x_e  & \text{for all $e\in E$}  \\
        & \displaystyle\sum_{T \in \mathcal{T}} \lambda_{Ts} = 1 \\
        & y \geq 0\\
        &\lambda \geq 0
    \end{array}
```
We have dropped the integrality constraint on ``y`` since the spanning tree polytope is integral. 
Let us drop the ``\frac{1}{|S|}``, and remove the variables ``y`` disappear to obtain the equivalent LP

```math
    \begin{array}{rlll}
        \min_\lambda\,& \displaystyle \sum_{T \in \mathcal{T}}\lambda_T d_{Ts} - \overbrace{\sum_{s}d_{es}x_{es}}^{\text{constant}}\\
        \mathrm{s.t.} \,& \displaystyle\sum_{T \in \mathcal{T}\colon e \in T} \lambda_{Ts} \geq x_e   & \text{for all $e\in E$} & \text{(dual $\mu_{es}$)} \\
        & \displaystyle\sum_{T \in \mathcal{T}} \lambda_{Ts} = 1 &  & \text{(dual $\nu_{s}$)} \\
        &\lambda \geq 0
    \end{array}
```
where ``d_{Ts} = \sum_{e \in T}d_{es}``.
Taking its dual, we get

```math
    \begin{array}{rlll}
        \max_{\mu,\nu}\,& \displaystyle \nu_s + \sum_{e}\mu_{es}x_e - \overbrace{\sum_{s}d_{es}x_{es}}^{\text{constant}}\\
        \mathrm{s.t.} \,& d_{Rs} - \nu_s - \sum_{e \in T} \mu_{es} \geq 0 & \text{for all }T \in \mathcal{T}\\
        &\mu \geq 0
    \end{array}
```

which we can solve using a constraint generation.


#### Cut generation algorithm

The separation problem for the dual above is 
 
```math 
    \min_{T \in \mathcal{T}} \sum_{e \in T} d_{es} - \mu_{es},
```

where we have replaced ``d_{Ts}`` by its value. It is a minimum spanning tree problem and can be solved using Kruskal's algorithm.

### Feasibility and optimality cuts

If the primal admits a solution, an optimal solution ``\nu_s,\mu_{es}`` of the dual problem provides an optimality cut

```math
    \theta_s \geq \nu_s + \sum_{e}\mu_{es}x_e - \sum_{s}d_{es}x_{es}
```

If the primal problem is not feasible, the solver is supposes to return an unbounded ray for the dual, that is, a solution ``\mu,\nu`` such that

```math
    \begin{array}{ll}
        \nu_s + \sum_{e}\mu_{es}x_e > 0 \\
        -\nu_s - \sum_{e \in T} \mu_{es} \geq 0 & \text{for all }T \in \mathcal{T}
    \end{array}
```

Such an unbounded ray leads to a feasibility cut

```math
    \nu + \sum_{e}\mu_{e}x_e \leq 0
```

where we intentionally drop the scenario subscript since these feasibility cuts are not scenario dependent.
Practically, since solvers do not always provide extreme-rays, probably due to identification of unfeasibility at presolve, we consider the following simplex initialization of the primal

```math
    \begin{array}{rlll}
        \min_{\lambda,x}\,& \displaystyle \sum_{e \in E}w_e\\
        \mathrm{s.t.} \,& \displaystyle\sum_{T \in \mathcal{T}\colon e \in T} \lambda_{Ts} + w_e \geq x_e   & \text{for all $e\in E$} & \text{(dual $\mu_{es}$)} \\
        & \displaystyle\sum_{T \in \mathcal{T}} \lambda_{Ts} = 1 &  & \text{(dual $\nu_{s}$)} \\
        &\lambda,w \geq 0
    \end{array}
```
where we have added the slack variables ``w``.
Taking its dual, we get

```math
    \begin{array}{rlll}
        \max_{\mu,\nu}\,& \displaystyle \nu_s + \sum_{e}\mu_{es}x_e \\
        \mathrm{s.t.} \,& - \nu_s - \sum_{e \in T} \mu_{es} \geq 0 & \text{for all }T \in \mathcal{T}\\
        & 0 \leq \mu \leq 1 \\
        & \nu \leq 1
    \end{array}
```
which we can solve using a similar constraint generation.

Practically, we start with a constraint generation on the feasibility cut dual. 
If we do not identify a feasibility cut, we pass the constraint generated to the optimality cut dual.
Both constraint generation are implemented using a callback mechanism in function `separate_mst_Benders_cut!`. 

### Benders master problem

Let us denote by ``\mathcal{F}`` and ``\mathcal{O}_s`` the feasibility cuts and the optimality cuts for scenario ``s``. This leads us to the Benders master problem

```math
    \begin{array}{rlll}
        \min\,& \displaystyle\sum_{e \in E}c_e x_e + \sum_{s \in S} \theta_s\\
        \mathrm{s.t.} \,
        & \sum_{e \in E} x_e \leq |V| - 1 && \text{Initial constraint, not mandatory} \\
        &     \nu + \sum_{e}\mu_{e}x_e \leq 0 & \text{for all $\mu,\nu \in \mathcal{F}$}\\
        & \theta_s \geq \nu_s + \sum_{e}\mu_{es}x_e - \sum_{s}d_{es}x_{es} &  \text{for all  $s \in S$ and $(\mu,\nu) \in \mathcal{O}_s$} \\
        & x \in \{0,1\} \\
    \end{array}
```
Again, we implement the separation of the feasibility and optimality cuts using a callback mechanism in `two_stage_spanning_tree_benders`.

## Lagrangian relaxation

Let us introduce one copy of ``x`` per scenario. An equivalent formulation of the problem is

```math
\begin{array}{ll}
\min\, & \displaystyle \sum_{e\in E}c_e x_e + \sum_{e \in E} \sum_{s \in S}d_{es}y_{es} \\
\mathrm{s.t.}\, & \mathbf{x}_s + \mathbf{y}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$}  \\
& x_{es} = x_e, \quad \quad \quad \,\text{for all $e$ in $E$ and $s$ in $S$}
\end{array}
```


### Lagrangian dual function and its gradient

Let us relax the constraint ``x_{es} = x_e``. We denote by ``\theta_{es}`` the associated Lagrange multiplier.

The Lagrangian dual problem becomes

```math
\begin{array}{rlrlrl}
\max_{\theta}\mathcal{G}(\theta)= &&
\min_{x}& \sum_{e \in E}(c_e + \frac{1}{|S|}\sum_{s \in S} \theta_{es})x_e + & \frac{1}{|S|}\sum_{s \in S}\min_{\mathbf{x}_s,\mathbf{y}_s} & \sum_{e \in E}d_{es}y_{es} - \theta_{es}x_{es}\\
&&\mathrm{s.t.}& 0 \leq \mathbf{x} \leq M 
& \mathrm{s.t.}\, & \mathbf{x}_s + \mathbf{y}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$}  
\end{array}
```

where ``M`` is a large constant. 
In theory, we would take ``M=+\infty``, but taking a finite ``M`` leads to more informative gradients.

Solving the first stage subproblem amounts to checking the sign of ``c_e + \sum_{s \in S} \theta_{es}``, while the optimal solution of the second stage problem can be computed using Kruskal's algorithm.

We have

```math
    (\nabla \mathcal{G}(\theta))_{es}= \frac{1}{|S|} (x_e - x_{es}).
```

### Stochastic gradient

Considering the sum on the second stage scenarios as an expectation, we can get stochastic gradients.
Using a stochastic gradient descent on ``\max_{\theta}\mathcal{G}(\theta)`` amounts to a block coordinate descent. However, this will become more interesting in the context of learning pipelines.