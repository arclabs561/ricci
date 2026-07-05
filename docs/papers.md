# Papers

What each paper referenced by the README actually shows, and what this crate
takes from it. One entry per reference, in the same order.

### Ricci curvature of Markov chains on metric spaces (Ollivier, 2009)

Defines a curvature for any metric space carrying a random walk: put a
probability measure `μ_x` around each point (here, one step of a walk from
`x`), and set `κ(x, y) = 1 − W1(μ_x, μ_y) / d(x, y)`, where `W1` is the
optimal-transport distance. Positive curvature means neighborhoods are
closer together than their base points (sphere-like); zero is flat
(grids); negative means neighborhoods spread apart faster than the points
(trees, expanders). On manifolds this recovers classical Ricci curvature
up to scaling.

ricci takes: this is the exact quantity `curvature` computes per edge, with
the lazy-walk parameter `alpha` holding mass at the base node.

### Understanding over-squashing and bottlenecks on graphs via curvature (Topping, Di Giovanni, Chamberlain, Dong, Bronstein, ICLR 2022)

Formalizes oversquashing: information from exponentially many distant nodes
must compress through fixed-width representations, and the layer-to-layer
Jacobian bounds show the loss concentrates on edges that act as narrow
bridges between dense neighborhoods. Those are precisely the negatively
curved edges, and the paper's rewiring algorithm (stochastic discrete Ricci
flow) adds support edges around the most negative ones to relieve the
bottleneck.

ricci takes: the diagnostic use. Compute edge curvature directly and read
the most negative edges as bottleneck candidates, before reaching for
rewiring or architecture changes.

### Semi-supervised classification with graph convolutional networks (Kipf, Welling, ICLR 2017)

The GCN layer: multiply node features by a symmetrically normalized
adjacency matrix, then by a weight matrix, then apply a nonlinearity. It is
a first-order truncation of spectral graph convolutions, and its simplicity
is what made graph neural networks a practical default.

ricci takes: `GCNConv` is this layer.

### Hyperbolic neural networks (Ganea, Becigneul, Hofmann, NeurIPS 2018)

Makes neural building blocks work inside the Poincare ball using Mobius
gyrovector operations: Mobius addition, scalar multiplication, and exp/log
maps at arbitrary base points, tied together by the conformal factor
`λ_x`. With these, linear layers and recurrent cells get principled
hyperbolic analogues instead of ad-hoc projections.

ricci takes: `PoincareBall`'s operation set (project, mobius_add, exp/log
maps, distance, parallel transport) is this toolkit; the README's formula
table is the paper's operation table.

### Hyperbolic graph convolutional neural networks (Chami, Ying, Re, Leskovec, NeurIPS 2019)

HGCN: run graph convolution in hyperbolic space by round-tripping through
the tangent space. Log-map point features to the tangent plane, apply the
linear transform and neighborhood aggregation there, exp-map back, and
apply the activation in tangent space. Hierarchical graphs embed with far
less distortion than in Euclidean space, and the gains track how tree-like
the graph is.

ricci takes: `HGCNConv` implements the tangent-space recipe on the Poincare
ball at fixed curvature.

### Sinkhorn distances (Cuturi, NeurIPS 2013)

Adding an entropy term to the optimal-transport objective makes it strictly
convex and solvable by Sinkhorn's matrix-scaling iterations, orders of
magnitude faster than exact linear-programming solvers. The regularization
strength trades accuracy against speed and conditioning.

ricci takes: every edge's `W1` in the curvature computation is solved with
log-domain Sinkhorn; `CurvatureConfig` exposes the regularization and the
iteration cap.

### Lovasz meets Weisfeiler and Leman (Dell, Grohe, Rattan, ICALP 2018)

Two graphs are indistinguishable by the k-dimensional Weisfeiler-Leman
refinement (the hierarchy bounding message-passing expressiveness) iff they
have equal homomorphism counts from every graph of treewidth at most k. For
k = 1 that means trees: a 1-WL-bounded network literally cannot see any
structure beyond tree counts. The standard blind spot: a 6-cycle and two
disjoint triangles have identical tree homomorphism counts.

ricci takes: the motivation for `features::hom_profile`, and the
C6-versus-two-triangles pair as the module's test oracle.

### Graph neural networks with local graph parameters (Barcelo, Geerts, Reutter, Ryschkov, NeurIPS 2021)

Injecting homomorphism counts of small patterns as extra node features
provably lifts message-passing expressiveness beyond 1-WL, at preprocessing
cost only, and the paper characterizes which patterns add power. Cycle
(closed-walk) counts are the cheapest useful family, since they carry
exactly the information tree-bounded aggregation misses.

ricci takes: `features` is this recipe, with walk and closed-walk profiles
as the feature vectors.

### Modeling relational data with graph convolutional networks (Schlichtkrull et al., ESWC 2018)

R-GCN: extend graph convolution to directed, labeled multigraphs by giving
every relation type its own transform, summed with a self-loop transform
(`Σ_r Σ_{j ∈ N_i^r} (1/c_{i,r}) W_r h_j + W_0 h_i`); relations appear in
both canonical and inverse directions as distinct types. Because parameters
grow linearly with relation count, two regularizations are introduced:
basis decomposition (`W_r = Σ_b a_rb V_b`, shared bases with per-relation
coefficients) and block-diagonal decomposition; on FB15k-237 the block
variant won, and an R-GCN encoder under a DistMult decoder beat the
decoder-only baseline by 29.8%. The paper flags its own weak point: fixed
`1/c` normalization degrades on high-degree hub nodes.

ricci takes: `RGCNConv` is the layer with the basis decomposition
implemented (block-diagonal is not); adjacencies are caller-normalized as
with `GCNConv`, and both directions of a relation enter as separate stack
entries, per the paper's convention.
