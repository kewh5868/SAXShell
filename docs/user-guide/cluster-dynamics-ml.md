# Cluster Dynamics ML

`clusterdynamicsml` is the predictive companion to `clusterdynamics`. It can be
launched directly from the `clusterdynamicsml` application entry point or from
the main SAXSShell UI. It
combines:

- time-binned cluster dynamics from extracted XYZ or PDB frame folders
- observed reference structure ensembles organized by stoichiometry label
- a lightweight regularized regression model for extrapolating larger clusters
- geometry statistics learned from the observed structures

The result is a ranked set of predicted larger-cluster candidates, predicted
structure files, combined histogram views, and an optional SAXS mixture model
that includes both observed and predicted structures.

## What the application does

At a high level, Cluster Dynamics ML answers this question:

> Given the smaller clusters that are observed in a trajectory and the
> reference structures available for those smaller clusters, what larger
> clusters are plausible, how much population should they carry, and what
> representative structures should they have?

It is not a black-box generative model. The current implementation is a
feature-engineered predictive model with explicit physical constraints and
geometry rules.

## Inputs

Cluster Dynamics ML expects all of the inputs required by `clusterdynamics`,
plus a structure library for the observed smaller clusters.

### Required inputs

- an extracted frames folder from `mdtrajectory`
- atom-type definitions that identify `node`, `linker`, and optional `shell`
  atoms
- pair-cutoff definitions or a default cutoff
- a smaller-cluster structures folder organized by stoichiometry label

### Optional inputs

- experimental SAXS data for fitting and comparing the predicted mixture model
- a CP2K `.ener` file for the lower subplot inherited from `clusterdynamics`
- an active SAXSShell project directory so datasets, reports, and history are
  saved into the project
- periodic boundary conditions, shell-growth levels, and shell-sharing options

## Typical workflow

1. Load the extracted XYZ or PDB frame folder.
2. Set the atom-type definitions and pair cutoffs.
3. Point the tool at the folder containing the observed smaller-cluster
   structure files.
4. Set the target node-count range to extrapolate.
5. Set the number of candidate stoichiometries to keep per target size.
6. Set the predicted share threshold used to prune tiny candidate populations.
7. Optionally load experimental SAXS data.
8. Run **Analyze and Predict Larger Clusters**.
9. Review the `Summary`, `Lifetimes`, `Debye-Waller`, `Histograms`, and
   `SAXS` tabs.
10. Save the dataset, CSV exports, or a detailed PowerPoint report if needed.

## Training data assembled by the workflow

The workflow first runs the standard cluster-dynamics analysis and then joins
that kinetic summary with the observed structure library.

For each observed stoichiometry label, the training row combines:

- stoichiometry and node count
- observed mean count per frame
- occupancy fraction
- association and dissociation rates
- completed and window-truncated lifetime counts
- mean and standard-deviation lifetime
- mean atom count
- mean radius of gyration
- mean maximum radius
- mean semiaxis lengths
- representative structure path and motif directories

These rows are represented internally by
`ClusterDynamicsMLTrainingObservation`.

## What the model actually learns

The current implementation fits separate regularized linear models for each
predicted scalar quantity. It does not use a neural network, graph neural
network, random forest, or diffusion model.

### Feature vector

For a candidate stoichiometry with node elements \(N\) and non-node elements
\(X_1, X_2, \dots, X_k\), the feature vector is:

\[
\mathbf{x} =
\left[
1,\;
\text{node count},\;
\text{total atom count},\;
\frac{X_1}{\text{node count}},\;
\frac{X_2}{\text{node count}},\;
\dots,\;
\frac{X_k}{\text{node count}}
\right]
\]

In code, this is the `_candidate_feature_vector` helper.

### Properties predicted by regression

Separate models are fit for:

- mean count per frame
- occupancy fraction
- mean lifetime
- association rate
- dissociation rate
- radius of gyration
- maximum radius
- semiaxis `a`
- semiaxis `b`
- semiaxis `c`
- each non-node element count

### Regression form

The models are weighted ridge regressions with a small diagonal penalty:

\[
\hat{\beta} =
\left(X^\mathsf{T} W X + \lambda I \right)^{-1}
X^\mathsf{T} W y
\]

where:

- \(X\) is the feature matrix
- \(W\) is the diagonal matrix of per-observation stability weights
- \(\lambda = 10^{-6}\) in the current implementation

Several positive-valued targets are fit in `log1p` space and transformed back
after prediction so the model is smoother for counts, rates, radii, and
lifetimes.

### Stability weighting

Training rows are not weighted equally. Each row receives a larger weight when
it is supported by more structural examples, more completed lifetimes, larger
mean count per frame, larger occupancy, and longer lifetime. This biases the
fit toward better-supported observed clusters rather than treating all labels as
equally reliable.

## How candidate stoichiometries are generated

The workflow does not directly regress from one observed label to one predicted
label. It first proposes a small set of candidate compositions for each target
node count and then scores them.

Two candidate-generation routes are used:

1. `Trend extrapolation`
   Uses the weighted average node-element fractions plus the non-node element
   count regressors to build a new composition from the target node count.
2. `Composition scaled from observed <label>`
   Starts from each observed label and scales its non-node counts in proportion
   to the requested target node count.

The candidate list is then filtered by explicit support rules.

### Stoichiometry support constraints

Candidates are removed when they violate any of the following checks:

- Required linker floors:
  If a linker element is present in every observed multi-node cluster, the
  larger predicted cluster must also include that linker.
- Pure-node support:
  A candidate with no non-node atoms is only kept when the training data show
  that pure-node clusters are already plausible at nearby sizes.
- Deduplication:
  Duplicate stoichiometries are merged after normalization.

These rules are why the workflow can reject obviously confusing candidates such
as iodide-free larger clusters when all observed multi-node references include
iodide.

## Geometry statistics extracted from the observed structures

After the training rows are assembled, the workflow scans the observed
structure files and learns empirical geometry summaries from them.

### Quantities measured

- node-node bond lengths
- node-linker and node-shell bond lengths
- nearest-pair contact distances for all tracked element pairs
- contact distances grouped by atom type (`node`, `linker`, `shell`)
- node-centered bond angles
- node coordination medians by neighbor type
- non-node coordination to one or more node atoms
- non-node coordination medians to other non-node atom types

In practical terms, the learned statistics include the kinds of values users
care about when judging the predicted structures:

- bond lengths
- coordination numbers
- bond angles
- relative atom positions around node atoms
- linker-linker distances
- linker-shell distances
- other non-node contact distances

## How the predicted structure file is built

The output structure is generated in stages. It is not copied from a single
template file, and it is not generated by directly sampling a force field.

### 1. Seed the node scaffold

If a representative observed structure is available, the workflow tries to
reuse its node geometry as an initial seed. Otherwise it starts from a minimal
node seed.

### 2. Grow the larger node network

Additional node atoms are placed one at a time using:

- the learned median node-node bond length
- node-node connectivity inferred from observed scaffolds
- node-centered angle preferences
- collision penalties that discourage unrealistic crowding

### 3. Place linker and shell atoms

Non-node atoms are placed after the node scaffold is built. Their placement
order is determined by the learned coordination behavior:

- atoms that usually bridge multiple node atoms are placed first
- terminal atoms are attached afterward

For each atom, the workflow evaluates candidate positions using:

- target bond lengths to attached node atoms
- node-centered bond angles
- learned non-node contact distances
- learned non-node coordination counts
- penalties for short contacts and over-coordination

This is why linker-linker and linker-shell behavior now influences the final
predicted structures instead of only satisfying node-centered coordination.

### 4. Preserve the geometry-guided local distances

The code still carries the predicted maximum radius as a learned descriptor, but
the final global rescaling step is currently a no-op. In practice this means
the output structure keeps the local bond lengths and angles generated by the
geometry-guided placement stage instead of being stretched afterward.

### 5. Write predicted structure files

Each retained predicted candidate is written as its own XYZ structure file. The
export includes node and non-node atoms that belong to the predicted cluster
definition.

## Debye scattering with pairwise Debye-Waller damping

Cluster Dynamics ML now distinguishes between two SAXS-component cases:

- `Averaged component`
  When a SAXS component is already averaged over many structure files, the
  averaging itself captures thermal disorder and motif variability.
- `Single-structure component`
  When the component is computed directly from one XYZ or PDB structure, the
  trace is missing that ensemble broadening unless a disorder model is added
  explicitly.

This second case is exactly where the Debye-Waller-aware Debye equation is
used. In practice, that means it applies to predicted-structure SAXS traces and
to any observed component that must fall back to a single representative
structure instead of an averaged project component.

<!-- prettier-ignore-start -->

### Classical Debye equation

For atomic coordinates $\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_N$, the
current Debye scattering calculation is:

$$
I(q) =
\sum_{i=1}^{N}
\sum_{j=1}^{N}
f_i(q) f_j(q)
\frac{\sin\!\left(q r_{ij}\right)}{q r_{ij}}
$$

where:

- $q$ is the magnitude of the scattering vector
- $f_i(q)$ is the X-ray form factor of atom $i$
- $r_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert$

In the code this is implemented with the normalized sinc form,
$\operatorname{sinc}(q r_{ij} / \pi)$, which is mathematically equivalent to
$\sin(q r_{ij}) / (q r_{ij})$.

### Debye-Waller-extended single-structure equation

For a single representative structure, Cluster Dynamics ML uses a pairwise
Debye-Waller damping factor on the off-diagonal pair contributions:

$$
I_{\mathrm{DW}}(q) =
\sum_{i=1}^{N} f_i(q)^2
+
\sum_{\substack{i=1 \\ i \ne j}}^{N}
\sum_{j=1}^{N}
f_i(q) f_j(q)
\frac{\sin\!\left(q r_{ij}\right)}{q r_{ij}}
\exp\!\left(
-\frac{q^2 \sigma_{\alpha(i)\beta(j)}^2}{2}
\right)
$$

where:

- $\alpha(i)$ is the element of atom $i$
- $\beta(j)$ is the element of atom $j$
- $\sigma_{\alpha\beta}$ is the pair-specific thermal displacement parameter
  for the element pair $(\alpha, \beta)$

The diagonal self-scattering terms $i=j$ are left undamped. Only the
interference terms between distinct atoms are attenuated.

### Relation between $\sigma$ and $B$

Cluster Dynamics ML reports both the Gaussian displacement width
$\sigma_{\alpha\beta}$ and the equivalent Debye-Waller $B$ coefficient:

$$
B_{\alpha\beta} = 8 \pi^2 \sigma_{\alpha\beta}^2
$$

so the same damping factor can also be written as:

$$
\exp\!\left(
-\frac{q^2 \sigma_{\alpha\beta}^2}{2}
\right)
=
\exp\!\left(
-\frac{B_{\alpha\beta} q^2}{16 \pi^2}
\right)
$$

This is the form that will be useful later if these coefficients are exposed to
main-model refinement.

## How Debye-Waller coefficients are estimated

Cluster Dynamics ML estimates pairwise disorder from the observed structure
ensembles before it predicts values for the larger clusters.

### Observed-cluster ensemble estimate

For one observed stoichiometry label $L$ and one element pair
$(\alpha, \beta)$, each structure file $s$ contributes all pair distances of
that element type:

$$
\left\{
d^{(s)}_{L,\alpha\beta,1},
d^{(s)}_{L,\alpha\beta,2},
\dots
\right\}
$$

Those distances are sorted within each structure:

$$
d^{(s)}_{L,\alpha\beta,(1)}
\le
d^{(s)}_{L,\alpha\beta,(2)}
\le
\dots
\le
d^{(s)}_{L,\alpha\beta,(n_s)}
$$

and aligned by rank across the ensemble up to the smallest available pair
count

$$
n_* = \min_s n_s
$$

For each aligned rank $k$, the workflow measures the ensemble spread:

$$
\sigma_{L,\alpha\beta,(k)}
=
\operatorname{StdDev}_s
\left[
d^{(s)}_{L,\alpha\beta,(k)}
\right]
$$

and then aggregates those rankwise spreads into one label-level pair estimate:

$$
\sigma_{L,\alpha\beta}
=
\sqrt{
\frac{1}{n_*}
\sum_{k=1}^{n_*}
\sigma_{L,\alpha\beta,(k)}^2
}
$$

Finally,

$$
B_{L,\alpha\beta} = 8 \pi^2 \sigma_{L,\alpha\beta}^2
$$

This gives one pairwise $\sigma$ and $B$ estimate per observed cluster type and
per element pair type whenever the structure ensemble is large enough to
measure a spread.

### Predicted-cluster estimate

The larger predicted clusters do not have their own ensembles yet, so Cluster
Dynamics ML fits a separate weighted ridge-regression model for each element
pair type using the observed $\sigma_{L,\alpha\beta}$ values as the training
targets.

For a candidate feature vector $\mathbf{x}$, the predicted disorder value is:

$$
\hat{\sigma}_{\alpha\beta}(\mathbf{x})
=
\exp\!\left(
\mathbf{x}^{\mathsf{T}} \hat{\beta}_{\alpha\beta}
\right) - 1
$$

when the target is fit in `log1p` space, followed again by

$$
\hat{B}_{\alpha\beta}(\mathbf{x}) =
8 \pi^2 \hat{\sigma}_{\alpha\beta}(\mathbf{x})^2
$$

The feature vector is the same one already used for the other Cluster Dynamics
ML properties:

$$
\mathbf{x} =
\left[
1,\;
\text{node count},\;
\text{total atom count},\;
\frac{X_1}{\text{node count}},\;
\dots,\;
\frac{X_k}{\text{node count}}
\right]
$$

This keeps the Debye-Waller prediction consistent with the rest of the
population, size, and geometry prediction workflow.

<!-- prettier-ignore-end -->

## How predicted populations are assigned

Each candidate receives:

- a predicted mean count per frame
- a predicted occupancy fraction
- a predicted mean lifetime
- a derived stability score used for ranking

The predicted population share is normalized from these predicted quantities.
If the direct SAXS-style weight collapses to zero, the code falls back to
occupancy and then to lifetime divided by the frame timestep. This avoids a
pathological zero-share result for a candidate that is still physically plausible.

When predicted structures are mixed with the observed structures, the total
predicted mass is anchored to the observed size tail rather than letting a
single extrapolated candidate dominate the whole distribution.

## Outputs

Cluster Dynamics ML can produce:

- the standard time-binned colormap from `clusterdynamics`
- a combined lifetime table containing observed and predicted rows
- a `Debye-Waller` table listing the resolved \(\sigma\) and \(B\) values for
  each observed and predicted element pair
- histogram views for observed-only and observed-plus-predicted populations
- SAXS traces for observed-only and observed-plus-predicted models
- one predicted structure file per retained predicted candidate
- reloadable JSON datasets
- CSV exports for the colormap and lifetime table
- a detailed PowerPoint report

If prediction history is enabled and a project folder is active, each run is
also cached in the project so different parameter settings can be compared
later.

## Parameters that directly influence the prediction

The most important user-controlled parameters are:

- `Clusters folder`
  The observed reference structures that define the training ensemble.
- `Predict from node count` / `Predict through node count`
  The target size range for extrapolation.
- `Candidates / size`
  The number of ranked stoichiometry candidates retained per target size.
- `Share threshold`
  The minimum predicted share used to prune the low-population tail.
- `Atom type definitions`
  Which elements are treated as nodes, linkers, and shell atoms.
- `Pair cutoffs`
  The structural neighborhood rules used in cluster extraction and geometry
  statistics.
- `Shell options`
  Whether shell atoms are counted in stoichiometry labels and how shell growth
  is handled.
- `Experimental data`
  Whether a fitted SAXS comparison is built for the predicted mixture.

## Important constraints and limitations

The current algorithm is intentionally conservative. Its main constraints are:

- Small-data regime:
  The model assumes that only a modest number of observed cluster labels are
  available. It is designed to work in a data-sparse extrapolation setting.
- Extrapolation by composition trends:
  The regression model only sees node count, total atom count, and non-node to
  node ratios. It does not learn a latent representation from raw coordinates.
- Median-based geometry summaries:
  Geometry is driven by empirical medians rather than full probabilistic
  distributions.
- Single representative structure per predicted candidate:
  The output is a representative structure, not an ensemble of conformers. The
  Debye-Waller extension partially restores thermal broadening in the SAXS trace
  but does not replace an explicit conformer ensemble.
- No explicit energy minimization:
  The placement routine is geometry-guided and penalty-based; it is not a
  molecular mechanics or DFT relaxation.
- No atom-identity tracking:
  The kinetics come from count changes over time bins, not persistent atomwise
  trajectories of individual cluster instances.
- Not a graph neural network:
  The current implementation does not learn directly from the full graph or 3D
  coordinate tensor of each structure.

These constraints are deliberate. They keep the workflow transparent and stable
in the low-data extrapolation regime, but they also limit how expressive the
model can be.

## Similar machine-learning algorithms

The current method is closest to a regularized, feature-engineered predictive
model. Related model families include:

- Ridge regression:
  The direct ancestor of the current scalar prediction model.[^hoerl1970]
- Lasso and elastic net:
  Useful when feature selection or stronger sparsity is desired.[^tibshirani1996]
  [^zou2005]
- Gaussian process regression:
  A flexible predictive model with uncertainty estimates, often useful in
  small-data scientific settings.[^rasmussen2006]
- Message passing neural networks:
  A much more expressive graph-based alternative for molecular property
  prediction when larger training sets are available.[^gilmer2017]

## Why the current algorithm was chosen

Cluster Dynamics ML is trying to extrapolate from a small observed size series
to larger unobserved clusters. In that setting, a simple regularized model has
practical advantages:

- easier to inspect and debug
- less likely to overfit a tiny observed set
- easier to constrain with chemical support rules
- easier to couple to explicit geometry heuristics
- easier to explain when a predicted stoichiometry or structure looks wrong

That tradeoff is the main reason the implementation currently favors weighted
ridge-style regression plus geometry rules over a higher-capacity learned model.

## TODO

The current Debye-Waller workflow is intentionally scoped to Cluster Dynamics ML
result inspection and to the single-structure component traces built inside that
tool. A later extension may expose these pairwise \(B\) or \(\sigma\)
coefficients to the main SAXS prefit and DREAM refinement templates as optional
refinable parameters, but that is not yet part of the default SAXSShell model
workflow.

## Related pages

- [Cluster Extraction](cluster-extraction.md)
- [Cluster Dynamics](cluster-dynamics.md)
- [Project Configuration](project-configuration.md)
- [SAXS Prefit](saxs-prefit.md)
- [Results and Export](results-and-export.md)

## References

[^hoerl1970]: Hoerl, A. E., and Kennard, R. W. "Ridge Regression: Biased Estimation for Nonorthogonal Problems." _Technometrics_ 12, no. 1 (1970): 55-67. <https://doi.org/10.1080/00401706.1970.10488634>

[^tibshirani1996]: Tibshirani, R. "Regression Shrinkage and Selection via the Lasso." _Journal of the Royal Statistical Society: Series B (Methodological)_ 58, no. 1 (1996): 267-288. <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>

[^zou2005]: Zou, H., and Hastie, T. "Regularization and Variable Selection via the Elastic Net." _Journal of the Royal Statistical Society: Series B (Statistical Methodology)_ 67, no. 2 (2005): 301-320. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>

[^rasmussen2006]: Rasmussen, C. E., and Williams, C. K. I. _Gaussian Processes for Machine Learning._ MIT Press, 2006. <https://gaussianprocess.org/gpml>

[^gilmer2017]: Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. "Neural Message Passing for Quantum Chemistry." _Proceedings of the 34th International Conference on Machine Learning_ 70 (2017): 1263-1272. <https://proceedings.mlr.press/v70/gilmer17a.html>
