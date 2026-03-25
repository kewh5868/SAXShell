# pyDREAM Workflow

The **SAXS DREAM Fit** tab is the posterior-sampling side of SAXShell. It uses
the current project, Prefit parameter table, and active template runtime inputs
to generate a pyDREAM bundle and launch a Bayesian refinement.

SAXShell uses **pyDREAM**, a Python implementation of the
**MT-DREAM(ZS)** sampler. In plain language, pyDREAM runs several Markov chains
at once, explores parameter space, and then uses the accepted samples to
estimate both a best-fit model and the uncertainty around that fit.

## What DREAM Is Doing In SAXShell

Use DREAM when you want more than one optimized answer.

Prefit gives you a fast editable preview. DREAM goes further by sampling many
plausible parameter combinations from the posterior distribution. That lets you
look at:

- one representative best-fit parameter set
- uncertainty intervals for each parameter
- how strongly the data support different parameter ranges
- whether your fit depends on a narrow or broad region of parameter space

In this repository, the DREAM workflow currently includes:

- runtime bundle generation
- prior-map editing
- saved settings presets
- progress and log output
- model-vs-experimental plots
- violin plots for posterior distributions
- filter-aware posterior summaries
- export helpers for statistics, model-fit bundles, and violin data

## Relationship To Prefit

The DREAM tab depends heavily on Prefit:

- the parameter map is built from the current Prefit parameter table
- geometry-aware templates inherit the current cluster-geometry state
- template runtime inputs are rebuilt from the saved Prefit workflow state
- the list of varying parameters comes from the current prior editor

If Prefit is out of sync, DREAM usually should not be your first stop.

## Typical Order Of Operations

1. Build a usable project in **Project Setup**.
2. Confirm the model behaves sensibly in **SAXS Prefit**.
3. Open **SAXS DREAM Fit**.
4. Review or edit the DREAM parameter map.
5. Review the search/filter preset and individual DREAM settings.
6. Write the runtime bundle.
7. Run DREAM.
8. Load the results and inspect the model plot, summary statistics, and violin
   plot.

## Core Concepts For A Lay User

### What Is A Chain?

A **chain** is one running copy of the Markov-chain Monte Carlo sampler. Each
chain starts from a different location in parameter space and walks through
candidate parameter values. Running multiple chains helps pyDREAM explore more
robustly and reduces the chance that one unlucky starting point dominates the
result.

In SAXShell:

- more chains usually means better exploration
- more chains also usually means more runtime
- the runtime bundle will raise the chain count if needed so there is at least
  one chain per varying parameter

### What Is Burn-in?

**Burn-in** is the early part of a chain that you throw away because the
sampler is still settling into the high-probability region of parameter space.

In SAXShell, burn-in is implemented as:

1. compute the number of iterations in each chain
2. remove the earliest `burnin_percent` from every chain
3. apply posterior filtering only to the remaining samples

So burn-in happens **before** summary statistics, MAP selection, violin plots,
and posterior filtering.

### What Does MAP Mean?

**MAP** stands for **Maximum a Posteriori**.

In Bayesian fitting, this means the parameter set with the highest posterior
probability among the samples being considered. In SAXShell, that is
implemented as:

1. apply burn-in
2. apply the selected posterior filter
3. among the retained samples, find the one with the highest
   log-posterior value

That sample becomes the `MAP` best-fit parameter set.

Important distinction:

- `MAP` is a **best-fit method**
- `MAP Chain Only` is a **violin-sample display option**
- `MAP` is **not** a separate posterior filter mode in the current UI

## Search/Filter Presets

The DREAM tab includes three built-in presets that change both search depth and
posterior filtering behavior:

| Preset          | Chains | Iterations | Burn-in | nSeedChains | Crossover burn-in | Default filter         | Top % | Top N |
| --------------- | -----: | ---------: | ------: | ----------: | ----------------: | ---------------------- | ----: | ----: |
| Less Aggressive |      4 |       5000 |     15% |          24 |               500 | All Post-burnin        |    20 |  1000 |
| Medium          |      4 |      10000 |     20% |          40 |              1000 | All Post-burnin        |    10 |   500 |
| More Aggressive |      8 |      20000 |     25% |          80 |              2000 | Top % by Log-posterior |     5 |   250 |

Use them as follows:

- **Less Aggressive**: faster first pass, broader retained posterior
- **Medium**: balanced default for most routine runs
- **More Aggressive**: deeper search and tighter posterior screening

If you change any of the linked controls manually, the preset switches to
`Custom`.

## DREAM Settings Explained

This section explains the main pyDREAM controls in the current SAXShell UI in
plain language.

## Edit Priors And Smart Prior Presets

The **Edit Priors** window starts from the current DREAM parameter map and lets
you change:

- which parameters vary
- the distribution family for each parameter
- the numeric distribution parameters
- a row-level **Smart Preset Status**

The smart presets do **not** replace the current prior family. Instead, they
rescale the **current width** of each row's distribution. In other words, the
current table is the baseline and the preset multiplies that baseline width.

### Width Rescaling Rule

Let `f` be the smart-preset spread factor. SAXShell applies:

- **Very Strict**: `f = 0.40`
- **Strict**: `f = 0.65`
- **Proportional**: `f = 1.00`
- **Lenient**: `f = 1.50`
- **Very Lenient**: `f = 2.25`

The exact update depends on the distribution family currently assigned to the
row.

For a normal prior:

```text
loc' = loc
scale' = f * scale
```

For a lognormal prior:

```text
loc' = loc
scale' = scale
s' = f * s
```

For a uniform prior, SAXShell preserves the current center and rescales the
width:

```text
c = loc + scale / 2
scale' = f * scale
loc' = c - scale' / 2
```

So the preset changes spread, not the intended center of the prior.

### Apply To: All Structures vs Selected Structures

For the single-mode presets above, the **Apply to** control determines whether
SAXShell adjusts:

- all structure groups in the table
- or only the currently selected structure rows

In the current implementation, a "structure group" means:

- all rows sharing the same `(structure, motif)` pair, if those fields are set
- otherwise only the specific row itself

That means if you apply `Strict` to one selected cluster structure, SAXShell
updates the whole structure/motif group together so its associated weight and
related rows stay synchronized.

### Mixed Size-Aware Presets

The two mixed presets are:

- **Strict Small / Lenient Large**
- **Lenient Small / Strict Large**

These always apply across **all** structures because SAXShell must rank the
structures against each other before deciding which ones are "small" or
"large".

First, SAXShell builds one effective radius per weight parameter:

- if a sphere radius row exists, it uses `r_eff_wN`
- if an ellipsoid is represented by semiaxes, it converts that to an
  equivalent-volume radius

For ellipsoids, the equivalent radius is:

```text
r_eq = (a * b * c)^(1/3)
```

where `a`, `b`, and `c` are the active semiaxes for that component.

Next, SAXShell computes the median radius across all weight-linked structures:

```text
r_med = median({r_i})
tol = max(r_med * 1e-9, 1e-9)
```

Each structure is then classified as:

```text
small   if r_i < r_med - tol
large   if r_i > r_med + tol
neutral otherwise
```

Once the structures are labeled, the spread factor `f` is assigned by class.

For **Strict Small / Lenient Large**:

```text
f_small = 0.65
f_large = 1.50
f_neutral = 1.00
```

For **Lenient Small / Strict Large**:

```text
f_small = 1.50
f_large = 0.65
f_neutral = 1.00
```

That factor is then applied to every row belonging to that structure group
using the same width-update rules described above.

### Smart Preset Status Column

Each row in the priors table also shows a **Smart Preset Status** column.

For the single-mode presets, the status usually matches the applied preset:

- `Very Strict`
- `Strict`
- `Proportional`
- `Lenient`
- `Very Lenient`
- `Custom / Manual`

For the mixed size-aware presets, the status is shown per structure as:

- `Strict`
- `Lenient`
- `Proportional`

This is intentional: the mixed presets are global, but the status column tells
you how each individual structure was classified and therefore how its priors
were tightened or relaxed.

### Manual Overrides

After a smart preset is applied, each structure can still be overridden
independently:

- choosing a new row-level Smart Preset Status reapplies that preset to the
  full structure/motif group for that row
- editing the value, distribution family, or raw distribution parameters marks
  that structure group as `Custom / Manual`

So the smart presets are a starting point, not a lock.

### Model Name

The run label stored in the runtime bundle and exported outputs. It helps you
identify runs later.

### Chains

How many DREAM chains to run.

- Higher values usually improve exploration.
- Higher values also increase runtime and output size.
- SAXShell may raise this automatically so there is at least one chain per
  varying parameter.

### Iterations

How many sampler steps to run **per chain**.

If you use 4 chains and 10,000 iterations, that means 40,000 total raw samples
before burn-in and filtering.

### Burn-in (%)

What fraction of the start of **each chain** is discarded before any posterior
summary is computed.

Example:

- 10,000 iterations per chain
- burn-in = 20%
- first 2,000 samples in each chain are discarded
- last 8,000 remain available for filtering and summary statistics

### History Thin

Controls how densely DREAM writes chain-history output to disk. A larger value
keeps fewer saved history points and can reduce output size.

### nSeedChains

The number of initial draws used to seed DREAM's proposal history.

Practical meaning:

- too small can make early proposal adaptation weaker
- larger values can help stabilize the sampler in harder problems

SAXShell will raise this automatically if needed so it is at least `2 x Chains`.

### Crossover Burn-in

How long DREAM waits while learning or fitting its crossover-probability
behavior.

Lay interpretation: this is part of the sampler's own adaptation period, not
the same thing as the posterior burn-in percentage above.

### Lambda

The DREAM proposal step-size scaling factor. This affects how far proposals can
jump.

- too small can make exploration slow
- too large can lower acceptance and make the chain unstable

Most users should leave this at the default unless they have a specific reason
to tune proposal behavior.

### Zeta

A very small numerical jitter added to proposals to prevent degenerate moves.
Most users should leave this unchanged.

### Snooker

The probability of using DREAM's **snooker update** move. This is an advanced
proposal option designed to help with difficult posterior geometries.

### p_gamma_unity

The probability of using proposal scaling with `gamma = 1`. This is another
advanced DREAM tuning parameter. Most users should leave the default alone.

### Verbose Sampler Output

Whether DREAM writes frequent textual progress updates to the UI log.

### Verbose Interval (s)

How often verbose output is allowed to update the UI. Smaller values mean more
frequent log text.

### Run Chains In Parallel

Whether DREAM is allowed to execute chains in parallel. This usually improves
runtime on machines with available CPU resources.

### Adapt Crossover

Whether DREAM adapts its crossover probabilities during the crossover-burn-in
period. In most cases, leaving this enabled is appropriate.

### Restart Previous Run

Continue an earlier DREAM run instead of starting from scratch.

### History File

An optional existing chain-history `.npy` file to reuse when continuing or
comparing runs.

## Best-Fit Methods

The **Best-fit method** control changes how SAXShell reduces the retained
posterior to a single representative parameter set for the model plot and
summary.

### MAP

**Maximum a Posteriori**.

SAXShell picks the retained sample with the highest log-posterior value after
burn-in and posterior filtering.

Use this when you want the single most probable retained sampled state.

### Chain Mean MAP

SAXShell finds the best retained sample **within each chain** and then averages
those per-chain MAP parameter vectors.

Use this when you want a slightly more chain-balanced representative estimate
instead of a single winning sample.

### Median

SAXShell computes the parameter-wise median across the retained posterior
samples.

Use this when you want a robust central estimate that is less sensitive to one
very sharp posterior peak.

## Posterior Filtering

Posterior filtering controls **which retained post-burn-in samples are allowed
to contribute** to the summary statistics, best-fit selection, and violin plot
data source.

In SAXShell, filtering is performed in this order:

1. apply burn-in to every chain
2. flatten the remaining samples across all chains
3. rank samples by log-posterior if the chosen filter requires ranking
4. keep the samples allowed by the active filter
5. compute MAP, medians, credible intervals, and fit-quality summaries from
   that retained set

### All Post-burnin Samples

Keep every sample that remains after burn-in.

Use this when:

- you want the broadest posterior view
- you do not want extra screening beyond burn-in
- you want violin plots and intervals to reflect the full post-burn-in sample
  cloud

### Top % by Log-posterior

Sort all post-burn-in samples by log-posterior from highest to lowest, then
keep only the top percentage.

SAXShell keeps:

- `ceil(total_post_burnin_samples * top_percent / 100)`
- with a minimum of 1 retained sample

Use this when you want to focus on the highest-probability region without
hard-coding an exact sample count.

### Top N by Log-posterior

Sort all post-burn-in samples by log-posterior from highest to lowest, then
keep only the best `N` samples.

SAXShell clamps this to:

- at least 1 sample
- no more than the total number of post-burn-in samples

Use this when you want a fixed-size retained subset across runs.

## Automatic Posterior Filter Assessment

If **Auto-select best filter after run** is enabled, SAXShell evaluates all
three filter modes after the run finishes:

- `all_post_burnin`
- `top_percent_logp`
- `top_n_logp`

It evaluates them using:

- the current best-fit method
- the current default `Top %`
- the current default `Top N`

SAXShell then recommends the filter with the best fit quality using this tie
break order:

1. lowest RMSE
2. lowest mean absolute residual
3. highest R²

If auto-select is on, that recommendation is applied automatically. If it is
off, the recommendation is reported but not applied.

## Violin Plot Sample Sources

### Filtered Posterior

Use the full set of retained samples after burn-in and posterior filtering.

This is the best choice when you want the violin plot to reflect the full
screened posterior.

### MAP Chain Only

Use only the retained samples from the single chain that contains the **global
MAP** point.

This is not a posterior filter. It is only a violin-plot data source.

Use it when you want to inspect the local behavior of the chain that produced
the winning MAP sample.

## Credible Intervals

The interval controls set the percentiles used for reported posterior bars and
summary statistics.

The defaults are:

- low = 16%
- high = 84%

That is a common choice because it roughly matches a one-standard-deviation
equivalent interval for a Gaussian-like posterior, but the posterior does not
need to be Gaussian for the percentiles themselves to remain meaningful.

## Practical Advice

- Start with **Medium** unless you already know your model is easy or very
  difficult.
- If the violin plot is still very broad, try more iterations before assuming
  the model is underdetermined.
- If the retained sample count becomes tiny under `Top %` or `Top N`, the
  summary can look overly sharp.
- If `MAP` and `Median` disagree strongly, that often means the posterior is
  skewed, multimodal, or both.
- If chains disagree strongly, increase iterations before over-tuning advanced
  sampler parameters.
- Rewrite the runtime bundle after changing priors, vary flags, geometry state,
  or template settings.

## References

- [Shockley EM, Vrugt JA, Lopez CF. _PyDREAM: high-dimensional parameter inference for biological models in python_. Bioinformatics (2018).](https://pubmed.ncbi.nlm.nih.gov/29028896/)
- [PyDREAM documentation: `run_dream()` and sampler settings.](https://pydream.readthedocs.io/en/latest/genindex.html)
- [Laloy E, Vrugt JA. _High-dimensional posterior exploration of hydrologic models using multiple-try DREAM(ZS) and high-performance computing_. Water Resources Research (2012).](https://faculty.sites.uci.edu/jasper/files/2016/04/WRR_2012.pdf)
- [Vrugt JA, ter Braak CJF, Diks CGH, Robinson BA, Hyman JM, Higdon D. _Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling_. Int. J. Nonlinear Sciences and Numerical Simulation (2009).](https://doi.org/10.1515/IJNSNS.2009.10.3.273)
