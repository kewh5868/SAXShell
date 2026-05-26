# APS Detector Stitch Legacy Notes

These notes preserve the notebook code that originally stitched APS 5-ID-D
SAXS detector exports from `hs104`, `hs103`, and `hs102`.

## Located Notebooks

- `/Users/keithwhite/repos/SolvShell/solvshellx1/saxs/`
  `APS5IDD_subtraction_pvsk.ipynb`
  - Primary workflow.
  - Loads `sol_102`, `sol_103`, `sol_104`, subtracts buffer and empty
    capillary references, trims detector overlap, rescales adjacent detectors,
    concatenates `(sub_104, sub_103, sub_102)`, plots, and writes a text file.
- `/Users/keithwhite/repos/MDScatter/scripts/pydream/_saxs/pd_tchaney/`
  `pvsk_species_simulation.ipynb`
  - Derivative workflow.
  - Loads solution and empty-capillary detector files, performs simple
    subtraction, applies the same detector-boundary scaling, interpolates onto
    simulation q values, and saves a `.npy` file.

## Legacy Stitching Pattern

The old notebook ordering was:

1. `hs104` as the low-q detector.
2. `hs103` as the middle detector.
3. `hs102` as the high-q detector.

The boundary scaling used:

```python
sub_102 = sub_102[9:, :]
sub_104 = sub_104[:485, :]

sub_103_CF = sub_104[-1, 1] / sub_103[0, 1]
sub_103[:, 1] = sub_103[:, 1] * sub_103_CF
sub_103[:, 2] = sub_103[:, 2] * sub_103_CF

sub_102_CF = sub_103[-1, 1] / sub_102[0, 1]
sub_102[:, 1] = sub_102[:, 1] * sub_102_CF
sub_102[:, 2] = sub_102[:, 2] * sub_102_CF

sub_full = np.concatenate((sub_104, sub_103, sub_102), axis=0)
```

The new implementation in `saxshell.saxs.aps_detector_stitch` keeps this
detector order but replaces fixed point trimming with automatic overlap
matching, robust median scaling, and blended overlap regions.
