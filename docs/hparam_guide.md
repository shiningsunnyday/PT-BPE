# GeoBPE Usage Guide
## 1. Overview
This document provides:
- **List of GeoBPE hyperparameters** and descriptions.
- **Practical guidelines for choosing GeoBPE hyperparameters** depending on downstream application.
- **General principles** describing the tradeoff between compression, runtime, and distortion.
- **Learned Lessons** from past experiment logs, plots, and scripts.


---

## 2. List of Hyperparameters

| Hyperparameter | Suggested Value | Description | Sensitivity on Performance | Notes |
|----------------|---------|------|-------------|--------|
| **`--max-iter`** | see below | How many BPE iterations to perform | High | Will save tokenizer checkpoints, can choose which iter to use
| **`--bins`** | see below | How many bins to quantize angles (and bond lengths if `--free-bonds true`) | High |
| **`--num-p`** | see below | How many medoids to introduce from clustering with BPE iter schedule | High | Passed as colon-delimited `{key}-{value}` pairs
| `--res-init` | true | Whether to initialize tokens at residue orientation modes or at bond level | N/A | Should always be True unless doing dev |
| `--free-bonds` | true | Whether GeoBPE should consider variability in bond lengths  | Marginal from ablation studies | Backbone bond lengths are usually restrained |
| `--bin-strategy` | histogram | How to quantize angles/lengths | Low-Medium | histogram is generally best
| `--max-num-strucs` | 500 | Max occurrences for k-medoids | Low from ablation in paper | Runtime bottleneck, but ablations show no gains beyond 5000
| `--glue-opt` | true | Whether to do rigid body refine | Medium | Setting false makes run faster but much higher distortion
| **`--glue-opt-every`** | 10 | How often to glue-opt | Not tested | Periodic glue opt for faster runs
| `--glue-opt-method` | all  | Whether to optimize all glues or iteratively | High | Should be true for best results unless for custom uses
| `--wr --wt` | 1.0, 0.1 | torsion IK loss (R, t) weighting terms | Medium | Can be tuned for performance |
| `--rmsd-super-res` | true | Whether motif occurrences reference original | Low-Medium | Can achieve super-resolution effects, should be further explored

### 2.1 Which ones actually matter
Although GeoBPE exposes many parameters, only a few dictate overall behavior, performance and runtime. Essential knobs are:
- **Vocabulary Size**
    - `num_p` (number of medoids)
    - `max_iter`(#iterations to run)  
- **Fidelity (trades off distortion/runtime)**
    - `glue_opt`/`glue_opt_method`/`glue_opt_every` (glue optimization)

Aside from the bolded ones, we suggest leaving the rest to default values.

Other lesser hyperparameters and utility parameters (plotting, logging, etc.) are omitted. You can read the help strings in the scripts if you want to know more.

---

## 3. Guidelines for Hyperparameter Selection

### 3.1 Choosing **`num_p`** (medoids per step) and **`bins`** (angle/length quantization strength)

We define `num_p` to follow a step schedule. For example, `{2:2, 3:5, 5: 10}` (passed as `--num-p 2-2:3-5:5-10`) means:
- Introduce $2$ tokens for geo keys with $2$ bonds (C-terminus residue orientations)
- $5$ for geo keys with $3$ bonds (all non-terminus residues)
- $10$ for all merged geo keys with $5$ or more bonds (every GeoBPE step after residue initialization).

`bins` uses the same syntax as `num_p`. For example, `{1: 100, 3: 10}` (passed as `--bins 1-100:3-10`) introduces $100$ bins to bin the angular histogram at initialization. The binning strategy is controlled by `bin-strategy`. If glue opt optimizes angles outside the range, we snap to closest bin. It should be increased or decreased together with `num_p`.

We will provide recommendations based on downstream use.

<u>GeoBPE for Compression/Reconstruction</u>

*Intuition.* Higher `num_p` values $\rightarrow$ better reconstruction at the cost of storing a bigger vocabulary and leading to noisier merges.  We see gains on reconstruction (RMSD/LDDT) diminish rapidly beyond `num_p={2:200, 3:1000}`. There are only limited modes in the conformational variability of energetically favored backbone regions (Ramachandran). 

*Recommendation.* Use a high `bins[size]` should be used to maximize reconstruction, but it should not be too high to avoid a combinatorial explosion in space of possible geo keys. `bins={1: 500}` works well for most cases.

<u>GeoBPE for Representation Learning</u>

*Intuition.* GeoBPE emits both sequential tokens and merge hierarchies, with emphasis on the latter for learning strong representations from the residue to protein level. A good hierarchy should capture higher-level patterns from basic secondary structures to functional sites, and not be sensitive to regions wtih high-frequency vibrations. Thus, the focus shifts from compression to *coarsening*. 

*Recommendation.* We suggest small `num_p` values (our paper results used `2-2:3-5:5-1:6-2:8-1`). It should be accompanied by small `bins` values (we used 1-50 with the `histogram-cover` strategy).

---

### 3.2 Choosing **Number of Merge Iterations**

Vocabulary size increases by `num_p[|key|]` per iteration. Intuitively, more iterations result in a more varied vocabulary, but each ``word" (motif) gets less usage. The best stopping iteration again depends on the downstream use:

<u>GeoBPE for Representation Learning</u>

GeoBPE's merge hierarchy serve as useful inductive bias for downstream predictive tasks. Merged token pairs correspond to secondary structures and align with domain hits. Thus, GeoBPE should coarsen high-resolution details into higher-level motifs. We suggest 

<u>GeoBPE for Compression/Reconstruction</u>

Based on BPE, originally a text compression algorithm, GeoBPE seeks to minimize distortion while maximizing compression. Thus, `num_p`

<u>GeoBPE for Language Modeling</u>

Since BPE is a standard tokenizer for LLM training, a common heuristic from NLP is to stop when $|V|/L\approx N/1000$ ($L$ is avg. tokens per structure, $N$ structures).

This plays out like the following for different model scales.

|   LM Scale             |   #Structures N  |   L (motifs/structure)  |   \|V\|≈(L×N)/1000=T/1000  |   Model #params  |
|---------------------|------------------|-------------------------|--------------------------|------------------|
|   Toy/demo          |   1k             |   100                   |   ~100                   |   1M             |
|   Small/usable      |   10k            |   100                   |   ~1k                    |   10-50M         |
|   Base “GPT-small”  |   100k           |   100                   |   ~10k                   |   100M           |
|   Mid-scale         |   1M             |   100                   |   ~100k                  |   >=1B           |
|...|...|...|...|...

We implement a stopping criterion based on this heuristic. The star indicates when to stop. It is computed and shown in `run_{iter}.png` plots in the run directory.

![LM_heuristic](./lm-heuristic.png)


P.S. In practice, you can set a high `max_iter` and let it run in the background. You can choose which checkpoint to use later.


### 3.3 Runtime Considerations
- K medoids complexity is quadratic in the parameter `max-num-strucs` ($M$), not `num_p`. An ablation shows $M=20k$ yields no gains ($\Delta_{RMSD}\approx 0.01$) over $M=5k$.
- `glue_opt_every` ($P$) is the period of glue opt, which can take over runtime if done frequently and on many merges (hinging on $\frac{T}{P} O(N \log N)$ term). We use $P=10$ for our results, and anecdotally did not find performance drops vs $P=1$.
