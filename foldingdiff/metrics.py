from pathlib import Path
import multiprocessing
import pandas as pd
import os
import logging
import warnings
from foldingdiff.algo import *
from typing import *
from tqdm import tqdm
import tempfile
from foldingdiff.angles_and_coords import create_new_chain_nerf
from foldingdiff.tmalign import max_tm_across_refs, run_tmalign_gdt_ts
from foldingdiff.lddt import lddt
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals
from itertools import groupby
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
    basename_prefix: str = "generated_",
    threads: int = multiprocessing.cpu_count(),
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(
        f"Writing sampled angles as PDB files to {outdir} using {threads} threads"
    )
    # Create the pairs of arguments
    arg_tuples = [
        (os.path.join(outdir, f"{basename_prefix}{i}.pdb"), samp)
        for i, samp in enumerate(final_sampled)
    ]
    # Write in parallel
    with multiprocessing.Pool(threads) as pool:
        files_written = pool.starmap(create_new_chain_nerf, arg_tuples)

    return files_written
    

def generate_residues_proteinmpnn(
    pdb_fname: str, n_sequences: int = 8, temperature: float = 0.1
) -> List[str]:
    """
    Generates residues for the given pdb_filename using ProteinMPNN

    Trippe et al uses a temperature of 0.1 to sample 8 amino acid sequences per structure
    """
    bname = os.path.basename(pdb_fname).replace(".pdb", ".fa")
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        cmd = f'python {PROTEINMPNN_SCRIPT} --pdb_path_chains A --out_folder {tempdir} --num_seq_per_target {n_sequences} --seed 1234 --batch_size {n_sequences} --pdb_path {pdb_fname} --sampling_temp "{temperature}" --ca_only'
        retval = subprocess.call(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        assert retval == 0, f"Command {cmd} failed with return value {retval}"
        outfile = tempdir / "seqs" / bname
        assert os.path.isfile(outfile)

        # Read the fasta file, return the sequences that were generated
        seqs = read_fasta(outfile)
        seqs = {k: v for k, v in seqs.items() if k.startswith("T=")}
    assert len(seqs) == n_sequences
    return list(seqs.values())

def best_tm_vs_train(pdb, train_pdbs):
    best_tm, _ = max_tm_across_refs(pdb, train_pdbs, parallel=True)
    return best_tm

def best_lddt_vs_train(pdb, train_pdbs):
    return max(lddt(pdb, ref) for ref in train_pdbs)


def phi_psi_angles(pdb_path):
    # returns (N,2) array of [φ,ψ] in degrees
    df = canonical_distances_and_dihedrals(str(pdb_path))
    return df[["phi", "psi"]].dropna().values


def hist2d(dataset):
    H, _x, _y = np.histogram2d(
        x = np.concatenate([d[:,0] for d in dataset]),
        y = np.concatenate([d[:,1] for d in dataset]),
        bins=bins, density=True
    )
    # avoid zero‑prob
    H += 1e-8
    return H / H.sum()

def best_gdt_ts_vs_train(query_pdb: str,
                        train_pdbs: list[str],
                        fast: bool = True) -> float:
    """
    For one generated backbone `query_pdb`, run TMalign against every
    PDB in `train_pdbs` and return the maximum GDT‑TS score found.
    """
    best = -np.inf
    for ref in train_pdbs:
        res = run_tmalign_gdt_ts(query_pdb, ref, fast=fast)
        gdt = res["gdt_ts"]          # may be NaN if TMalign failed
        if np.isnan(gdt):
            continue
        best = max(best, gdt)
    return best if best > -np.inf else np.nan

def compute_gdt_ts_novelty(generated_pdbs: list[str],
                        train_pdbs: list[str],
                        fast: bool = True) -> dict:
    """
    Returns a dict with the mean best GDT‑TS and the per‑sample list.
    """
    best_scores = [
        best_gdt_ts_vs_train(q, train_pdbs, fast=fast)
        for q in tqdm(generated_pdbs, desc="GDT‑TS vs train")
    ]
    best_scores = np.array(best_scores, dtype=float)
    mean_best   = float(np.nanmean(best_scores))   # ignore NaNs
    return {
        "gdt_ts_best_per_sample": best_scores.tolist(),
        "gdt_ts_mean_best":       mean_best,
    }

SSE_BACKEND = Literal["dssp", "psea"]


def count_structures_in_pdb(
    fname: str, backend: SSE_BACKEND = "psea"
) -> Tuple[int, int]:
    """Count the secondary structures (# alpha, # beta) in the given pdb file"""
    assert os.path.exists(fname)

    # Get the secondary structure
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    source = PDBFile.read(fname)
    if source.get_model_count() > 1:
        return (-1, -1)
    source_struct = source.get_structure()[0]
    chain_ids = np.unique(source_struct.chain_id)
    assert len(chain_ids) == 1
    chain_id = chain_ids[0]

    if backend == "psea":
        # a = alpha helix, b = beta sheet, c = coil
        ss = struc.annotate_sse(source_struct, chain_id)
        # https://stackoverflow.com/questions/6352425/whats-the-most-pythonic-way-to-identify-consecutive-duplicates-in-a-list
        ss_grouped = [(k, sum(1 for _ in g)) for k, g in groupby(ss)]
        ss_counts = Counter([chain for chain, _ in ss_grouped])

        num_alpha = ss_counts["a"] if "a" in ss_counts else 0
        num_beta = ss_counts["b"] if "b" in ss_counts else 0
    elif backend == "dssp":
        # https://www.biotite-python.org/apidoc/biotite.application.dssp.DsspApp.html#biotite.application.dssp.DsspApp
        app = dssp.DsspApp(source_struct)
        app.start()
        app.join()
        ss = app.get_sse()
        ss_grouped = [(k, sum(1 for _ in g)) for k, g in groupby(ss)]
        ss_counts = Counter([chain for chain, _ in ss_grouped])

        num_alpha = ss_counts["H"] if "H" in ss_counts else 0
        num_beta = ss_counts["B"] if "B" in ss_counts else 0
    else:
        raise ValueError(
            f"Unrecognized backend for calculating secondary structures: {backend}"
        )
    logging.debug(f"From {fname}:\t{num_alpha} {num_beta}")
    return num_alpha, num_beta


def ss_counts(pdb_paths, backend="psea", n_threads=8):
    """
    Return a list of (n_alpha, n_beta) tuples for each pdb in pdb_paths,
    skipping multichain structures (count_structures_in_pdb returns (-1,-1)).
    """
    # from multiprocessing import Pool
    # with Pool(n_threads) as pool:
    #     counts = list(
    #         pool.starmap(
    #             count_structures_in_pdb,
    #             [(p, backend) for p in pdb_paths],
    #             chunksize=10,
    #         )
    #     )
    counts = [count_structures_in_pdb(str(p), backend) for p in pdb_paths]
    # drop failed
    return [c for c in counts if c != (-1, -1)]
def ss_kl_divergence(generated_pdbs, train_pdbs,
                    max_bins=8, backend="psea") -> dict:
    """
    Compute secondary‑structure content metrics.
    Returns dict with KL divergence and mean helix/sheet counts.

    The 2‑D histogram ranges over bins [0,max_bins) for both α and β counts.
    All counts >= max_bins go into the last bin.
    """
    gen_counts   = ss_counts(generated_pdbs, backend=backend)
    train_counts = ss_counts(train_pdbs,   backend=backend)
    # gather means for convenience
    alpha_gen  = np.array([a for a, _ in gen_counts])
    beta_gen   = np.array([b for _, b in gen_counts])
    alpha_tr   = np.array([a for a, _ in train_counts])
    beta_tr    = np.array([b for _, b in train_counts])
    # 2‑D histograms
    bins = np.arange(max_bins + 1)  # edges 0..max_bins
    H_gen, _x, _y = np.histogram2d(alpha_gen.clip(max_bins-1),
                                beta_gen.clip(max_bins-1),
                                bins=[bins, bins],
                                density=True)
    H_tr,  _x, _y = np.histogram2d(alpha_tr.clip(max_bins-1),
                                beta_tr.clip(max_bins-1),
                                bins=[bins, bins],
                                density=True)
    # add pseudocount to avoid log(0)
    H_gen += 1e-8
    H_tr  += 1e-8
    H_gen /= H_gen.sum()
    H_tr  /= H_tr.sum()
    # symmetric KL divergence (bits)
    kl = 0.5 * (
        np.sum(H_gen * np.log2(H_gen / H_tr)) +
        np.sum(H_tr  * np.log2(H_tr  / H_gen))
    )
    return {
        "ss_kl_bits"    : float(kl),
        "mean_alpha_gen": float(alpha_gen.mean()),
        "mean_beta_gen" : float(beta_gen.mean()),
        "mean_alpha_tr" : float(alpha_tr.mean()),
        "mean_beta_tr"  : float(beta_tr.mean()),
    }

def seq_identity(seq1: str, seq2: str) -> float:
    """Percent identity over min(len(seq1), len(seq2))"""
    L = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:L], seq2[:L]))
    return 100.0 * matches / L

# def designability_sequence_recovery(
#     pdb_paths: List[str],
#     sampler: Callable[..., List[str]] = generate_residues,
#     n_designs: int = 10,
#     sampler_kwargs: Dict = None,
# ) -> Dict[str, float]:
#     """
#     For each backbone pdb:
#     1) extract native sequence
#     2) generate N candidate sequences with `sampler`
#     3) compute percent identity of each candidate vs native
#     4) keep the BEST identity for that backbone
#     Returns a dict {pdb_basename : best_identity}
#     """
#     sampler_kwargs = sampler_kwargs or {}
#     best_id_dict = {}
#     for pdb in tqdm(pdb_paths, desc="Designability"):
#         native = extract_aa_seq(pdb)
#         # generate_residues() returns list[str]
#         cand_seqs = sampler(pdb, n=n_designs, **sampler_kwargs)
#         best_id = max(seq_identity(native, cand) for cand in cand_seqs)
#         best_id_dict[Path(pdb).stem] = best_id
#     return best_id_dict

# ---------------------------------------------------------------------
# a tiny wrapper to fold ONE sequence with OmegaFold & return PDB path
# ---------------------------------------------------------------------
def fold_seq_with_omegafold(seq: str, gpu_id: int = 0, weights: str = "") -> str:
    """Write `seq` to a temp fasta, fold with OmegaFold, return PDB filename."""
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "q.fasta"
        fasta.write_text(">Q\n" + seq + "\n")
        outdir = Path(tmp) / "pred"
        outdir.mkdir()
        run_omegafold(str(fasta), str(outdir), gpu=gpu_id, weights=weights)
        pred_pdb = next(outdir.glob("*.pdb"))
        # move the file out of tmp so it survives
        final = Path(tmp).with_suffix(".pdb")
        shutil.copy(str(pred_pdb), final)
        return str(final)

# ---------------------------------------------------------------------
# scTM computation
# ---------------------------------------------------------------------
def sctm_designability(
    backbone_pdbs: List[str],
    gpu_id: int = 0,
    tm_cutoff: float = 0.5,          # scTM ≥ 0.5 → designable
    n_designs: int = 8,
) -> Dict[str, float]:
    """
    For each backbone PDB:
      1. design N sequences with ProteinMPNN
      2. fold each with OmegaFold
      3. compute TM‑score (via TM‑align) vs. original backbone
      4. keep the max TM‑score  ->  scTM
    Returns dict {basename : scTM}.
    """
    results: Dict[str, float] = {}
    for bb in tqdm(backbone_pdbs, desc="scTM designability"):
        # 1) ProteinMPNN designs
        seqs = generate_residues_proteinmpnn(
            bb, n_sequences=n_designs, temperature=0.1
        )

        best_tm = 0.0
        for seq in seqs:
            pred_pdb = fold_seq_with_omegafold(seq, gpu_id=gpu_id)
            tm_res = run_tmalign_gdt_ts(pred_pdb, bb, fast=True)
            tm_score = max(tm_res["tm_score_chain_1"], tm_res["tm_score_chain_2"])
            best_tm = max(best_tm, tm_score)

        results[Path(bb).stem] = best_tm
    return results


def summarize_sctm(sc_tm_dict: Dict[str, float], cutoff: float = 0.5) -> Dict[str, float]:
    """Return mean scTM and fraction ≥ cutoff."""
    scores = np.array(list(sc_tm_dict.values()))
    return {
        "scTM_mean"      : float(scores.mean()),
        "designability_fraction" : float((scores >= cutoff).mean()),
    }


def compute_metrics(generated_pdb_paths, train_pdb_paths):
    # tm_vals = [
    #     best_tm_vs_train(gen_pdb, train_pdb_paths)
    #     for gen_pdb in generated_pdb_paths
    # ]
    # mean_best_tm = float(np.mean(tm_vals))   
    # tm_vals = [
    #     best_tm_vs_train(gen_pdb, train_pdb_paths)
    #     for gen_pdb in generated_pdb_paths
    # ]
    # mean_best_tm = float(np.mean(tm_vals))    
    # # --- Novelty (min‑RMSD to train) ---
    # novelty_vals = [
    #     min(compute_rmsd(g, t) for t in train_coords)
    #     for g in generated_coords
    # ]
    # mean_novelty = float(np.mean(novelty_vals))
    # # --- Diversity (mean pairwise RMSD) ---
    # pair_vals = [
    #     compute_rmsd(gi, gj)
    #     for gi, gj in itertools.combinations(generated_coords, 2)
    # ]
    # mean_diversity = float(np.mean(pair_vals))
    # # --- Uniqueness (fraction with nearest‑neighbor RMSD > τ) ---
    # τ = 1.0  # Å
    # unique = []
    # for g in generated_coords:
    #     if not unique:                      # first sample
    #         unique.append(g); continue
    #     if min(compute_rmsd(g,u) for u in unique) > τ:
    #         unique.append(g)
    # fraction_unique = len(unique) / len(generated_coords)    
    # lddt_vals = [
    #     best_lddt_vs_train(gen_pdb, train_pdb_paths)
    #     for gen_pdb in generated_pdb_paths
    # ]
    # mean_best_lddt = float(np.mean(lddt_vals)) 
    # # generated_seqs : list[str]  (amino‑acid sequences for the sampled backbones)
    # plddt_vecs = get_plddt_esmfold_batched(generated_seqs, batch_size=2)
    # mean_plddt = float(np.mean([v.mean().item() for v in plddt_vecs]))
    # histograms for generated vs. train  (2D 36×36 bins → 10° resolution)
    # bins = [np.linspace(-180, 180, 37), np.linspace(-180, 180, 37)]
    # gen_hist  = hist2d([phi_psi_angles(p) for p in generated_pdb_paths])
    # train_hist= hist2d([phi_psi_angles(p) for p in train_pdb_paths])
    # # symmetric KL divergence (bits)
    # kl = 0.5 * (
    #     np.sum(gen_hist * np.log(gen_hist/train_hist)) +
    #     np.sum(train_hist * np.log(train_hist/gen_hist))
    # )
    # ramach_kl = float(kl)  
    # ------------------------------------------------------------------
    # example usage
    # ------------------------------------------------------------------
    gdt_metrics = compute_gdt_ts_novelty(
        generated_pdbs   = generated_pdb_paths,   # list[str]
        train_pdbs       = train_pdb_paths,       # list[str]
        fast             = True                  # use TMalign -fast
    )  
    # -------------------------------------------------------------
    # example usage
    # -------------------------------------------------------------
    ss_metrics = ss_kl_divergence(
        generated_pdbs = generated_pdb_paths,   # list[str]
        train_pdbs     = train_pdb_paths,       # list[str]
        max_bins       = 8,                     # same binning as make_ss_cooccurrence_plot
        backend        = "psea",
    )    
    # -----------------------------------------------------------
    # Example usage
    # -----------------------------------------------------------
    # best_identity = designability_sequence_recovery(
    #     pdb_paths       = generated_pdb_paths,     # list of generated backbones
    #     sampler         = generate_residues,       # or generate_residues_proteinmpnn
    #     n_designs       = 10,
    #     sampler_kwargs  = dict(temperature=1.0)    # forwarded to sampler
    # )
    # mean_best_identity = float(np.mean(list(best_identity.values())))
    # print("Designability (sequence recovery):", mean_best_identity, "%")    
    sc_tm_per_backbone = sctm_designability(
        backbone_pdbs = generated_pdb_paths,  # list of PDBs you generated
        gpu_id        = 0,                    # choose GPU
        tm_cutoff     = 0.5,
        n_designs     = 8,
    )
    metrics = {
        "novelty_rmsd"     : mean_novelty,
        "diversity_rmsd"   : mean_diversity,
        "uniqueness_frac"  : fraction_unique,
        "best_tm_mean"     : mean_best_tm,
        "best_lddt_mean"   : mean_best_lddt,
        "mean_plddt"       : mean_plddt,
        "ramach_kl_bits"   : ramach_kl,
        "gdt_ts_mean_best": gdt_metrics["gdt_ts_mean_best"]
    } | ss_metrics | sc_tm_per_backbone
    # metrics["seq_recovery_best_mean"] = mean_best_identity


if __name__ == "__main__":
    refset = [np.random.randn(5, 3), np.random.randn(10, 3)]
    genset = [np.random.randn(5, 3), np.random.randn(10, 3)]    
    genfile_dir = os.path.abspath("../ckpts/1752523675.4143364/sampled_pdb")
    trainfile_dir = os.path.abspath("../data/struct_token_bench/interpro/conserved/")
    genfiles = [Path(f"{genfile_dir}/generated_0.pdb"), Path(f"{genfile_dir}/generated_1.pdb")]
    trainfiles = [Path(f"{trainfile_dir}/12ca_A.pdb")]
    breakpoint()
    compute_metrics(genfiles, trainfiles)
