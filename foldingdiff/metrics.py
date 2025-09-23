from pathlib import Path
import multiprocessing as mp
import pandas as pd
import os, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    mp.set_start_method("spawn", force=True)   # avoid fork-after-init deadlocks
except RuntimeError:
    pass
print(f"[child] pid={os.getpid()} CVD={os.getenv('CUDA_VISIBLE_DEVICES')} args={sys.argv}", flush=True)
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import logging
import warnings
from foldingdiff.datasets import extract_backbone_coords
from foldingdiff.algo import *
from typing import *
from tqdm import tqdm
from itertools import combinations
import tempfile
import shutil
import glob
import json
import argparse
import functools
import subprocess
from foldingdiff.angles_and_coords import create_new_chain_nerf
from foldingdiff.tmalign import run_tmalign, max_tm_across_refs
from foldingdiff.lddt import lddt
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES, EXHAUSTIVE_DISTS
from foldingdiff.utils import str2bool
from itertools import groupby, product
from collections import defaultdict
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from bin.pdb_to_residue_proteinmpnn import PROTEINMPNN_SCRIPT, read_fasta, generate_residues_proteinmpnn
from bin.sample import write_preds_pdb_folder
from bin.annot_secondary_structures import count_structures_in_pdb
from bin.omegafold_across_gpus import my_run_omegafold_with_env
cwd = Path(__file__).parents[1]


def best_tm_vs_train(pdb, train_pdbs):
    best_tm, _ = max_tm_across_refs(pdb, train_pdbs, parallel=True)
    return best_tm


def tm_score(p1, p2):
    res = run_tmalign(p1, p2, fast=True, dual=True)
    return max(res["Chain_1"], res["Chain_2"])


def best_lddt_vs_train(pdb, train_pdbs):
    return max(lddt(pdb, ref) for ref in train_pdbs)


def phi_psi_angles(pdb_path):
    # returns (N,2) array of [φ,ψ] in degrees
    df = canonical_distances_and_dihedrals(str(pdb_path), distances=EXHAUSTIVE_DISTS, angles=EXHAUSTIVE_ANGLES)
    if df is None:
        return None
    return df[["phi", "psi"]].dropna().values


def hist2d(dataset, bins):
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



def ss_counts(pdb_paths, backend="psea", n_threads=8):
    """
    Return a list of (n_alpha, n_beta) tuples for each pdb in pdb_paths,
    skipping multichain structures (count_structures_in_pdb returns (-1,-1)).
    """
    with mp.Pool(n_threads) as pool:
        counts = list(
            pool.starmap(
                count_structures_in_pdb,
                [(p, backend) for p in pdb_paths],
                chunksize=10,
            )
        )
    # counts = [count_structures_in_pdb(str(p), backend) for p in pdb_paths]
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
def fold_seqs_with_omegafold(
    seqs: list[str],
    gpu_id: int = 0,
    weights: str = "",
    env_name: str = "omegafold_env",
) -> list[str]:
    """
    Fold a list of sequences with OmegaFold as a batch. 
    Returns: list of output PDB filenames (in order matching input seqs).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "batch.fasta"
        # Write all sequences to a single fasta
        fasta.write_text(
            "".join([f">Q{i}\n{s}\n" for i, s in enumerate(seqs)])
        )
        outdir = Path(tmp) / "pred"
        outdir.mkdir()
        # Run OmegaFold once for all sequences
        my_run_omegafold_with_env(str(fasta), str(outdir), gpu=gpu_id, weights=weights, env_name=env_name)
        pdbs = sorted(outdir.glob("*.pdb"))   # sorted by filename = by index
        # Move all pdbs out of tmp so they survive context exit
        finals = []
        for i, pdb in enumerate(pdbs):
            final = Path(tmp).with_suffix(f".{i}.pdb")
            shutil.copy(str(pdb), final)
            finals.append(str(final))
        return finals


def run_esmfold_batch(
    generated_seqs, 
    batch_size=2, 
    esmfold_env="esmfold", 
    out_type="pdb"  # or "plddt", etc., depending on your downstream use
):
    """
    Batch-predict structures or confidences for sequences via esmfold external script.
    Returns: (ids, outputs)
    """
    # 1. Assign unique IDs
    ids = [f"sample_{i}" for i in range(len(generated_seqs))]

    # 2. Write sequences to temp input
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as in_f:
        for seq in generated_seqs:
            in_f.write(seq + '\n')
        in_path = in_f.name

    # 3. Prepare output file path
    out_fd, out_path = tempfile.mkstemp(suffix='.pt' if out_type == "plddt" else '.txt')
    os.close(out_fd)

    # 4. Find the script (as in previous code, assumed at scripts/get_plddt.py)
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "get_plddt.py")

    # 5. Run ESMFold in the conda environment
    try:
        subprocess.run([
            "conda", "run", "-n", esmfold_env,
            "python", script_path,
            "--in-file", in_path,
            "--out-file", out_path
        ], check=True)
        # 6. Read output
        if out_type == "plddt":
            outputs = torch.load(out_path)
        else:
            # For PDB/text output, read as appropriate
            with open(out_path, "r") as f:
                outputs = f.read()
    except subprocess.CalledProcessError as e:
        print("Command failed with exit code", e.returncode)
        print("=== STDOUT ===")
        print(e.stdout)
        print("=== STDERR ===")
        print(e.stderr)
        outputs = None
    finally:
        # 7. Clean up
        os.remove(in_path)
        os.remove(out_path)

    return ids, outputs   


def _visible_device_tokens() -> List[str]:
    """
    Return the list of CUDA device tokens visible to this job.
    Handles both numeric indices (e.g., "0,1,2") and MIG UUIDs
    (e.g., "MIG-...,...").
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # preserve MIG UUIDs verbatim; trim whitespace
        return [tok.strip() for tok in cvd.split(",") if tok.strip()]

    # Fallback: sniff via nvidia-smi -L (handles MIG & non-MIG)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except Exception:
        return ["0"]  # last resort: assume a single device

    mig = re.findall(r"(MIG-[A-Za-z0-9\-]+)", out)
    if mig:
        return mig  # each MIG UUID is a device token
    # else: count physical GPUs and return "0","1",...
    gpus = re.findall(r"GPU \d+:", out)
    return [str(i) for i in range(len(gpus))] or ["0"]

def _shard_round_robin(items: List[str], n: int) -> List[List[str]]:
    shards = [[] for _ in range(n)]
    for i, it in enumerate(items):
        shards[i % n].append(it)
    return shards


def sctm_designability(
    backbone_pdbs: List[str],
    gpu_id: int = 0,
    tm_cutoff: float = 0.5,          # scTM ≥ 0.5 → designable
    n_designs: int = 8,
) -> Dict[str, float]:
    """
    For each backbone PDB:
      1. design N sequences with ProteinMPNN
      2. batch fold with OmegaFold
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

        # 2) Fold in batch (new)
        pred_pdbs = fold_seqs_with_omegafold(seqs, gpu_id=gpu_id)

        # 3) Score
        best_tm = 0.0
        for pred_pdb in pred_pdbs:
            score = tm_score(pred_pdb, bb)
            if score == score:
                best_tm = max(best_tm, score)
                print(best_tm)

        results[Path(bb).stem] = best_tm
    return results

# --- worker ---

def _worker_shard(args: Tuple[List[str], str, float, int]) -> Dict[str, float]:
    """
    args: (pdb_paths_for_this_worker, device_token, tm_cutoff, n_designs)
    device_token may be '0' or a MIG UUID like 'MIG-xxxx'.
    """
    shard, device_token, tm_cutoff, n_designs = args

    # Pin this process to exactly one device (MIG UUID or index)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_token
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    # Inside this process, 'cuda:0' == the token we just exposed.
    return sctm_designability(
        shard,
        gpu_id=0,
        tm_cutoff=tm_cutoff,
        n_designs=n_designs,
    )

# --- public API ---

def parallel_sctm_designability(
    backbone_pdbs: List[str],
    n_devices: int | None = None,
    tm_cutoff: float = 0.5,
    n_designs: int = 8,
) -> Dict[str, float]:
    """
    Shard PDBs across all visible CUDA devices (MIG-aware) and run SCTM in parallel.
    """
    tokens = _visible_device_tokens()
    if n_devices is not None:
        tokens = tokens[:n_devices]
    if not tokens:
        raise RuntimeError("No CUDA devices visible to this job")

    shards = _shard_round_robin(backbone_pdbs, len(tokens))
    tasks = [(shard, tok, tm_cutoff, n_designs)
             for shard, tok in zip(shards, tokens) if shard]

    results: Dict[str, float] = {}
    # Use 'spawn' to avoid accidental CUDA context inheritance
    with ProcessPoolExecutor(max_workers=len(tasks),
                             mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(_worker_shard, t) for t in tasks]
        for fut in as_completed(futures):
            results.update(fut.result())
            print(f"{len(results)}/{len(tasks)} done")
    return results


def summarize_sctm(sc_tm_dict: Dict[str, float], cutoff: float = 0.5) -> Dict[str, float]:
    """Return mean scTM and fraction ≥ cutoff."""
    scores = np.array(list(sc_tm_dict.values()))
    return {
        "scTM_mean"      : float(scores.mean()),
        "designability_fraction" : float((scores >= cutoff).mean()),
    }

def compute_metrics(generated_pdb_paths, generated_coords, train_pdb_paths=None, train_coords=None):
    assert (train_pdb_paths is not None) == (train_coords is not None)
    train_avail = (train_pdb_paths is not None)
    metrics = {}
    if train_avail:
        tm_vals = [
            best_tm_vs_train(gen_pdb, train_pdb_paths)
            for gen_pdb in generated_pdb_paths
        ]
        mean_novelty_tm = float(np.mean(tm_vals))    
        logging.info(f"mean_novelty_tm: {mean_novelty_tm}")
        metrics["mean_novelty_tm"] = mean_novelty_tm    
        # # --- Novelty (min‑RMSD to train) ---
        # pairs = list(product(generated_pdb_paths, train_pdb_paths))
        # with mp.Pool(mp.cpu_count()) as pool:
        #     scores = pool.starmap(tm_score, tqdm(pairs, desc="novelty"), chunksize=1)    
        # scores_by_g = defaultdict(list)
        # for (g, t), s in zip(pairs, scores):
        #     scores_by_g[g].append(s)
        # novelty_vals = [min(scores_by_g[g]) for g in generated_pdb_paths]
        # mean_novelty_tm = float(np.mean(novelty_vals))
        # logging.info(f"mean_novelty_tm: {mean_novelty_tm}")
        # metrics["mean_novelty_tm"] = mean_novelty_tm    
        # --- LDDT ---
        # pairs = list(product(generated_pdb_paths, train_pdb_paths))
        # with mp.Pool(mp.cpu_count()) as pool:
        #     lddt_scores = pool.starmap(lddt, tqdm(pairs, desc="lddt"), chunksize=1)
        # lddt_by_gen = defaultdict(list)
        # for (gen, train), sc in zip(pairs, lddt_scores):
        #     lddt_by_gen[gen].append(sc)
        # lddt_vals = [max(lddt_by_gen[g]) for g in generated_pdb_paths]
        # mean_best_lddt = float(np.mean(lddt_vals))         
        # --- Histograms for generated vs. train  (2D 36×36 bins → 10° resolution) ---
        bins = [np.linspace(-180, 180, 37), np.linspace(-180, 180, 37)]
        gen_hist  = hist2d([res for res in (phi_psi_angles(p) for p in generated_pdb_paths) if res is not None], bins=bins)
        train_hist = hist2d([res for res in (phi_psi_angles(p) for p in train_pdb_paths) if res is not None], bins=bins)
        # symmetric KL divergence (bits)
        kl = 0.5 * (
            np.sum(gen_hist * np.log(gen_hist/train_hist)) +
            np.sum(train_hist * np.log(train_hist/gen_hist))
        )
        ramach_kl = float(kl)  
        logging.info(f"ramach_kl: {ramach_kl}")
        metrics["ramach_kl_bits"] = ramach_kl
        # gdt_metrics = compute_gdt_ts_novelty(
        #     generated_pdbs   = generated_pdb_paths,   # list[str]
        #     train_pdbs       = train_pdb_paths,       # list[str]
        #     fast             = True                  # use TMalign -fast
        # )
        ss_metrics = ss_kl_divergence(
            generated_pdbs = generated_pdb_paths,   # list[str]
            train_pdbs     = train_pdb_paths,       # list[str]
            max_bins       = 2,                     # same binning as make_ss_cooccurrence_plot
            backend        = "psea",
        )
        logging.info(f"ss_metrics: {ss_metrics}")
        metrics.update(ss_metrics)        
    # --- Diversity (mean pairwise RMSD) ---
    pairs = list(combinations(generated_pdb_paths, 2))
    with mp.Pool(mp.cpu_count()) as pool:
        div_vals = pool.starmap(tm_score, tqdm(pairs, desc="diversity"), chunksize=1)
    mean_diversity_tm = float(np.mean(div_vals))
    logging.info(f"mean_diversity_tm: {mean_diversity_tm}")
    metrics["diversity_tm"] = mean_diversity_tm
    # --- Uniqueness (fraction with nearest‑neighbor RMSD > τ) ---
    unique = []
    for g in generated_pdb_paths:
        if not unique: unique.append(g); continue
        best_tm = max(tm_score(g, u) for u in unique)
        if best_tm < 0.5:          # below fold‑level similarity
            unique.append(g)
    fraction_unique_tm = len(unique) / len(generated_pdb_paths)
    logging.info(f"fraction_unique_tm: {fraction_unique_tm}")
    metrics["uniqueness_frac_tm"] = fraction_unique_tm
    # logging.info(f"mean_best_lddt: {mean_best_lddt}")
    # # generated_seqs : list[str]  (amino‑acid sequences for the sampled backbones)
    # plddt_vecs = run_esmfold_batch(generated_seqs)
    # mean_plddt = float(np.mean([v.mean().item() for v in plddt_vecs]))
      
    # best_identity = designability_sequence_recovery(
    #     pdb_paths       = generated_pdb_paths,     # list of generated backbones
    #     sampler         = generate_residues,       # or generate_residues_proteinmpnn
    #     n_designs       = 10,
    #     sampler_kwargs  = dict(temperature=1.0)    # forwarded to sampler
    # )
    # mean_best_identity = float(np.mean(list(best_identity.values())))
    # print("Designability (sequence recovery):", mean_best_identity, "%")    
    # metrics.update({
        # "best_lddt_mean"   : mean_best_lddt,
        # "mean_plddt"       : mean_plddt,        
        # "gdt_ts_mean_best": gdt_metrics["gdt_ts_mean_best"]
    # })        
    # metrics["seq_recovery_best_mean"] = mean_best_identity
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone Metrics")
    parser.add_argument("--gen-pdb-dir")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--sctm", default=True, type=str2bool)
    # target sctm directly
    parser.add_argument("--pdb-list", help="Text file with one PDB path per line")
    parser.add_argument("--n-gpus", type=int, default=None)
    args = parser.parse_args()
    if not (args.gen_pdb_dir) and not (args.pdb_list and args.n_gpus):
        parser.error("must provide gen-pdb-dir or pdb-list and n-gpus")    
    return args


def main(args):
    if args.gen_pdb_dir:
        gen_pdb_files = glob.glob(f"{args.gen_pdb_dir}/*.pdb")
        if args.sctm:
            sc_tm_per_backbone = sctm_designability(
                backbone_pdbs = gen_pdb_files,  # list of PDBs you generated
                gpu_id        = 0,                    # choose GPU
                tm_cutoff     = 0.5,
                n_designs     = 8,
            )
            sctm_metrics = summarize_sctm(sc_tm_per_backbone)
            logging.info(f"sctm_metrics: {sctm_metrics}")
            metrics = sctm_metrics
        else:   
            full_coords_pfunc = functools.partial(extract_backbone_coords, atoms=["N", "CA", "C"])
            pool = pool = mp.Pool(processes=mp.cpu_count())
            generated_coords = pool.map(full_coords_pfunc, tqdm(gen_pdb_files, desc="extract coords gen"))                 
            metrics = compute_metrics(gen_pdb_files, generated_coords)
        json.dump(metrics, open(args.out_file, "w+"))
    else:
        pdbs = [ln.strip() for ln in open(args.pdb_list) if ln.strip()]
        metrics = parallel_sctm_designability(
            pdbs, n_devices=args.n_gpus, tm_cutoff=0.5, n_designs=8
        )
    json.dump(metrics, open(args.out_file, "w+"))


if __name__ == "__main__":
    args = parse_args()  
    main(args)
    # refset = [np.random.randn(5, 3), np.random.randn(10, 3)]
    # genset = [np.random.randn(5, 3), np.random.randn(10, 3)]    
    # genfile_dir = os.path.abspath("ckpts/1752523675.4143364/sampled_pdb")
    # trainfile_dir = os.path.abspath("data/struct_token_bench/interpro/conserved/")
    # genfiles = [f"{genfile_dir}/generated_0.pdb", f"{genfile_dir}/generated_1.pdb"]
    # trainfiles = [f"{trainfile_dir}/12ca_A.pdb", f"{trainfile_dir}/1a03_A.pdb"]
    # breakpoint()
    # metrics = compute_metrics(genfiles, trainfiles, refset, genset)
    # print(metrics)
