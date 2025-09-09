"""
Short script to parallelize omegafold across GPUs to speed up runtime.
https://github.com/HeliXonProtein/OmegaFold
"""

import os
import logging
import argparse
import subprocess
import shutil
import multiprocessing as mp
from typing import *

from tqdm import tqdm
import torch
import numpy as np
from biotite.sequence import ProteinSequence, AlphabetError


def read_fasta(fname: str, check_valid: bool = True) -> Dict[str, str]:
    """Read the sequences in the fasta to a dict"""

    def add_seq_if_valid(d: Dict[str, str], k: str, v: str) -> None:
        """Add v to d[k] if v is a valid sequence"""
        if not check_valid:
            d[k] = v
            return
        try:
            _ = ProteinSequence(v)
            d[k] = v
        except AlphabetError:
            logging.warning(f"Illegal character in entry {k}: {v} | skipping")

    retval = {}
    curr_k, curr_v = "", ""
    with open(fname) as source:
        for line in source:
            if line.startswith(">"):
                if curr_k:  # Record and reset
                    assert curr_v
                    assert curr_k not in retval, f"Duplicated fasta entry: {curr_k}"
                    add_seq_if_valid(retval, curr_k, curr_v)
                curr_k = line.strip().strip(">")
                curr_v = ""
            else:
                curr_v += line.strip()
    # Write the last sequence
    assert curr_k
    assert curr_v
    add_seq_if_valid(retval, curr_k, curr_v)
    return retval


def write_fasta(sequences: Dict[str, str], out_fname: str):
    """Write the sequeces to the given fasta file"""
    with open(out_fname, "w") as sink:
        for k, v in sequences.items():
            sink.write(f">{k}\n")
            for segment in [v[i : i + 80] for i in range(0, len(v), 80)]:
                sink.write(segment + "\n")


def run_omegafold(input_fasta: str, outdir: str, gpu: int, weights: str = ""):
    """
    Runs omegafold on the given fasta file
    """
    logging.info(
        f"Running omegafold on {input_fasta} > {outdir} with gpu {gpu} with weights {weights}"
    )
    assert shutil.which("omegafold")
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} omegafold {input_fasta} {outdir} --device cuda:0"
    if weights:
        assert os.path.isfile(weights)
        cmd += f" --weights_file {weights}"

    bname = os.path.splitext(os.path.basename(input_fasta))[0]
    with open(
        os.path.join(outdir, f"omegafold_{bname}_gpu_{gpu}.stdout"), "wb"
    ) as sink:
        output = subprocess.call(cmd, shell=True, stdout=sink)


def run_omegafold_with_env(
    input_fasta: str,
    outdir: str,
    gpu: int,
    weights: str = "",
    env_name: str = "omegafold_env",
):
    """
    Runs OmegaFold inside the specified conda environment (`env_name`)
    on the given FASTA file, writing predictions to `outdir`.

    Parameters
    ----------
    input_fasta : str
        Path to a FASTA file containing one or more sequences.
    outdir : str
        Directory where OmegaFold will write its PDB predictions.
    gpu : int
        Which CUDA‐visible GPU index to use.
    weights : str, optional
        Path to a custom weights file for OmegaFold.
    env_name : str, optional
        Name of the conda env where OmegaFold is installed.
    """
    log_msg = (
        f"OmegaFold ({env_name}) : {input_fasta} → {outdir}  "
        f"[GPU {gpu}]  weights={weights or 'default'}"
    )
    logging.info(log_msg)

    # Ensure `conda run` is available
    if shutil.which("conda") is None:
        raise RuntimeError("`conda` executable not found in PATH")

    # Build the OmegaFold CLI arguments
    of_cmd = [
        "omegafold",
        input_fasta,
        outdir,
        "--device",
        "cuda:0",
    ]
    if weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Weights file {weights} not found")
        of_cmd += ["--weights_file", weights]

    # Wrap in `conda run` and set GPU via env var
    full_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + of_cmd
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Log stdout/stderr to file
    bname = os.path.splitext(os.path.basename(input_fasta))[0]
    stdout_path = os.path.join(outdir, f"omegafold_{bname}_gpu{gpu}.stdout")
    with open(stdout_path, "wb") as sink:
        subprocess.check_call(full_cmd, env=env, stdout=sink, stderr=subprocess.STDOUT)

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
        run_omegafold_with_env(str(fasta), str(outdir), gpu=gpu_id, weights=weights, env_name=env_name)
        pdbs = sorted(outdir.glob("*.pdb"))   # sorted by filename = by index
        # Move all pdbs out of tmp so they survive context exit
        finals = []
        for i, pdb in enumerate(pdbs):
            final = Path(tmp).with_suffix(f".{i}.pdb")
            shutil.copy(str(pdb), final)
            finals.append(str(final))
        return finals


def build_parser() -> argparse.ArgumentParser:
    """
    Build a basic CLI
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "fastafile",
        type=str,
        nargs="*",
        help="Fasta file(s) containing sequences to run. If multiple given, all will be read in.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.abspath(os.path.join(os.getcwd(), "omegafold_predictions")),
        help="Output directory, create if doesn't exist",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        default="",
        help="Path to weights file for omegafold, optional",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        nargs="*",
        default=list(range(torch.cuda.device_count())),
        help="GPUs to use",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=False)

    # Read in input files
    input_sequences = {}
    for fname in args.fastafile:
        fname_seqs = read_fasta(fname)
        assert fname_seqs.keys().isdisjoint(input_sequences.keys())
        input_sequences.update(fname_seqs)
    n = len(input_sequences)
    logging.info(f"Parsed {n} sequences")

    # Divide up the inputs, shuffling their indices su that the load is spread
    # across GPUs; otherwise, if we just give them in order, the first GPU will
    # finish first since it has shorter sequences.
    idx = np.arange(n)
    rng = np.random.default_rng(seed=1234)
    rng.shuffle(idx)
    idx_split = np.array_split(idx, len(args.gpus))
    all_keys = list(input_sequences.keys())
    all_keys_split = [[all_keys[i] for i in part] for part in idx_split]
    # Write the tempfiles and call omegafold

    processes = []
    for i, key_chunk in enumerate(all_keys_split):
        fasta_filename = os.path.join(args.outdir, f"{i}_omegafold_input.fasta")
        assert not os.path.exists(fasta_filename)
        logging.info(f"Writing {len(key_chunk)} sequences to {fasta_filename}")
        write_fasta({k: input_sequences[k] for k in key_chunk}, fasta_filename)
        proc = mp.Process(
            target=run_omegafold,
            args=(fasta_filename, args.outdir, args.gpus[i], args.weights),
        )
        proc.start()
    for p in processes:
        p.join()


# Example CLI
if __name__ == "__main__":
    main()
