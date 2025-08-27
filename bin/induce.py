import argparse
import pickle
import time
import shutil
import os
import sys
from tqdm import tqdm
from pathlib import Path
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.tokenizer import Tokenizer
from concurrent.futures import ProcessPoolExecutor
from foldingdiff.utils import load_args_from_txt, validate_args_match

def _effective_cpus() -> int:
    """Return the number of CPUs *actually* available to this task."""
    if "SLURM_CPUS_PER_TASK" in os.environ:        
        n = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"SLURM_CPUS_PER_TASK={n}")
        return n
    try:                                   # Linux cpuset / cgroups
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff Induction Script")
    # folder
    parser.add_argument("--src-pkl")
    parser.add_argument("--base-dir", type=str, default="./")
    parser.add_argument("--save-dir")
    parser.add_argument("--data-dir")
    # data params
    parser.add_argument("--toy", default=0, type=int)
    parser.add_argument("--pad", default=512, type=int)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    return args


def _init_tokenize_worker(bpe, save_dir):
    global BPE, SAVE_DIR
    BPE = bpe
    SAVE_DIR = save_dir


def tokenize_structure(args):
    """Build and tokenize a single structure inside the worker."""
    idx, struc = args
    try:
        pickle.load(open(SAVE_DIR / f"{idx}.pkl", "rb"))
        return
    except:
        print('start tokenize_structure')    
    tok = Tokenizer(struc)
    BPE.tokenize(tok)
    pickle.dump(tok, open(SAVE_DIR / f"{idx}.pkl", "wb+"))


def main():
    args = parse_args()
    if args.debug:
        max_workers = 0
    else:
        max_workers = _effective_cpus()
    print(f"{max_workers} workers")
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        cur_time = time.time()
        save_dir = Path(args.base_dir) / f'./ckpts/{cur_time}'
        os.makedirs(save_dir, exist_ok=True)        
    
    print(f"save_dir: {save_dir}")
    src_file = Path(args.src_pkl)
    bpe = pickle.load(open(src_file, "rb"))
    # store metadata
    arg_path = src_file.parent / "args.txt"
    args_path = save_dir / "args.txt"
    out_path = save_dir / src_file.name
    shutil.copyfile(arg_path, save_dir / "orig_args.txt")
    if os.path.exists(args_path):
        print(f"loading args from {args_path}")
        loaded_args = load_args_from_txt(args_path)    
        validate_args_match(
            current   = args,
            loaded    = loaded_args,
            skip = ["auto", "save_dir"]
        )
    else:
        with open(args_path, "w") as f:
            for arg_name, arg_value in sorted(args.__dict__.items()):
                f.write(f"{arg_name}: {arg_value}\n")    
    # induce
    dataset = FullCathCanonicalCoordsDataset(args.data_dir, 
                                        zero_center=False, 
                                        toy=args.toy, 
                                        pad=args.pad)    
    pargs = []
    idx = 0
    for struc in dataset.structures:
        if (struc['angles']['psi']==struc['angles']['psi']).sum() < len(struc['angles']['psi'])-1:
            print(f"skipping {i}, {struc['fname']} because of missing dihedrals")
        else:
            pargs.append((idx, struc))
            idx += 1

    N = len(pargs)
    tokenizers = []
    if max_workers:
        # ----- parallel tokenisation -----------------------------------------------------    
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_tokenize_worker,
            initargs=(bpe, save_dir)                                # only BPE is broadcast
        ) as pool:
            for _ in tqdm(pool.map(
                    tokenize_structure, pargs, chunksize=1),
                    total=N, desc="tokenizing"):
                pass        
        for i in tqdm(range(N), desc="loading done tokenizers"):
            assert os.path.exists(save_dir / f"{i}.pkl")
            tokenizers.append(pickle.load(open(save_dir / f"{i}.pkl", "rb")))                
    else:
        for idx, struc in pargs:
            tok = Tokenizer(struc)
            bpe.tokenize(tok)
            tokenizers.append(tok)
        
    bpe.tokenizers = tokenizers
    pickle.dump(bpe, open(out_path, "wb+"))
    print(os.path.abspath(out_path))


if __name__ == "__main__":
    main()
