import argparse
import concurrent.futures
import subprocess
import pickle
import os, sys, glob
import pymol
from pymol import cmd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff Induction Script")
    # folder
    parser.add_argument("--pkl-path")
    parser.add_argument("--pdb-dir")
    args = parser.parse_args()
    if args.pdb_dir and args.pkl_path:
        parser.error("only provide one of pkl-path or pdb-dir")
    return args

def recurse(node):
    return [node.value] + (recurse(node.left) if node.left else []) + (recurse(node.right) if node.right else [])


def _build_pymol_cmd(t, i):
    """Return the single-line PyMOL command for tokenizer t / index i."""
    start, _, length = max(t.bond_to_token.values(), key=lambda x: x[2])
    node = t.bond_to_token.tree.nodes[start]
    cmds = []
    for (start, _, length) in recurse(node):
        start, length = start//3, length//3
        chain  = Path(t.fname).stem.split('_')[1]
        outpng = f"{Path(t.fname).stem}_{start}_{length}.png"
        cmds.append(
            "pymol -cq "
            "-d 'run highlight_span.py' "
            f"-d 'highlight_span {t.fname}, {start}, {start+length}, {outpng}, {chain}'"
        )
    return cmds

def _run_batch(batch):
    """
    Worker helper: activate conda env `pstbench` once, then execute
    every PyMOL command in *batch* sequentially.
    `batch` is a list of (tokenizer, index) tuples.
    """
    # create the shell lines
    lines = [
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate pstbench",
    ] + sum([_build_pymol_cmd(t, i) for t, i in batch], [])
    # run them in a single shell so activation is done only once
    script = " && ".join(lines)
    subprocess.run(["bash", "-c", script], check=True)

# --------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.pkl_path:
        bpe = pickle.load(open(args.pkl_path, "rb"))

        # pick the ten most interesting tokenizers (same logic as before)
        selected = sorted(
            [(i,
            max(l for (_, _, l) in t.bond_to_token.values()),
            -len(t.bond_to_token) / t.n)
            for i, t in enumerate(bpe.tokenizers)],
            key=lambda x: x[1:]
        )[-10:]

        tasks = [(bpe.tokenizers[i], i) for i, *_ in selected]

        # split the 10 tasks across a handful of workers
        n_workers   = min(4, len(tasks))                 # tweak as you like
        chunk_size  = (len(tasks) + n_workers - 1) // n_workers
        batches     = [tasks[k:k + chunk_size] for k in range(0, len(tasks), chunk_size)]

        # run the batches in parallel
        # with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            # pool.map(_run_batch, batches)
        for batch in batches:
            _run_batch(batch)
    else:
        for f in glob.glob(f"{args.pdb_dir}/*.pdb"):
            backbone_render_core(f, f.replace(".pdb", ".png"))

def _save_png(path, width=2000, height=1500, force_no_ray=False):
    path = os.path.abspath(os.path.expanduser(path))
    d = os.path.dirname(path) or "."
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    try:
        if force_no_ray:
            raise RuntimeError("noray")
        cmd.png(path, width=width, height=height, ray=1)
        if not os.path.exists(path):
            raise RuntimeError("ray PNG not created")
        print(f"[OK] Ray-traced PNG saved: {path}")
        return
    except Exception as e:
        print(f"[INFO] Ray render unavailable/failed ({e}); saving without ray…")
        cmd.png(path, width=width, height=height, ray=0)
        if not os.path.exists(path):
            raise RuntimeError("non-ray PNG not created")
        print(f"[OK] PNG saved (no ray): {path}")

def backbone_render_core(input_pdb_or_id, output_png, force_no_ray=False):
    obj = "mol"

    # Load local file or fetch by 4-letter PDB ID
    if os.path.exists(input_pdb_or_id):
        cmd.load(input_pdb_or_id, obj)
    else:
        code = str(input_pdb_or_id).strip()
        if len(code) == 4 and code.isalnum():
            cmd.fetch(code, obj)
        else:
            print(f"[ERROR] '{input_pdb_or_id}' is not a file or 4-letter PDB ID.")
            return

    # Minimal style: plain cartoon on white
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.hide("everything", obj)
    cmd.show("cartoon", obj)
    cmd.color("gray70", obj)

    cmd.orient(obj)
    cmd.zoom(obj, 1.0)

    _save_png(output_png, force_no_ray=force_no_ray)


if __name__ == "__main__":
    main()