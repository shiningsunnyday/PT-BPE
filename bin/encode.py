from tqdm import tqdm
import tempfile
import imageio
import os
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.bpe import *
from foldingdiff.plotting import *
from foldingdiff.utils import load_args_from_txt, validate_args_match
import scipy.io
import numpy as np
import subprocess
import argparse
import pickle
from datetime import datetime


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().info("Logger initialized.")

def get_logger():
    """Helper to retrieve the global logger."""
    return LOGGER


def parse_pdb(pdb_file):
    # Create a PDB parser object
    parser = PDBParser(QUIET=True)

    # Path to your PDB file (e.g., '12asA00.pdb')
    structure = parser.get_structure("protein", pdb_file)

    # We'll store coordinates for each residue as a tuple: (N, CA, C)
    backbone_coords = []

    # Iterate over all residues in all chains
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check that the residue has the backbone atoms we need.
                if all(atom_name in residue for atom_name in ['N', 'CA', 'C']):
                    # Extract coordinates
                    N_coord = residue['N'].get_coord()
                    CA_coord = residue['CA'].get_coord()
                    C_coord = residue['C'].get_coord()
                    backbone_coords.append((N_coord, CA_coord, C_coord))

    # Now, backbone_coords is a list of tuples, each containing three numpy arrays of shape (3,).
    # For a protein with N residues, you have N entries, corresponding to 3 x 3D coordinates.
    for i, (N_coord, CA_coord, C_coord) in enumerate(backbone_coords, start=1):
        print(f"Residue {i}:")
        print(f"  N:  {N_coord}")
        print(f"  CA: {CA_coord}")
        print(f"  C:  {C_coord}")

    return backbone_coords

def call_freqgeo(G):
    scipy.io.savemat('./data/cath/graphs.mat', {"G": G})

    breakpoint()
    try:
        result = subprocess.run(
            ["matlab", "-batch", "testmexfreqgeo"],
            cwd="./freqgeo-1.0/src",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print("MATLAB finished successfully.")
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("MATLAB failed with exit code", e.returncode)
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        
    res = scipy.io.loadmat("./freqgeo-1.0/src/mexfreqgeo_output.mat")
    count = res['count']
    graphs = res['graphs']
    occurence = res['occurence']

    for i in range(len(graphs[0])):
        g = graphs[0, i][0, 0]
        nodelabels, nodepos, edges = g
        n = len(nodelabels)
        breakpoint()    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def str2dict(v):
    m = re.match('\d+-\d+(?::\d+-\d+)*$', v)
    if not m:
        raise argparse.ArgumentTypeError("Wrong format, see help.")
    pairs = re.findall(r'(\d+)-(\d+)', v)        
    bins = {}
    for (a, b) in pairs:
        bins[int(a)] = int(b)
    return bins

def str2dictorint(v):
    if v.isdigit():
        return int(v)
    else:
        return str2dict(v)

def int_or_inf(x: str):
    # allow case‐insensitive “inf”
    if x.lower() in ("inf", "infinity"):
        return float("inf")
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"‘{x}’ is not an integer or ‘inf’")
    
def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    # folder
    parser.add_argument("--auto", action='store_true', help='auto set folders')
    parser.add_argument("--base-dir", type=str, default="./")
    parser.add_argument("--save-dir", type=str, 
                        help="Directory to save output files (images, pdb files, plots, etc.).")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")
    parser.add_argument("--data-dir", type=str, default="cath", help="""Which dataset. Suggested choices:  \
                        'cath', 'homo', 'ec', 'bindint', 'bindbio', 'repeat', 'catint', 'catbio', 'conserved'
                        """)
    parser.add_argument("--toy", type=int, default=0, 
                            help="Number of PDB files. 0 for all.")    
    parser.add_argument("--pad", type=int, default=512, help="Max protein size")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--vis", type=str2bool, default=False)
    parser.add_argument("--num-vis", type=int, default=3, help="number of examples to visualize")
    # hparams
    parser.add_argument("--res-init", type=str2bool, default=False, help="base token type, residue vs bond (default bond)")
    parser.add_argument("--free-bonds", type=str2bool, default=False, help="whether to not standardize bond lengths")
    parser.add_argument("--rmsd-super-res", type=str2bool, default=False, help="whether to resolve structures every new rmsd key")
    parser.add_argument("--rmsd-only", type=str2bool, default=False, help="whether to not set rmsd keys")
    parser.add_argument("--bin-strategy", help="how to bin values", default="histogram", choices=["histogram", "histogram-cover", "uniform"])
    parser.add_argument("--bins", type=str2dict, help=":-separated number of bins per size step, example: 1-100:10-10 means 100 bins from token size 1 to 9, 10 bins from 10 onwards", default="1-10")
    parser.add_argument("--sec", type=str2bool, default=False, help="whether to compute sec structures to guide token discovery")
    parser.add_argument("--sec-eval", type=str2bool, default=False, help="whether to evaluate sec structure overlap")
    parser.add_argument(
        "--p-min-size",
        type=int_or_inf,
        default=float("inf"),
        help="when to start using rmsd binning; 0 to turn off bpe; or ‘inf’ to mean no limit",
    )
    parser.add_argument(
        "--num-p",
        type=str2dictorint,
        default=3,
        help="num partitions for rmsd binning; OR -separated number of rmsd partitions per size step, example: 3-100:9-10 means 100 partitions from token size 3 to 9, 10 bins from 9 onwards"
    )
    parser.add_argument(
        "--max-num-strucs",
        type=int,
        default=1000,
        help="max N for running medoids",
    )
    parser.add_argument("--glue-opt", type=str2bool, default=False, help="whether to opt the glue angles for rmsd keys")
    parser.add_argument("--glue-opt-prior", type=float, default=0.0, help="whether to impose a prior loss in glue opt")
    parser.add_argument("--glue-opt-method", choices=["each", "all"], default="each", help="optimize each glue after rounding or all glues together")
    parser.add_argument("--cache", action='store_true', help="whether to use cached data")
    parser.add_argument("--save-every", type=int, default=10, help="how often to dump")
    parser.add_argument("--plot-every", type=int, default=50, help="how often to plot")
    parser.add_argument("--num-ref", type=int, default=10, help="how many ref structures to eval error")
    args = parser.parse_args()
    # Post‐parse validation
    # if args.p_min_size == 0 and args.bins != {1: 1}:
    #     parser.error("--bins must be '1-1' when --p-min-size is 0")
    if args.vis and args.num_ref and args.num_vis and args.num_vis > args.num_ref:
        parser.error(f"num-ref={args.num_ref} must be >= num-vis={args.num_vis}")
    return args


def amino_acid_sequence(fname):
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',  'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',  'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}       
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0] 
    seq = [d3to1[k] for k in struc.get_residues(source_struct)[1]]
    return seq


def main():
    args = parse_args()  
    if args.save_dir:
        save_dir = Path(args.save_dir)
        name = save_dir.name
        plot_dir = os.path.join(args.base_dir, f'./plots/learn/{name}')
        assert os.path.exists(plot_dir)
        assert os.path.exists(save_dir)
        setattr(args, 'plot_dir', plot_dir)
    elif args.auto:
        cur_time = time.time()
        setattr(args, 'plot_dir', os.path.join(args.base_dir, f'./plots/learn/{cur_time}'))
        setattr(args, 'save_dir', os.path.join(args.base_dir, f'./ckpts/{cur_time}'))
        os.makedirs(args.plot_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, "args.txt")
    if os.path.exists(args_path):
        print(f"loading args from {args_path}")
        loaded_args = load_args_from_txt(args_path)
        validate_args_match(
            current   = args,
            loaded    = loaded_args,
            skip      = ["auto", "save_dir", "vis"],   # fields you don’t need to compare
        )        
    else:
        with open(args_path, "w") as f:
            for arg_name, arg_value in sorted(args.__dict__.items()):
                f.write(f"{arg_name}: {arg_value}\n")
    setup_logger(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info("Script started.")    
    # Use args.save_dir for saving outputs.
    # Input folder remains the same for now.    
    # all_coords = []
    
    # files = os.listdir(cath_folder)
    # files = sorted(files, key=len)
    # for f in tqdm(files[:10]):
    #     if f:
    #         logger.info("Processing file: %s", f)
    #         all_coords.append(parse_pdb(os.path.join(cath_folder, f)))
    _iter, ckpt = -1, ""
    for f in glob.glob(f"{args.save_dir}/bpe_iter=*.pkl"):
        f = Path(f).name
        m = re.match("bpe_iter=(\d+).pkl", f)
        if m is None:
            continue
        cur_iter = int(m.groups()[0])
        if cur_iter > _iter:
            _iter = cur_iter
            ckpt = f"{args.save_dir}/{f}"    
    ref_path = f"{args.save_dir}/ref_coords.npy"
    lims_path = os.path.join(args.save_dir, f"lims.npy")
    if _iter > -1:
        logger.info(f"loading {ckpt} at iter={_iter}")        
        bpe = pickle.load(open(ckpt, 'rb'))
        N = len(bpe.tokenizers)    
        num_vis = min(N, args.num_vis) if args.num_vis else N        
        ref_coords = np.load(ref_path, allow_pickle=True)
        num_ref = len(ref_coords)
        xlims, ylims, zlims = map(lambda t: list(map(tuple, t)), tuple(np.load(lims_path)))        
        if args.save_dir != bpe.save_dir:
            logger.info(f"resetting save_dir from {bpe.save_dir} to {args.save_dir}")
            bpe.save_dir = args.save_dir
    else:
        init_bpe_path = os.path.join(args.save_dir, f'bpe_init.pkl')
        post_init_bpe_path = os.path.join(args.save_dir, f'bpe_post_init.pkl')
        pre_init_glue_opt_path = os.path.join(args.save_dir, f'bpe_pre_glue_opt.pkl')
        if Path(init_bpe_path).exists():
            print(f"loading {init_bpe_path}")
            bpe = pickle.load(open(init_bpe_path, "rb"))
        else:
            dataset = FullCathCanonicalCoordsDataset(args.data_dir, 
                                                    use_cache=args.cache, 
                                                    debug=False, 
                                                    zero_center=False, 
                                                    toy=args.toy, 
                                                    pad=args.pad, 
                                                    secondary=args.sec)     
            cleaned_structures = []
            for i, struc in enumerate(dataset.structures):
                if (struc['angles']['psi']==struc['angles']['psi']).sum() < len(struc['angles']['psi'])-1:
                    print(f"skipping {i}, {struc['fname']} because of missing dihedrals")
                else:
                    cleaned_structures.append(struc)
            logger.info(f"Removed {len(dataset.structures)-len(cleaned_structures)}/{len(dataset.structures)} structures with nan dihedrals.")
            N = len(cleaned_structures)
            dataset.structures = cleaned_structures        
            bpe = BPE(dataset.structures, 
                    bins=args.bins, 
                    bin_strategy=args.bin_strategy, 
                    save_dir=args.save_dir, 
                    rmsd_partition_min_size=args.p_min_size, 
                    rmsd_super_res=args.rmsd_super_res,
                    rmsd_only=args.rmsd_only,
                    num_partitions=args.num_p,
                    max_num_strucs=args.max_num_strucs,
                    compute_sec_structs=args.sec, 
                    plot_iou_with_sec_structs=args.sec_eval,                  
                    res_init=args.res_init,
                    std_bonds=not args.free_bonds,
                    glue_opt=args.glue_opt,
                    glue_opt_prior=args.glue_opt_prior,
                    glue_opt_method=args.glue_opt_method)
                    
            pickle.dump(bpe, open(init_bpe_path, 'wb+'))

        N = len(bpe.tokenizers)    
        num_vis = min(N, args.num_vis) if args.num_vis else N
        if Path(ref_path).exists():
            ref_coords = np.load(ref_path, allow_pickle=True)
            num_ref = len(ref_coords)
        else:
            num_ref = min(N, args.num_ref) if args.num_ref else N
            ref_coords = [bpe.tokenizers[i].compute_coords() for i in range(num_ref)]
            np.save(ref_path, ref_coords)
        
        xlims = [None for _ in range(num_vis)]
        ylims = [None for _ in range(num_vis)]
        zlims = [None for _ in range(num_vis)]
        for ind in range(num_vis):
            visual_path = os.path.join(args.save_dir, f"backbone_{ind}_iter=-1.png")
            res = bpe.tokenizers[ind].visualize(visual_path, vis_dihedral=False)
            xlims[ind], ylims[ind], zlims[ind] = tuple(res) # for later        
        np.save(lims_path, np.array([xlims, ylims, zlims]))
        if Path(post_init_bpe_path).exists():
            print(f"loading {post_init_bpe_path}")
            bpe = pickle.load(open(post_init_bpe_path, "rb"))
        else:
            if os.path.exists(pre_init_glue_opt_path):
                print(f"loading {pre_init_glue_opt_path}")
                bpe = pickle.load(open(pre_init_glue_opt_path, "rb"))
            else:
                bpe.initialize(path=os.path.join(args.save_dir, "hist_plot.png"))
                pickle.dump(bpe, open(pre_init_glue_opt_path, "wb+"))
            bpe.glue_opt_all()
            pickle.dump(bpe, open(post_init_bpe_path, 'wb+'))
        for ind in range(num_vis):
            visual_path = os.path.join(args.save_dir, f"backbone_{ind}_iter=init.png")
            bpe.tokenizers[ind].visualize(visual_path, vis_dihedral=False, xlim=xlims[ind], ylim=ylims[ind], zlim=zlims[ind])
        bpe.bin()    
        if args.debug: 
            bpe_debug = BPE(dataset.structures, bins=args.bins, save_dir=args.save_dir)
            bpe_debug.initialize()
            bpe_debug.old_bin()
    
    vis_paths = [[] for _ in range(num_vis)]    
    for t in range(_iter+1, 10000):
        ## visualization        
        if args.vis and t in list(range(0,10)) + list(range(10,100,10)) + list(range(100, 1000, 100)) + list(range(1000,10000,1000)):
            # Save current visualization.
            for ind in range(num_vis):
                visual_path = os.path.join(args.save_dir, f"backbone_{ind}_iter={t}.png")
                bpe.tokenizers[ind].visualize(visual_path, vis_dihedral=False, xlim=xlims[ind], ylim=ylims[ind], zlim=zlims[ind])
                vis_paths[ind].append(visual_path)            
                # Define the output GIF path.
                gif_path = os.path.join(args.save_dir, f"backbone_{ind}_iter_up_to={t}.gif")            
                # Read all PNG images and collect them as frames.
                frames = [imageio.imread(png_file) for png_file in vis_paths[ind]]            
                # Save the frames as a GIF with a 1 second delay per frame.
                durations = [1] * len(frames)
                try:
                    imageio.mimsave(gif_path, frames, format="GIF", duration=1, loop=0) 
                except ValueError:
                    print(f"frames have different sizes")
        bpe.step()
        for ind in range(num_vis):
            bpe.tokenizers[ind].bond_to_token.tree.visualize(os.path.join(args.save_dir, f'tokens_{ind}_iter={t}.png'), horizontal_gap=0.5, font_size=6)
        if t % args.save_every == 0:
            # save            
            pickle.dump(bpe.tokenizers[:num_ref], open(os.path.join(args.save_dir, f'ref_tokenizers={t}.pkl'), 'wb+'))
            # stats
            stats_path = os.path.join(args.save_dir, f'stats={t}.json')
            json.dump(
                {
                    "K": len(bpe._tokens),
                    "L": np.mean([len(t.bond_to_token) for t in bpe.tokenizers])
                },
                open(stats_path, "w+")
            )
            time_path = os.path.join(args.save_dir, f"times_iter={t}.png")            
            bpe.plot_times(time_path)
            if bpe.plot_iou_with_sec_structs:
                iou_path = os.path.join(args.save_dir, f"iou_iter={t}.png")
                bpe.plot_iou(iou_path)
            # finally dump the iter
            pickle.dump(bpe, open(os.path.join(args.save_dir, f'bpe_iter={t}.pkl'), 'wb+'))                
        if t % args.plot_every == 0:
            run_path = os.path.join(args.save_dir, f"run_iter={t}.png")
            if ref_coords is not None:
                plot(bpe,
                    len(bpe._tokens),
                    ref_coords,
                    run_path,
                    prev_iter=(t-args.plot_every),
                    no_iters=t, 
                    step_iter=args.save_every, 
                    ratio=N/1000)
        if args.debug: 
            bpe_debug.old_step()
            for i in range(bpe.n):
                t = bpe.tokenizers[i]
                t_ = bpe_debug.tokenizers[i]
                if t.bond_to_token != t_.bond_to_token:
                    breakpoint()
                if t.token_pos != t_.token_pos:
                    breakpoint()
            for k in bpe._geo_dict:
                if k not in bpe_debug._geo_dict:
                    breakpoint()
                elif set(bpe._geo_dict[k]) != set(bpe_debug._geo_dict[k]):
                    breakpoint()
            for k in bpe_debug._geo_dict:
                if k not in bpe._geo_dict:
                    breakpoint()
                elif set(bpe_debug._geo_dict[k]) != set(bpe._geo_dict[k]):
                    breakpoint()
    logger.info("Script finished.")



if __name__ == "__main__":
    main()
