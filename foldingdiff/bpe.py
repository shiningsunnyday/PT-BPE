from foldingdiff.tokenizer import *
from foldingdiff.plotting import *
from foldingdiff.algo import *
from foldingdiff.utils import *
import sys
import heapq
import json
import logging
import zipfile, os.path as _p
from collections import defaultdict
from functools import partial
from sortedcontainers import SortedDict
from torch.optim import LBFGS
from torch.multiprocessing import get_context
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
logger = logging.getLogger(__name__)

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


class BPE():
    def __init__(self, structures, bins, 
        bin_strategy="histogram",
        save_dir="./plots/bpe",
        compute_sec_structs=False, 
        plot_iou_with_sec_structs=False,
        res_init=False,
        std_bonds=True,
        rmsd_partition_min_size=4,
        rmsd_super_res=False,
        rmsd_only=False,
        num_partitions=3,
        max_num_strucs=500,
        glue_opt=False,
        glue_opt_prior=0.0,
        glue_opt_every=10,
        glue_opt_method="all",
        seed=None
    ):
        """
        structures: list of dataset objects
        bins: resolution of discretization, dict from size to num_bins
        bin_strategy: how to bin values
        compute_sec_structs: whether to use secondary structure count to define token pair priority
        rmsd_packing_min_size: when to start using rmsd partitioning, if 0 run with special setting
        num_partitions: how many partitions, int or dict from size to num_partitions
        """
        self.tokenizers = []
        for structure in structures:
            tokenizer = Tokenizer(deepcopy(structure), compute_sec_structs=compute_sec_structs | plot_iou_with_sec_structs)
            self.tokenizers.append(tokenizer)
        self.compute_sec_structs = compute_sec_structs
        self.plot_iou_with_sec_structs = plot_iou_with_sec_structs
        self.rmsd_partition_min_size = rmsd_partition_min_size
        self.rmsd_super_res = rmsd_super_res
        self.rmsd_only = rmsd_only
        self.glue_opt = glue_opt
        self.glue_opt_every = glue_opt_every
        self.glue_opt_prior = glue_opt_prior
        self.glue_opt_method = glue_opt_method
        if isinstance(num_partitions, dict):
            self.num_partitions = ThresholdDict(num_partitions)
        else:
            self.num_partitions = num_partitions
        self.max_num_strucs = max_num_strucs
        self.res_init = res_init
        self.std_bonds = std_bonds
        self.bins = bins
        self.bin_strategy = bin_strategy
        print(self.bin_strategy)
        self.n = len(self.tokenizers)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.save_dir = save_dir
        self._step = 0
        self._times = []
        self._ious = []
    
    
    def initialize(self, path=None):
        logger.info(f"Initialize start")
        start_time = time.perf_counter()
        self._init_thresholds(path=path) # call this before _init_tokens
        logger.info(f"_init_thresholds took {time.perf_counter()-start_time}")
        start_time = time.perf_counter()        
        if self.res_init:
            self._init_res_tokens()
            logger.info(f"_init_res_tokens took {time.perf_counter()-start_time}")
        else:
            self._init_tokens()        
            logger.info(f"_init_tokens took {time.perf_counter()-start_time}")
        logger.info(f"Initialize finish")
    

    def glue_opt_all(self):
        logger.info("Glue opt all start")
        max_workers = _effective_cpus()
        N = len(self.tokenizers)
        if max_workers == 0:
            global BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR
            BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR = self._bin_centers, self._thresholds, self.glue_opt_prior
            self.tokenizers = [BPE._opt_glue_worker(t, 3*t.n-4) for t in self.tokenizers]
        else:
            os.makedirs(os.path.dirname(self._assignment_cache_path(f"init_glue_opt", 0)), exist_ok=True)
            missing_jobs = []
            for idx in range(N):
                p = self._assignment_cache_path("init_glue_opt", idx, "pkl")
                if os.path.isfile(p):                       # already computed
                    try:
                        self.tokenizers[idx] = pickle.load(open(p, "rb"))
                    except:
                        missing_jobs.append(idx)
                else:                                       # still to be done
                    missing_jobs.append(idx)
            if missing_jobs:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=BPE._init_opt_glue_worker,
                    initargs=(self._bin_centers, self._thresholds, self.glue_opt_prior)
                ) as pool:       
                    for idx, t in zip(missing_jobs, tqdm(pool.map(BPE._opt_glue_worker, [self.tokenizers[idx] for idx in missing_jobs], chunksize=10), total=len(missing_jobs), desc="optimizing all glues")):
                        self.tokenizers[idx] = t
                        pickle.dump(t, open(self._assignment_cache_path("init_glue_opt", idx, "pkl"), "wb+"))
        logger.info("Glue opt all finish")


    def _init_res_tokens(self):
        """
        See docstring for _init_tokens
        """
        self._tokens = {}
        label_dict = {}
        res_geo = defaultdict(list)   
        max_workers = _effective_cpus()
        N = len(self.tokenizers)
        super_res = self.rmsd_super_res if hasattr(self, "rmsd_super_res") else False
        lookup = self._thresholds if getattr(self, "std_bonds", True) else self._thresholds[1]
        relv_thresholds = {bt: lookup[bt] for bt in lookup if bt in Tokenizer.BOND_TYPES}
        # binned_bonds = [self._bin_val({Tokenizer.BOND_TYPES[i]: [0]}) for i in range(3)]
        if max_workers == 0: # debug
            global RELV_THRESHOLDS, STD_BONDS
            RELV_THRESHOLDS, STD_BONDS = relv_thresholds, getattr(self, "std_bonds", True)
            self.tokenizers = [BPE._set_bond_length_worker(t) for t in tqdm(self.tokenizers, desc="standardizing bond lengths")]
        else:
            os.makedirs(os.path.dirname(self._assignment_cache_path(f"bond_std_tokenizers", 0)), exist_ok=True)
            missing_jobs = []
            for idx in tqdm(range(N), desc="loading bond std tokenizers"):
                p = self._assignment_cache_path("bond_std_tokenizers", idx)
                if os.path.isfile(p):                       # already computed
                    try:
                        self.tokenizers[idx] = pickle.load(open(p, "rb"))
                    except:
                        missing_jobs.append(idx)
                else:                                       # still to be done
                    missing_jobs.append(idx)            
            if missing_jobs:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=BPE._init_set_bond_length_worker,
                    initargs=(relv_thresholds, getattr(self, "std_bonds", True))
                ) as pool:
                    futures = {}
                    for idx in tqdm(
                        missing_jobs,
                        total=len(missing_jobs),
                        desc="submitting bond std jobs"
                    ):
                        future = pool.submit(
                            BPE._set_bond_length_worker, 
                            self.tokenizers[idx],
                            self._assignment_cache_path("bond_std_tokenizers", idx)
                        )
                        futures[future] = idx

                for future in tqdm(as_completed(futures), total=len(futures),
                            desc="standardizing bond lengths"):
                    idx = futures.pop(future)
                    p = self._assignment_cache_path("bond_std_tokenizers", idx)
                    self.tokenizers[idx] = pickle.load(open(p, "rb"))
        
        if self.glue_opt and self.glue_opt_method == "all":            
            def is_complete_npz(path: str) -> bool:
                """Return True iff `path` looks like a valid, complete `.npz` file."""
                if not _p.isfile(path) or _p.getsize(path) == 0:
                    return False                           # missing or empty
                return zipfile.is_zipfile(path)  
            os.makedirs(os.path.dirname(self._assignment_cache_path(f"exit_frame_cache", 0)), exist_ok=True)
            missing_jobs = []
            for idx in tqdm(range(N), desc="loading exit frame cache"):
                p = self._assignment_cache_path("exit_frame_cache", idx, "npz")
                if is_complete_npz(p):                       # already computed
                    self.tokenizers[idx].cached_all_frames = p
                else:                                       # still to be done
                    missing_jobs.append(idx)
            if missing_jobs:
                with ProcessPoolExecutor(
                    max_workers=max_workers                
                ) as pool:
                    arg_iter = (
                        (
                            idx,
                            self.tokenizers[idx],                # cheap to serialise
                            self._assignment_cache_path("exit_frame_cache", idx, "npz")
                        )
                        for idx in missing_jobs
                    )                    
                    for idx in tqdm(pool.map(BPE._cache_exit_frames,
                                            arg_iter,
                                            chunksize=10),              # keeps queue small
                                    total=len(missing_jobs),
                                    desc="caching exit frames"):
                        p = self._assignment_cache_path("exit_frame_cache", idx, "npz")
                        # frames = np.load(p, allow_pickle=True)
                        # self.tokenizers[idx].cached_all_frames = (frames["R_occs"], frames["t_occs"])
                        self.tokenizers[idx].cached_all_frames = p
            # for idx, t in tqdm(enumerate(self.tokenizers), desc="caching exit frames"):
            #     R_occs, t_occs = t.exit_frame(3, 3*t.n-4, ret_all=True)
            #     t.cached_all_frames = (R_occs, t_occs)
        
        for ti, t in tqdm(enumerate(self.tokenizers), desc="iterating over residue geos", mininterval=1.0):
            # update binned residue geo
            labels = []
            for i in range(t.n): # for each residue
                start = 3*i
                length = 3 if i < t.n-1 else 2                
                if length < self.rmsd_partition_min_size:
                    geo = t.token_geo(start, length)                
                    self.quant_geo(geo)
                    # quantized key
                    key = self._bin_val(geo)
                    key_str = BPE.hash_geo(key)
                    if key_str not in label_dict:
                        n = len(label_dict)
                        label_dict[key_str] = n
                    else:
                        n = label_dict[key_str]
                    t.set_token_geo(start, length, key)
                    labels.append(n)
                else:
                    # add this to res_{geo}
                    res_geo[length].append((ti, start, length))
                    labels.append(None)

            # 3*(t.n-1) + 2 bonds
            new_tokens = [(3*i, labels[i], 3) for i in range(t.n-1)] + [(3*t.n-3, labels[t.n-1], 2)]
            bond_to_token = {t[0]: t for t in new_tokens} # start bond : token
            token_pos = [3*(i//3) for i in range(3*t.n-1)]                
            t.token_pos = token_pos
            t.tokens = new_tokens
            t.bond_to_token = bond_to_token

        if res_geo:
            self._sphere_dict = {} # for rmsd sphere packing
            self._tokens = {}
            for n, size in enumerate(res_geo):
                for t in self.tokenizers:
                    # need to re-init this later
                    t.bond_to_token_copy = dict(t.bond_to_token)
                # run k-medoids
                # compute all the residue medoids
                # decide how many participate    
                N = len(res_geo[size])            
                if N > self.max_num_strucs:
                    active_inds = self.rng.choice(N, self.max_num_strucs, replace=False)
                else:
                    active_inds = np.arange(N)
                active_coords = []
                active_occurs = []
                for i in tqdm(active_inds, desc=f"gathering {len(active_inds)} occurrences"):
                    ti, start, length = res_geo[size][i]
                    t = self.tokenizers[ti]
                    coords = t.compute_coords(start, length, super_res)
                    # sanity check
                    # geo_template = t.token_geo(0,5) if size == 3 else t.token_geo(0,2)
                    # geo = t.token_geo(start, length)
                    # for k in geo: geo_template[k][:len(geo[k])]=geo[k]
                    # coords_template = Tokenizer.geo_nerf(geo_template).cartesian_coords[:size+1]
                    # err = compute_rmsd(coords_template, coords)                    
                    # sanity check end
                    active_coords.append(coords)
                    active_occurs.append((ti, start, length))

                # run k-medoids                
                medoid_inds = k_medoids(active_coords, self.num_partitions[size], rng=self.rng)                
                # -------- final assignment for *all* structures ---------------------------                                                
                if max_workers == 0:
                    assignments = [None for _ in range(N)]
                    for idx, (ti, start, length) in enumerate(res_geo[size]):
                        assignments[idx] = BPE._compute_assignment((self.tokenizers[ti], start, length), active_coords, medoid_inds, super_res)
                else:   
                    fname = os.path.join(self.save_dir, f"assignments_{size}.npy")
                    if not os.path.exists(fname):
                        # fill with sentinel -1 so you know what is still missing
                        np.lib.format.open_memmap(fname, mode='w+', dtype=np.int32, shape=(N,))[:] = -1
                    assignments = np.load(fname, mmap_mode='r+')
                    missing_jobs = np.where(assignments == -1)[0]

                    if len(missing_jobs):
                        with ProcessPoolExecutor(
                                max_workers=max_workers,
                                initializer=BPE._init_assignment_worker,
                                initargs=(active_coords,
                                        medoid_inds,
                                        super_res,
                                        self.tokenizers,
                                        res_geo[size],
                                        fname)
                        ) as pool:
                            for _ in tqdm(pool.map(BPE._assignment_worker, missing_jobs,
                                                        chunksize=5000),   # big chunk
                                                total=len(missing_jobs),
                                                desc="computing assignments"):
                                pass
                        assignments.flush()

                # vis the res medoids     
                if size == 3:
                    k = '{"N:CA": [0], "CA:C": [0], "0C:1N": [0], "tau": [0], "CA:C:1N": [0], "psi": [0]}'
                elif size == 2:
                    k = '{"CA:C": [0], "0C:1N": [0], "CA:C:1N": [0]}'
                else:
                    raise NotImplementedError
                self._sphere_dict[k] = []
                for p, m in enumerate(medoid_inds):
                    ti, i1, length = active_occurs[m]
                    t = self.tokenizers[ti]
                    struc = t.token_geo(i1, length, orig=super_res)
                    logger.info(f"res partition {p}: {struc}")
                    self._sphere_dict[k].append(struc)
                    key_vis_path = os.path.join(self.save_dir, f"init_size={size}_{p}.png")
                    # t.visualize_bonds(i1, length, key_vis_path)
                    # add to _tokens
                    self._tokens[(n, p)] = struc                

                # for each assignment
                    # set_token_geo to the medoid
                    # (optional) run glue
                    # track it in labels
                # for each t for each token
                    # revise t.tokens
                    # revise t.bond_to_token

                medoids = [active_inds[m] for m in medoid_inds]
                for (ti1, start1, length1), p in tqdm(zip(res_geo[size], assignments), desc="iterating over assignments", mininterval=1.0):
                    t1 = self.tokenizers[ti1]
                    # ti2, start2, length2 = res_geo[size][medoids[p]]
                    # t2 = self.tokenizers[ti2]
                    # assert length1 == length2                    
                    if start1 > 0 and self.glue_opt and self.glue_opt_method == "each":                    
                        R_occ, t_occ = t1.exit_frame(start1, 3*((length1-2)//3)+2)
                        t1.set_token_geo(start1, length1, self._tokens[(n, p)])
                        # t2.token_geo(start2, length1, orig=super_res)
                        BPE.opt_glue(t1, start1, 3*((length1-2)//3)+2, R_occ, t_occ) # already quantizes
                    else:                        
                        t1.set_token_geo(start1, length1, self._tokens[(n, p)])
                    t1.tokens[start1//3] = (start1, (n, p), length1)
                    t1.bond_to_token_copy[start1] = t1.tokens[start1//3]
                

                for t in self.tokenizers:
                    if hasattr(t, "bond_to_token_copy"):
                        t.bond_to_token = t.bond_to_token_copy
                        delattr(t, "bond_to_token_copy")
                        
        if not (res_geo and self.glue_opt):
            # quantize inter-token angles and maybe bonds
            keys = ['omega', 'C:1N:1CA', 'phi']
            for ti, t in enumerate(self.tokenizers):
                # update omega, phi, and C-N-CA
                for angle in keys:
                    relv_thresh = self._thresholds[1][angle]
                    t.angles_and_dists[angle] = [sum(relv_thresh[BPE.get_ind(
                            v if angle in Tokenizer.BOND_TYPES else (v+2*np.pi)%(2*np.pi), 
                            relv_thresh
                        )])/2 if v==v else v for v in t.angles_and_dists[angle]]                    
            self._tokens = {n: json.loads(key_str) for key_str, n in label_dict.items()}
        logger.info(f"initialized {len(self._tokens)} residue-level tokens")     

    
    def _init_tokens(self):
        """
        Here we treat each bond as an inital "bond-type" token, then standardizing the length of the bond to a fixed value. 
        If self.res_init is true, we instead take the quantized residue geo's as the initial tokens and do quantize all occurrences (bin val) using the quantization strength for size 2 or 3 keys.
            For bond-residue tokens of size >= self.rmsd_partition_min_size, we instead partition the geo's with k-medoids, then take the medoids as the initial tokens. We "quantize" to these medoids. We also quantize the glue angles (omega, psi, C-N-CA). If opt_glue, we add an extra step to optimize the glue angles before quantizing them.
        In both cases, we initialize t.tokens and t.bond_to_tokens.
        """
        self._tokens = {}
        for i in range(3):
            self._tokens[i] = {Tokenizer.BOND_TYPES[i]: [0]}        

        for ti, t in enumerate(self.tokenizers):
            # quantize bond lengths
            if getattr(self, "std_bonds", True):
                for i in range(3*t.n-1):
                    t.set_token_geo(i, 1, self._bin_val({Tokenizer.BOND_TYPES[i%3]: [0]}))
            # quantize bond angles

            new_tokens = [(i,t.bond_labels[i],1) for i in range(3*t.n-1)]
            bond_to_token = {t[0]: t for t in new_tokens}
            token_pos = [i for i in range(3*t.n-1)]
            t.token_pos = token_pos
            t.tokens = new_tokens
            t.bond_to_token = bond_to_token
    

    @staticmethod
    def fk_segment_torch(t, idx, length, ω, θ, φ, ret_all=False):
        device = ω.device
        geo = t.token_geo(idx-3, length+3)
        for k in geo:
            geo[k] = torch.as_tensor(geo[k], device=device, dtype=ω.dtype)
        if ret_all:
            assert len(ω) == len(geo["omega"])
            geo["omega"] = ω
            geo["C:1N:1CA"] = θ
            geo["phi"] = φ
        else:
            geo["omega"][0] = ω
            geo["C:1N:1CA"][0] = θ
            geo["phi"][0] = φ
        assert len(geo['N:CA']) == len(geo['CA:C'])
        assert len(geo['CA:C']) == len(geo.get('0C:1N', []))+1
        num_bonds = Tokenizer.num_bonds(geo)
        assert num_bonds%3 == 2
        n_init, ca_init, c_init = update_backbone_positions(N_INIT, CA_INIT, C_INIT, geo['CA:C'][0].item(), geo['N:CA'][0].item(), geo['tau'][0].item())
        nerf = NERFBuilder(
            phi_dihedrals=torch.cat((torch.tensor([np.nan], device=device), geo['phi'])),
            psi_dihedrals=torch.cat((geo['psi'], torch.tensor([np.nan], device=device))),
            omega_dihedrals=torch.cat((geo['omega'], torch.tensor([np.nan], device=device))),
            bond_len_n_ca=geo['N:CA'][1:],
            bond_len_ca_c=geo['CA:C'][1:],
            bond_len_c_n=geo['0C:1N'],
            bond_angle_n_ca=geo['C:1N:1CA'],
            bond_angle_ca_c=geo['tau'][1:],
            bond_angle_c_n=geo['CA:C:1N'],
            init_coords=[n_init,ca_init,c_init]
        )
        coords = nerf.cartesian_coords
        if ret_all:
            return zip(*(frame_from_triad_torch(*coords[3*i:3*(i+1)]) for i in range(1, t.n)))
        else:
            return frame_from_triad_torch(*coords[-3:].unbind(dim=0))

    @staticmethod
    def optimize_glues_entry_torch(t, idx, length, R_occ, t_occ,
                                init_glue,  # (omega_{s-1}, theta_CNCA_s, phi_s)
                                wR=1.0, wt=0.1, 
                                lambda_prior=0.1,
                                bin_centers=None,
                                thresholds=None):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        # ---- fixed tensors from original chain ----
        if isinstance(R_occ, list):
            assert isinstance(t_occ, list)
            R_occs = [torch.tensor(r_occ,  dtype=torch.float32, device=device) for r_occ in R_occ]
            t_occs = [torch.tensor(t_occ,  dtype=torch.float32, device=device) for t_occ in t_occ]
            ret_all = True
        else:
            R_occ = torch.tensor(R_occ,  dtype=torch.float32, device=device)
            t_occ = torch.tensor(t_occ,  dtype=torch.float32, device=device)
            ret_all = False

        # ---- initial glue triple as unconstrained raw params ----
        init = torch.tensor(init_glue, dtype=torch.float32, device=device)
        raw   = torch.nn.Parameter(init)

        opt = LBFGS([raw], max_iter=20, line_search_fn='strong_wolfe')
        loss_log = []
        # opt = torch.optim.Adam([raw], lr=1e-2)

        def wrap(a):
            # atan2(sin,cos) gives (-π, π]; shift by 2π then mod 2π → [0, 2π)
            a_wrapped = torch.atan2(torch.sin(a), torch.cos(a)) + (2.0 * np.pi)
            return torch.remainder(a_wrapped, (2.0 * np.pi))

        def snap_bin(arr, x):
            """
            Parameters
            ----------
            arr : array-like of shape (m, 2)
                Consecutive bins; arr[i] = (left_edge, right_edge) of bin i.
            x   : scalar
                Value whose bin index we want.

            Returns
            -------
            int   ∈  {-1, … , m}
                ‑1  if x < left-most edge
                m   if x > right-most edge
                i   such that x ∈ [left_i , right_i) otherwise
            """
            # Convert to ndarray for easy slicing
            arr = np.asarray(arr)
            m   = arr.shape[0]

            # Fast checks for the two out-of-range cases
            if x < arr[0, 0]:
                return arr[0][0]
            if x > arr[-1, 1]:
                return arr[-1, 1]

            # Binary search on the right edges
            right_edges = arr[:, 1]           # length m
            i = bisect.bisect_right(right_edges, x)
            return sum(arr[i])/2
            

        def circ_kde_prior(angle, centers, weights, kappa=20.0):
            # mixture of von Mises densities (normalized up to a constant; good enough for regularization)
            # centers: tensor [K], weights: tensor [K] sum to 1
            # vm ~ exp(kappa * cos(angle - mu))
            diffs = angle - centers  # [K]
            log_terms = kappa * torch.cos(diffs) + torch.log(weights + 1e-12)  # [K]
            # log-sum-exp over modes (ignore normalization constants; this is a prior up to a constant)
            return -torch.logsumexp(log_terms, dim=0)  # negative log prior

        def closure():
            opt.zero_grad()
            if ret_all:
                ωs, θs, φs= (wrap(x) for x in raw.unbind(-1))
                R_news, t_news = BPE.fk_segment_torch(t, idx, length, ωs, θs, φs, ret_all=ret_all)                
                rot_loss = sum((0.5 * torch.sum((R_occ - R_new)**2) for R_new, R_occ in zip(R_news, R_occs)))
                trans_loss = sum(torch.sum((t_occ - t_new)**2) for t_new, t_occ in zip(t_news, t_occs))
            else:
                ω, θ, φ = wrap(raw)
                R_new, t_new = BPE.fk_segment_torch(t, idx, length, ω, θ, φ, ret_all=ret_all)                
                rot_loss = 0.5 * torch.sum((R_occ - R_new)**2)
                trans_loss = torch.sum((t_occ - t_new)**2)
            exit_frame_loss = wR*rot_loss + wt*trans_loss
            if lambda_prior > 0.0:
                prior = (circ_kde_prior(ω, bin_centers[length]['omega'], self._bin_weights[length]['omega'], kappa=50.)
                    + circ_kde_prior(θ, bin_centers[length]['C:1N:1CA'], self._bin_weights[length]['C:1N:1CA'], kappa=20.)
                    + circ_kde_prior(φ, bin_centers[length]['phi'], self._bin_weights[length]['phi'], kappa=20.))
                loss = exit_frame_loss + lambda_prior * prior                        
            else:
                loss = exit_frame_loss            
            loss.backward()
            loss_log.append(loss.item())
            return loss

        opt.step(closure)
        if ret_all:
            ω_opts, θ_opts, φ_opts= (wrap(x) for x in raw.unbind(-1))
            ω_opts = [snap_bin(thresholds[length]['omega'], ω_opt) for ω_opt in ω_opts]
            θ_opts = [snap_bin(thresholds[length]['C:1N:1CA'], θ_opt) for θ_opt in θ_opts]
            φ_opts = [snap_bin(thresholds[length]['phi'], φ_opt) for φ_opt in φ_opts]
            return (ω_opts, θ_opts, φ_opts)
        else:
            ω_opt, θ_opt, φ_opt = wrap(raw).detach().cpu().numpy()
            ω_opt = snap_bin(thresholds[length]['omega'], ω_opt)
            θ_opt = snap_bin(thresholds[length]['C:1N:1CA'], θ_opt)
            φ_opt = snap_bin(thresholds[length]['phi'], φ_opt)
            # print("loss trace (first 10):", loss_log[:10], "... last:", loss_log[-1])
            return (ω_opt, θ_opt, φ_opt)


    def optimize_glues_entry(self, t, idx, length, R_occ, t_occ,
                                init_glue,  # (omega_{s-1}, theta_CNCA_s, phi_s)
                                wR=1.0, wt=0.1):
        """
        Coordinate-descent style coarse-to-fine search over 3 angles.
        Assumes t_obj has snapped internal angles for [s,e] already.
        """

        # -------- build discrete candidate sets --------------------------------
        omegas = np.mean(self._thresholds[length]['omega'], axis=-1)   # sorted 1-D
        thetas = np.mean(self._thresholds[length]['C:1N:1CA'], axis=-1)
        phis   = np.mean(self._thresholds[length]['phi'],   axis=-1)
        # -----------------------------------------------------------------------

        # -------- snap start point to nearest grid values ----------------------
        o_idx = int(np.argmin(np.abs(omegas - init_glue[0])))
        t_idx = int(np.argmin(np.abs(thetas - init_glue[1])))
        p_idx = int(np.argmin(np.abs(phis   - init_glue[2])))
        best  = np.array([omegas[o_idx], thetas[t_idx], phis[p_idx]], dtype=float)
        # -----------------------------------------------------------------------

        def loss_fn(glue, idx, length, wR, wt):
            # set glues at the left boundary of the segment
            omega, theta, phi = glue
            t.set_glue_left(idx, (omega, theta, phi))
            R_new, t_new = t.exit_frame(idx, length)
            return (wR * rot_geodesic(R_occ, R_new) ** 2 +
                    wt * np.sum((t_occ - t_new) ** 2), R_new, t_new)

        loss_for = partial(loss_fn, idx=idx, length=length, wR=wR, wt=wt)
        best_val, _, _ = loss_for(best)

        # ------------- choose search strategy ---------------------------------
        if max(len(omegas), len(thetas), len(phis)) <= 5:        # small ⇒ brute
            for o in omegas:
                for th in thetas:
                    for ph in phis:
                        cand = np.array([o, th, ph])
                        val, _, _ = loss_for(cand)
                        if val < best_val:
                            best, best_val = cand, val
            return tuple(best), best_val
        # ------------- otherwise: discrete coordinate descent -----------------
        improved = True
        while improved:
            improved = False
            # loop over the three dimensions separately
            for dim, (arr, idx) in enumerate(
                    [(omegas, o_idx), (thetas, t_idx), (phis, p_idx)]):
                for delta in (-1, 1):                      # one step ↓ or ↑
                    new_idx = idx + delta
                    if 0 <= new_idx < len(arr):
                        cand_idx = [o_idx, t_idx, p_idx]
                        cand_idx[dim] = new_idx
                        cand = np.array([omegas[cand_idx[0]],
                                        thetas[cand_idx[1]],
                                        phis[cand_idx[2]]])
                        val, _, _ = loss_for(cand)
                        if val + 1e-9 < best_val:
                            o_idx, t_idx, p_idx = cand_idx   # accept move
                            best, best_val = cand, val
                            improved = True
        return tuple(best)

    @staticmethod
    def _compute_assignment(args):             # NEW
        """Worker that returns the index of the closest medoid for one segment."""
        ti, start, length = args
        t = TOKENIZERS[ti]
        coords = t.compute_coords(start, length, orig=ORIG)        
        return BPE._compute_assignment_inner(coords, [ACTIVE_COORDS[idx] for idx in MEDOID_INDS])        


    @staticmethod
    def _compute_assignment_inner(coords, list_of_coords):
        costs = [compute_rmsd(coords, active_coords) for active_coords in list_of_coords]
        return np.argmin(costs)

    
    @staticmethod
    def _init_assignment_worker(ac, mi, o, tok_list=None, rg=None, fname=None):
        global ACTIVE_COORDS, MEDOID_INDS, ORIG, TOKENIZERS, RES_GEO
        if fname is not None:
            global ASSIGN
            ASSIGN = np.load(fname, mmap_mode='r+')
        ACTIVE_COORDS = ac
        MEDOID_INDS = mi
        ORIG = o
        TOKENIZERS = tok_list
        RES_GEO = rg

    @staticmethod
    def _compute_assignment_wrapper(idx):
        ti, start, length = RES_GEO[idx]        
        ans = BPE._compute_assignment((ti, start, length))
        return ans

    
    @staticmethod
    def _init_set_bond_length_worker(rt, sb):
        global RELV_THRESHOLDS, STD_BONDS
        RELV_THRESHOLDS = rt
        STD_BONDS = sb
    
    @staticmethod
    def _init_opt_glue_worker(bc, th, gop):
        global BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR
        BIN_CENTERS = bc
        THRESHOLDS = th
        GLUE_OPT_PRIOR = gop


    @staticmethod
    def _assignment_worker(idx):                                          # lightweight
        if ASSIGN[idx] == -1:                                  # still empty
            ASSIGN[idx] = BPE._compute_assignment_wrapper(idx)
    

    @staticmethod
    def _cache_exit_frames(args):
        idx, t, path = args
        R_occs, t_occs = t.exit_frame(3, 3*t.n-4, ret_all=True)
        np.savez(path, R_occs=R_occs, t_occs=t_occs)
        return idx      


    def _assignment_cache_path(self, prefix: str, idx: int, ext: str = "npy") -> str:
        """
        Return the on-disk path of a cached assignment.
        """
        return os.path.join(self.save_dir, prefix, f"{idx}.{ext}")
    
    @staticmethod
    def _set_bond_length_worker(t, path=None, strict=True):
        # update avg bond lengths
        for i in range(3*t.n-1):
            geo = t.token_geo(i, 1)
            assert len(geo) == 1
            key = list(geo)[0]
            assert len(geo[key]) == 1 
            if STD_BONDS:
                assert len(RELV_THRESHOLDS[key]) == 1
                ind = 0
            else:
                if not strict and geo[key][0] < RELV_THRESHOLDS[key][0][0]:
                    ind = 0
                elif not strict and geo[key][0] > RELV_THRESHOLDS[key][-1][1]:
                    ind = -1
                else:
                    ind = BPE.get_ind(geo[key][0], RELV_THRESHOLDS[key])
            val = sum(RELV_THRESHOLDS[key][ind])/2
            t.set_token_geo(i, 1, {key: [val]})
        if path:
            pickle.dump(t, open(path, "wb+"))
        else:
            return t

    @staticmethod
    def _opt_glue_worker(t):
        torch.set_num_threads(1)               # avoid over-subscription
        assert hasattr(t, "cached_all_frames") # init frames
        if isinstance(t.cached_all_frames, tuple):
            R_occs, t_occs = t.cached_all_frames
        else:
            assert os.path.exists(t.cached_all_frames)
            frames = np.load(t.cached_all_frames, allow_pickle=True)
            R_occs, t_occs = frames["R_occs"], frames["t_occs"]
        assert len(R_occs) == len(t_occs) == t.n-1
        cur_R_occs, cur_t_occs = [], []
        for (start, _, length) in t.tokens[:-1]:
            # need which exit frame for glue from this token to next
            res_no = (start+length)//3+1 # this residue
            cur_R_occs.append(R_occs[res_no-2]) # because the 0'th exit frame is for residue 2
            cur_t_occs.append(t_occs[res_no-2])
        return BPE.opt_glue(t, 3, 3*t.n - 4, cur_R_occs, cur_t_occs)        


    @staticmethod
    def opt_glue(t: Tokenizer, i1, length, R_occ, t_occ, glue_opt_prior=None, bin_centers=None, thresholds=None):
        assert bin_centers or "BIN_CENTERS" in globals()
        assert thresholds or "THRESHOLDS" in globals()
        assert glue_opt_prior or "GLUE_OPT_PRIOR" in globals()
        if bin_centers is None:
            bin_centers = BIN_CENTERS
        if thresholds is None:
            thresholds = THRESHOLDS
        if glue_opt_prior is None:
            glue_opt_prior = GLUE_OPT_PRIOR
        # optimize angles left of residue starting at bond i1
        if i1 % 3:
            raise ValueError(f"i1={i1} has to be start of residue")
        if length % 3 != 2:
            raise ValueError(f"i1+length-1 must end the last residue")
        if isinstance(R_occ, list):
            init_glue = [t.get_glue_left(i+l) for (i, _, l) in t.tokens[:-1]]
        else:
            init_glue = t.get_glue_left(i1)    # (psi_{s-1}, theta_CNCA_s, phi_s)        
        # optimize
        best_glue = BPE.optimize_glues_entry_torch(
            t, i1, length,
            R_occ=R_occ, t_occ=t_occ,
            init_glue=init_glue,
            wR=1.0, wt=0.1, 
            lambda_prior=glue_opt_prior,
            bin_centers=bin_centers,
            thresholds=thresholds
        )

        # set the optimized glue and recompute coords
        if isinstance(init_glue, list):
            for idx, best_glue in enumerate(zip(*best_glue)):
                i, _, l = t.tokens[idx]
                t.set_glue_left(
                    i+l,
                    best_glue
                )                
        else:
            t.set_glue_left(
                i1,
                best_glue
            )
        return t
    
    def _bin_side_chain(self, key):
        """
        We obtain the side chain placement in a data-driven way
        We iterate through every residue of the same amino acid
        Where the N-CA-C bond angle is in the same bin
        Then plot the variation in positions
        For reference, also plot variation across all bins
        """      
        breakpoint()        


    def _init_thresholds(self, path=None):
        """
        We obtain the thresholds
        We fix all bond lengths, so the thresholds are trivial.
        These threshold determine statistical significance of a motif, so we should choose them carefully
        We make these depend on |token|
        Due to circular angular data, we use circular histogram
        Thresholds: Dict{|token|: List[(bin_start, bin_end)]}
        """
        keys = Tokenizer.BOND_ANGLES+Tokenizer.DIHEDRAL_ANGLES
        if not getattr(self, "std_bonds", True):
            keys += Tokenizer.BOND_TYPES
        self._thresholds = ThresholdDict()
        self._bin_counts = ThresholdDict()
        self._bin_centers = ThresholdDict()
        self._bin_weights = ThresholdDict()
        for size, num_bins in self.bins.items():
            _thresholds = {}
            _bin_counts = {}
            # we will fix the bond lengths
            vals = {}
            for t in self.tokenizers:                
                angles = t.angles_and_dists
                for key in keys: # these are mostly fixed
                    t_vals = angles[key][angles[key].fillna(0)!=0.].tolist()
                    if key == "tau":
                        t_vals.append(t._bond_angle(0))
                    elif key == "N:CA":
                        t_vals.append(t._bond_length(0))
                    elif key == "CA:C":
                        t_vals.append(t._bond_length(1))
                    vals[key] = vals.get(key, [])
                    vals[key].extend(t_vals)

            for key in keys:
                name = f"{key}_{self.bin_strategy}_{num_bins}.png"
                if path is not None:
                    path = Path(path).with_name(name)                
                if self.bin_strategy.startswith("histogram"):
                    starts, ends, widths, counts = save_histogram(vals[key], circular=(key not in Tokenizer.BOND_TYPES), path=path, bins=self.bins[size], title=f"{self.bin_strategy} {key}, {num_bins} bins", cover="cover" in self.bin_strategy)
                elif self.bin_strategy == "uniform":
                    starts, ends, widths, counts = save_histogram_equal_counts(vals[key], circular=(key not in Tokenizer.BOND_TYPES), path=path, bins=self.bins[size], title=f"{self.bin_strategy} {key}, {num_bins} bins")
                else:
                    raise NotImplementedError
                logger.info(f"# bins: {len(counts)}, bin starts: {starts}, bin ends: {ends}, counts: {counts}")            
                for start, end, width, count in zip(starts, ends, widths, counts):
                    _thresholds[key] = _thresholds.get(key, []) + [(float(start), float(end))]
                    _bin_counts[key] = _bin_counts.get(key, []) + [count]                 
                                
            self._thresholds[size] = _thresholds
            self._bin_counts[size] = _bin_counts
            self._bin_centers[size] = {k: torch.tensor(v, dtype=torch.float32).mean(axis=-1) for k, v in _thresholds.items()}
            self._bin_weights[size] = {k: torch.tensor(v, dtype=torch.float32)/sum(v) for k, v in _bin_counts.items()}
        
        if getattr(self, "std_bonds", True):
            for i in range(3):
                self._thresholds[Tokenizer.BOND_TYPES[i]] = [(Tokenizer.BOND_LENGTHS[i], Tokenizer.BOND_LENGTHS[i])]            

    @property
    def vocab_size(self):
        vocab = len(self._tokens)
        vocab += self.cum_bin_count()
        return vocab

    
    def capacity(self, tokenizer=False):
        # number of bits to store all tokens
        # plus tokenizers with uniform entropy if tokenizer
        total = 0
        for idx, token in self._tokens.items():
            n = Tokenizer.num_bonds(token)
            total += 4*(n+n-1+n-2)*8 # suppose we store bond lengths as floats, each 4 byte
            # n-1 bond angles, n-2 dihedrals
    
        if tokenizer:
            mbits = np.log2(len(self._tokens))
            bbits = np.log2(self.bins[1])
            for t in self.tokenizers:
                tt = t.tokenize()
                m = (len(tt)+3)//4
                total += mbits * m
                total += 3*(m-1)*bbits
        return total
        
    
    def cum_bin_count(self, key=None):
        count = 0
        if self.res_init:
            assert key is None or (key in ["omega", "phi", "C:1N:1CA"])
        for k, v in self._bin_counts[1].items():
            if key == k:
                break
            if self.res_init and k not in ["omega", "phi", "C:1N:1CA"]:
                continue
            count += len(v)
        return count


    def quantize(self, tokenized):
        quantized = []
        for i, token in enumerate(tokenized):
            if token[0] == "MOTIF":
                quant = list(self._tokens).index(token[1])
            else:                
                dt = token[1]
                cum = self.cum_bin_count(dt)
                relv = self._thresholds[1][dt]
                ind = BPE.get_ind((token[2] + 2*np.pi) % (2*np.pi), relv)
                quant = len(self._tokens)+cum+ind
            quantized.append(quant)
        return quantized


    def dequantize(self, quantized):
        cum = self.cum_bin_count()
        num_vocab = self.vocab_size
        tokenized = []
        for i, quant in enumerate(quantized):
            if quant < num_vocab-cum:
                if quant > len(self._tokens):
                    raise ValueError(f"pos {i} > vocab range=\(0, {len(self._tokens)}\)")                                    
                token = ("MOTIF", list(self._tokens)[quant])
            else:
                c = quant-(num_vocab-cum)
                token = None
                for k, v in self._thresholds[1].items():
                    if self.res_init and k not in ["omega", "phi", "C:1N:1CA"]:
                        continue
                    if c < len(v):
                        start, end = v[c]
                        prefix = "DIHEDRAL" if k in Tokenizer.DIHEDRAL_ANGLES else "BOND_ANGLE"
                        token = (prefix, k, (start+end)/2)
                        break
                    c -= len(v)
                if token is None:
                    raise ValueError(f"pos {i} > vocab_size={num_vocab}")                
            tokenized.append(token)
        return tokenized


    def recover(self, tokenized):
        repl = defaultdict(list)
        for token in tokenized:
            if token[0] == "MOTIF":
                bt = token[1]
                key_dict = self._tokens[bt]
                while isinstance(key_dict, str):
                    key_dict = json.loads(key_dict)
                for k in key_dict:
                    repl[k] += key_dict[k]
            else:
                dt = token[1]
                val = token[2]
                repl[dt].append(val)

        repl = dict(repl)  
        return repl

    @staticmethod
    def init_structure(n):
        angles = {
            "0C:1N": [0. for _ in range(n)],
            "N:CA": [0. for _ in range(n)],
            "CA:C": [0. for _ in range(n)],
            "phi": [np.nan for _ in range(n)],
            "psi": [np.nan for _ in range(n)],
            "omega": [np.nan for _ in range(n)],
            "tau": [np.nan for _ in range(n)],
            "CA:C:1N": [np.nan for _ in range(n)],
            "C:1N:1CA": [np.nan for _ in range(n)]
        }
        idxes = sum([[i,i,i] for i in range(1, n+1)], [])
        return {
            "angles": pd.DataFrame(angles),
            "coords": None,
            "c_beta": None,
            "full_idxes": idxes,
            "full_coords": None,
            "side_chain": None,
            "aa": None,
            "fname": None
        }

    def recover_structure(self, repl, tokenized):
        n = len(repl["N:CA"])
        struc = self.init_structure(n)
        # ref.tokenizers[0].angles_and_dists
        struc["angles"]["N:CA"].iloc[:-1] = repl["N:CA"][1:]
        struc["angles"]["CA:C"].iloc[:-1] = repl["CA:C"][1:]
        struc["angles"]["0C:1N"].iloc[:-1] = repl["0C:1N"]
        struc["angles"]["phi"].iloc[1:] = repl["phi"]
        struc["angles"]["psi"].iloc[:-1] = repl["psi"]
        struc["angles"]["omega"].iloc[:-1] = repl["omega"]
        struc["angles"]["tau"].iloc[:-1] = repl["tau"][1:]
        struc["angles"]["CA:C:1N"].iloc[:-1] = repl["CA:C:1N"]
        struc["angles"]["C:1N:1CA"].iloc[:-1] = repl["C:1N:1CA"]
        t_new = Tokenizer(struc)
        t_new.bond_to_token = {}
        cur = 0
        for key, *pargs in tokenized:    
            if key == "MOTIF":
                token_id = pargs[0]
                nb = Tokenizer.num_bonds(self._tokens[token_id]) 
                t_new.bond_to_token[cur] = (cur, token_id, nb)
                cur += nb
        return t_new

    def tokenize(self, t):
        if not self.res_init:
            raise NotImplementedError
        # init bond lengths
        lookup = self._thresholds if getattr(self, "std_bonds", True) else self._thresholds[1]
        relv_thresholds = {bt: lookup[bt] for bt in lookup if bt in Tokenizer.BOND_TYPES}            
        global RELV_THRESHOLDS, STD_BONDS
        RELV_THRESHOLDS, STD_BONDS = relv_thresholds, getattr(self, "std_bonds", True)        
        t = BPE._set_bond_length_worker(t, strict=False)
        R_occs, t_occs = t.exit_frame(3, 3*t.n-4, ret_all=True)
        t.cached_all_frames = (R_occs, t_occs)
        res_geo = defaultdict(list)
        for i in range(t.n): # for each residue
            start = 3*i
            length = 3 if i < t.n-1 else 2                
            res_geo[length].append(start)
        t.token_pos = [3*(i//3) for i in range(3*t.n-1)]
        t.tokens = [(3*i, None, 3) for i in range(t.n-1)] + [(3*t.n-3, None, 2)]
        t.bond_to_token = {t[0]: t for t in t.tokens}
        for n, size in enumerate(res_geo):                
            t.bond_to_token_copy = dict(t.bond_to_token)
            geo = t.token_geo(0, 5) if size == 3 else t.token_geo(0, 2)
            key_coords = []
            p = 0
            while (n, p) in self._tokens:
                key = self._tokens[(n, p)]
                for k in key:
                    geo[k][:len(key[k])] = key[k]
                p += 1
                key_coords.append(Tokenizer.geo_nerf(geo).cartesian_coords[:size+1])
                assert compute_rmsd(key_coords[-1], Tokenizer.key_coords(key)) < 1e-10
            for start in res_geo[size]:
                coords = t.compute_coords(start, size)                    
                costs = [compute_rmsd(coords, c) for c in key_coords]
                p = np.argmin(costs)
                # if t_post_init.bond_to_token[start][1] != (n, p):
                #     breakpoint()
                if start > 0 and self.glue_opt and self.glue_opt_method == "each":
                    R_occ, t_occ = t.exit_frame(start, 3*((size-2)//3)+2)
                    t.set_token_geo(start, size, self._tokens[(n, p)])
                    BPE.opt_glue(t, start, 3*((size-2)//3)+2, R_occ, t_occ)
                else:
                    t.set_token_geo(start, size, self._tokens[(n, p)])
                    t.bond_to_token_copy[start] = t.tokens[start//3]
                t.tokens[start//3] = (start, (n, p), size)
                t.bond_to_token_copy[start] = t.tokens[start//3]
            t.bond_to_token = t.bond_to_token_copy
        delattr(t, "bond_to_token_copy")
        if not self.glue_opt:
            keys = ['omega', 'C:1N:1CA', 'phi']
            for angle in keys:
                relv_thresh = self._thresholds[1][angle]
                t.angles_and_dists[angle] = [sum(relv_thresh[BPE.get_ind(
                        v if angle in Tokenizer.BOND_TYPES else (v+2*np.pi)%(2*np.pi), 
                        relv_thresh
                    )])/2 if v==v else v for v in t.angles_and_dists[angle]]
        elif self.glue_opt_method == "all":
            global BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR
            BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR = self._bin_centers, self._thresholds, self.glue_opt_prior
            t = BPE._opt_glue_worker(pickle_copy(t))
        # .bin() for single tokenizer
        geo_dict = self.bin_helper(t)
        # iterate through all keys in order
        uniq_keys = sorted(set(map(lambda t: t[0], self._tokens)))
        geo_keys = list(self._sphere_dict) # assume in order of addition
        assert len(uniq_keys) == len(geo_keys)
        count = 0
        for n, key in tqdm(zip(uniq_keys, geo_keys), desc="iterating through keys", total=len(uniq_keys)):
            if n < 2:
                continue            
            if key in geo_dict:
                # if in geo_dict, .step() for single tokenizer (without popping new key)                
                print("step start")
                t = self.step_helper(geo_dict, t, key, n, opt=count%self.glue_opt_every == 0)
                count += 1
                print("step end")

        # t_true = self.tokenizers[next((i for i in range(len(self.tokenizers)) if self.tokenizers[i].fname == t.fname))]
        # assert t.bond_to_token == t_true.bond_to_token
        # assert (t.angles_and_dists != t_true.angles_and_dists).sum().sum() == 6
            
        

    @staticmethod
    def hash_geo(geo):
        return json.dumps(geo, sort_keys=True)

    @staticmethod
    def get_ind(v, values):
        ind = -1        
        for l, (start, end) in enumerate(values):
            if (v == start) or (v >= start and v<end):
                ind = l
                break
        if ind == -1:
            if v != end:
                breakpoint()
            ind = len(values)-1
        return ind
    

    def compute_geo_key(self, token_pair, tokenizer_or_index, ignore_left=False, ignore_right=False):
        """
        Compute the quantized geo key for token_pair
        token_pair: (idx1, _, l1) and (idx2, _, l2)
        can be old tokens  
        ignore_left: whether to ignore the fact the left token is in a partitioned token   
        ignore_left: same for right   
        """
        if isinstance(tokenizer_or_index, int):
            t = self.tokenizers[tokenizer_or_index]
        else:
            t = tokenizer_or_index
        try:
            (idx1, _, l1), (idx2, _, l2) = token_pair
        except:
            return None
        assert idx1+l1 == idx2
        # what if same token?
        if t.token_pos[idx1] == t.token_pos[idx2]:
            breakpoint()
        # (idx1, l1) should be contained in exactly 1 token
        # (idx2, l2) should be contained in exactly 1 token
        tok1 = t.bond_to_token[t.token_pos[idx1]]
        tok2 = t.bond_to_token[t.token_pos[idx2]]
        bt1, bt2 = tok1[1], tok2[1]
        pt1 = not ignore_left and isinstance(bt1, tuple) # partition key
        pt2 = not ignore_right and isinstance(bt2, tuple) # partition key
        # two cases: idx1, l1 is a token or (idx1, idx1+l1) in a token sharing one endpoint
        assert (tok1[0] <= idx1 and tok1[0]+tok1[2] == idx1+l1)
        assert (tok2[0] == idx2 and tok2[2] >= l2)                
        # ind_pair: which tokens there're part of
        # if given (when partitioned), we find which tokens they're part of
        # if part of a partitioned token, we need the actual values
        geo = t.token_geo(idx1, l1+l2) # raw geometry
        for k in geo:
            quant_vals = []
            for i, v in enumerate(geo[k]):
                # if pt1 and pt2:
                #     # bonds 0,...,l1+l2-1
                #     # bond angles 0,...,l1-2,l1,...,l1+l2-2
                #     # dihedral angles 0,...,l1-3,l1,...,l1+l2-3
                # elif pt1:
                #     # bonds 0,...,l1-1
                #     # bond angles 0,...,l1-2
                #     # dihedral angles 0,...,l1-3
                # elif pt2:
                #     # bonds l1,...,l1+l2-1
                #     # bond angles l1,...,l1+l2-2
                #     # dihedral angles l1, ..., l1+l2-3
                
                # convert i into bond/angle index
                # bond ((Tokenizer.BOND_TYPES.index(k)+3)-idx1%3)%3 + 3*i is the i'th type-k bond
                # bond angle ((Tokenizer.BOND_ANGLES.index(k)+3)-idx1%3)%3 + 3*i is the i'th type-k bond angle
                # dihedral angle ((Tokenizer.DIHEDRAL_ANGLES.index(k)+3)-idx1%3)%3 + 3*i is the i'th type-k dihedral angle
                
                quant = False
                if k in Tokenizer.BOND_TYPES:
                    i = ((Tokenizer.BOND_TYPES.index(k)+3)-idx1%3)%3 + 3*i
                    if pt1 and pt2:
                        pass
                    elif pt1:
                        if i >= l1:
                            quant = True
                    elif pt2:
                        if i < l1:
                            quant = True
                    else:
                        quant = True # fixed                    
                elif k in Tokenizer.BOND_ANGLES:
                    i = ((Tokenizer.BOND_ANGLES.index(k)+3)-idx1%3)%3 + 3*i
                    if pt1 and pt2:
                        if i == l1-1:
                            quant = True
                    elif pt1:
                        if i >= l1-1:
                            quant = True
                    elif pt2:
                        if i < l1:
                            quant = True
                    else:
                        quant = True
                else:
                    i = ((Tokenizer.DIHEDRAL_ANGLES.index(k)+3)-idx1%3)%3 + 3*i
                    if pt1 and pt2:
                        if i == l1-2 or i == l1-1:
                            quant = True
                    elif pt1:
                        if i >= l1-2:
                            quant = True
                    elif pt2:
                        if i < l1:
                            quant = True
                    else:
                        quant = True
                if quant:
                    if getattr(self, "std_bonds", True) and k in Tokenizer.BOND_TYPES:
                        lookup = self._thresholds
                    else:
                        if k not in Tokenizer.BOND_TYPES:
                            v = (v+2*np.pi) % (2*np.pi) # Convert to [0, 2*pi]
                        lookup = self._thresholds[l1+l2]
                    ind = BPE.get_ind(v, lookup[k])
                    quant_vals.append(ind)
                else:
                    quant_vals.append(v)
            geo[k] = quant_vals
        geo_key = self.hash_geo(geo)
        return geo_key
    
    def bin_helper(self, t):
        # bin but as a helper function for given tokenizer t
        geo_dict = defaultdict(set)
        p1 = 0 # first ptr
        p2 = p1+1
        for j in range(len(t.bond_to_token)-1): # num tokens
            while t.token_pos[p2] == p1:
                p2 += 1
            token_pair = (t.bond_to_token[p1], t.bond_to_token[p2])
            geo_key = self.compute_geo_key(token_pair, t)
            (i1, _, l1), (i2, _, l2) = token_pair
            geo_dict[geo_key].add(i2) # store the start of the second token
            p1 = p2
        return geo_dict
    
    def step_helper(self, geo_dict, t, key, n, opt=False):
        """
        geo_dict: similar to self._geo_dict but for t only
        t: tokenizer
        key: which freq key to step on
        """
        super_res = self.rmsd_super_res if hasattr(self, "rmsd_super_res") else False
        key_dict = json.loads(key)
        length = Tokenizer.num_bonds(key_dict)
        vals = list(geo_dict[key])
        rmsd_key = key
        medoid_coords = [Tokenizer.key_coords(geo) for geo in self._sphere_dict[key]]
        assignments = [
            np.argmin([compute_rmsd(t.compute_coords(t.token_pos[index-1], length, orig=super_res), med) for med in medoid_coords])
                for index in vals
        ]
        sort_val_idxes = sorted(range(len(vals)), key=lambda i: vals[i])        
        last_i1 = None
        for idx in sort_val_idxes:  # _geo_dict is sorted
            index = vals[idx]
            p = assignments[idx]
            i2 = index  # start of second token
            i1 = t.token_pos[index-1]  # start of prev token
            l1 = i2 - i1
            l2 = length - l1
            overlaps = last_i1 is not None and (last_i1+length > i1) # overlaps
            not_present = index not in geo_dict[key]
            if overlaps != not_present:
                breakpoint()
            if overlaps:
                continue
            else:
                assert ((l1 > 0) and (l2 > 0) and (l1 + l2 == length))                
            geo_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t)
            if geo_key != key: # should never happen
                breakpoint()
                continue

            # --- Step 1: Decrement key ---
            geo_dict[key].remove(index)

            # --- Step 3 and 4: Decrement left and right key pair freq. ---
            if i1:
                i0 = t.token_pos[i1 - 1]
                l0 = i1 - i0
                left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1)), t)
            else:
                left_key = None
            if i2 + l2 < len(t.token_pos):
                i3 = i2 + l2
                l3 = 0
                while i3 + l3 < len(t.token_pos) and t.token_pos[i3 + l3] == i3:
                    l3 += 1
                right_key = self.compute_geo_key(((i2, None, l2), (i3, None, l3)), t) # 
            else:
                right_key = None

            if left_key:
                geo_dict[left_key].remove(i1)
            if right_key:
                geo_dict[right_key].remove(i3)

            # --- Step 2: Update t.token_pos and t.bond_to_token ---
            for j in range(i2, i2 + l2):
                t.token_pos[j] = i1                 
            t.bond_to_token.pop(i2)
            t.bond_to_token[i1] = (i1, (n, p), length)

            # --- Step 6: For RMSD keys, set geo now
            i1 = t.token_pos[index-1]
            if not self.rmsd_only:
                if i1 > 0 and self.glue_opt and self.glue_opt_method == "each":
                    # update now
                    R_occ, t_occ = t.exit_frame(i1, 3*((length-2)//3)+2)
                    t.set_token_geo(i1, length, self._sphere_dict[key][p])
                    BPE.opt_glue(t, i1, 3*((length-2)//3)+2, R_occ, t_occ, self.glue_opt_prior, self._bin_centers, self._thresholds)
                else:
                    t.set_token_geo(i1, length, self._sphere_dict[key][p])
            # --- Step 5: Increment new pairs ---                     
            if left_key:
                new_left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1 + l2)), t)
                geo_dict[new_left_key].add(i1)
            if right_key:
                assert i1+l1+l2 == i3
                new_right_key = self.compute_geo_key(((i1, None, l1 + l2), (i3, None, l3)), t)
                geo_dict[new_right_key].add(i3)
                           
            last_i1 = i1

        # -- Step 6 (cont.) for opt_glue
        if not self.rmsd_only and self.glue_opt and self.glue_opt_method == "all" and opt:
            global BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR
            BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR = self._bin_centers, self._thresholds, self.glue_opt_prior                 
            t_new = BPE._opt_glue_worker(pickle_copy(t))
            # find all the changed (dihedral, omega, C-N-CA) triples
            for i1 in t.bond_to_token:
                _, _, l1 = t.bond_to_token[i1]
                if i1+l1 == 3*t.n-1:
                    continue
                i2, _, l2 = t.bond_to_token[i1+l1]
                new_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t_new)
                old_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t)
                if new_key != old_key:
                    geo_dict[old_key].remove(i2)
                    geo_dict[new_key].add(i2)
                # in some rare instances, new_key is an existing token, so we max out its priority
                if new_key in self._sphere_dict:
                    pass
            t = t_new
        return t
                
                    
        


    def bin(self):
        """
        Loop over all tokenizers' current token pairs from scratch
        Bin each of their geometries
        Insert (binned_geometry, count) into pqueue
        """
        self._geo_dict = defaultdict(set)        
        self._priority_dict = SortedDict() # like c++ map, constant largest element, log n arbitrary element insertion/deletion
        self._key_to_priority = {}
        self._geo_step = {}
        for i in range(self.n):
            t = self.tokenizers[i]
            # t.token_pos
            p1 = 0 # first ptr
            p2 = p1+1
            for j in range(len(t.bond_to_token)-1): # num tokens
                while t.token_pos[p2] == p1:
                    p2 += 1
                token_pair = (t.bond_to_token[p1], t.bond_to_token[p2])
                geo_key = self.compute_geo_key(token_pair, i)
                (i1, _, l1), (i2, _, l2) = token_pair
                self._geo_dict[geo_key].add((i, i2)) # store the start of the second token
                p1 = p2
            assert p1 == t.token_pos[-1] # reached the end
        for key in self._geo_dict:
            self._geo_step[key] = 0
        # self._pqueue = [(-len(self._geo_dict[key]), key, 0) for key in self._geo_dict]        
        for key in self._geo_dict:
            length = Tokenizer.num_bonds(json.loads(key))            
            if self.compute_sec_structs:
                # priority: -(already defined, #(secondary membership), #(membership), key)
                sec_memb = 0
                for i, i2 in self._geo_dict[key]:
                    t = self.tokenizers[i]
                    i1 = t.token_pos[i2-1]
                    sec_memb += t.is_secondary(i1, length)
                priority = (True, -sec_memb, -len(self._geo_dict[key]), key)
            else:
                priority = (True, -len(self._geo_dict[key]), key)
            self._priority_dict[priority] = None # not useful
            self._key_to_priority[key] = priority
        # heapq.heapify(self._pqueue)
        # for RMSD packing
        self._sphere_keys = {}

    def old_bin(self):
        """
        Loop over all tokenizers' current token pairs from scratch
        Bin each of their geometries
        Insert (binned_geometry, count) into pqueue
        """
        self._geo_dict = defaultdict(list)
        self._geo_step = {}
        for i in range(self.n):
            t = self.tokenizers[i]
            # t.token_pos
            p1 = 0 # first ptr
            p2 = p1+1
            for j in range(len(t.tokens)-1): # num tokens
                token_pair = (t.tokens[j], t.tokens[j+1])
                geo_key = self.compute_geo_key(token_pair, i)
                (i1, _, l1), (i2, _, l2) = token_pair
                self._geo_dict[geo_key].append((i, i2)) # store the start of the second token
        # for key in self._geo_dict:
        #     self._geo_step[key] = 0
        self._pqueue = [(-len(self._geo_dict[key]), key) for key in self._geo_dict]
        heapq.heapify(self._pqueue)        


    def _bin_val(self, key_dict):
        size = Tokenizer.num_bonds(key_dict)
        key_dict_copy = {}
        for k, vals in key_dict.items():
            # assert np.all([isinstance(v, int) for v in vals])
            if k in Tokenizer.BOND_TYPES and self.std_bonds:
                relv_thresholds = self._thresholds
            else:
                relv_thresholds = self._thresholds[size]
            key_dict_copy[k] = [sum(relv_thresholds[k][v])/2 if isinstance(v, int) else v for v in vals]
        return key_dict_copy


    def quant_geo(self, geo):
        length = Tokenizer.num_bonds(geo)
        for k in geo:
            quant_vals = []
            for i, v in enumerate(geo[k]):
                if k in Tokenizer.BOND_TYPES:
                    lookup = self._thresholds
                else:
                    lookup = self._thresholds[length]
                    v = (v+2*np.pi) % (2*np.pi) # Convert to [0, 2*pi]
                # quantize val
                ind = BPE.get_ind(v, lookup[k])
                quant_vals.append(ind)      
            geo[k] = quant_vals    
    

    def plot_times(self, output_path):
        fig, ax = plot_times(self._times)
        ax.set_xlabel("BPE Step")
        ax.set_ylabel("Log Seconds")
        ax.set_title("Time per step")        
        fig.savefig(output_path)
    

    def plot_iou(self, output_path):
        # self._ious is assumed to be an (N, 5) NumPy array
        stats = np.array(self._ious)
        timesteps = np.arange(stats.shape[0])        
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, stats[:, 0], label='Min')
        plt.plot(timesteps, stats[:, 1], label='Q1')
        plt.plot(timesteps, stats[:, 2], label='Median')
        plt.plot(timesteps, stats[:, 3], label='Q3')
        plt.plot(timesteps, stats[:, 4], label='Max')        
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.title('Evolution of Secondary Structure IOUs Over Timesteps')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    
    def compute_iou(self):
        all_ious = []
        for i in range(self.n):
            t = self.tokenizers[i]
            best_ious = []
            for s in range(len(t.sec_types)):
                start, end = t.sec_bond_range[s]
                if end-start+1 < 5: # at least two residues
                    continue
                first = t.token_pos[start]
                last = t.token_pos[end]
                best_iou = 0
                while first <= last:
                    _, _, length = t.bond_to_token[first] # first,...,first+length-1
                    intersection = max(0, min(end, first+length-1) - max(start, first) + 1)                    
                    # Calculate the union length
                    union = (end-start+1) + length - intersection                    
                    # Compute IoU
                    iou = intersection / union if union > 0 else 0
                    best_iou = max(iou, best_iou)
                    first += length
                best_ious.append(best_iou)
            all_ious += best_ious
        summary = (np.min(all_ious), np.percentile(all_ious, 25), np.percentile(all_ious, 50), np.percentile(all_ious, 75), np.max(all_ious))
        self._ious.append(summary)
    

    def visualize(self, key, output_path):
        """
        Visualize a key, take the average of the bin
        Use _geo_dict to get a grounded occurrence
        """
        vals = self._geo_dict[key] # check ValueError
        key_dict = json.loads(key)
        key_dict = self._bin_val(key_dict)
        # since geo_nerf expects 3*n-1 bonds, we do a similar procedure as in Tokenizer.compute_coords, rounding and offsetting
        i, index = next(iter(vals))
        length = Tokenizer.num_bonds(key_dict)
        t = self.tokenizers[i]
        # index is the start of second token
        i1 = t.token_pos[index-1] # start of token pair
        start = 3*(i1//3)
        end = 3*(((i1+length-1)+1)//3)+1 # end bond id, but we round it up so it's 1 (mod 3)
        off_start = i1-start
        off_end = end-(i1+length-1)
        geo = t.token_geo(start, end-start+1) # round from nearest residues
        for k in geo:
            if k in key_dict:
                n = len(key_dict[k])
                if k in Tokenizer.BOND_TYPES:
                    bt_index = Tokenizer.BOND_TYPES.index(k)           
                    geo[k] = [Tokenizer.BOND_LENGTHS[bt_index] for _ in geo[k]] 
                elif k in Tokenizer.BOND_ANGLES:
                    bt_index = Tokenizer.BOND_ANGLES.index(k)           
                else:
                    bt_index = Tokenizer.DIHEDRAL_ANGLES.index(k)                       
                # what is first idx that key_dict defines
                first = i1%3
                if first > bt_index: # off by 1 from start
                    assert np.abs(len(geo[k])-n) <= 2.
                    for j in range(n):
                        geo[k][1+j] = key_dict[k][j]
                else: # off by 0 from start
                    assert np.abs(len(geo[k])-n) <= 1.
                    for j in range(n):
                        geo[k][j] = key_dict[k][j]
        geo_nerf = Tokenizer.geo_nerf(geo)
        coords = geo_nerf.cartesian_coords
        bts = np.tile(Tokenizer.ATOM_TYPES, (end-start+2)//3)
        bts = bts[off_start: len(bts)-off_end]
        coords = coords[off_start: len(coords)-off_end] # offset
        plot_backbone(coords, output_path, bts, zoom_factor=1.0)  

    
    def _update(self):
        """
        Update the priority queue
        """
        breakpoint()

    @staticmethod
    def remove_values(sorted_list, remove_list):
        """
        Returns a new list which is sorted_list with any elements found in remove_list removed.
        Both lists are assumed to be sorted.
        """
        i, j = 0, 0
        n, m = len(sorted_list), len(remove_list)
        result = []        
        while i < n and j < m:
            if sorted_list[i] < remove_list[j]:
                # Keep this element because it's not in remove_list
                result.append(sorted_list[i])
                i += 1
            elif sorted_list[i] > remove_list[j]:
                # Move remove_list pointer, because we're past this remove_list[j] in sorted_list
                j += 1
            else:
                # They are equal, so we skip this element in sorted_list
                i += 1
                j += 1
        # If there are leftover elements in sorted_list that couldn't possibly be in remove_list
        while i < n:
            result.append(sorted_list[i])
            i += 1
        return result    

    @staticmethod    
    def add_values(sorted_list, add_list):
        """
        Returns a new list which is the "sorted union" of sorted_list and add_list,
        skipping duplicates (typical 'merge' behavior).
        
        If you want to *include* duplicates, slightly tweak the condition where they are equal.
        """
        i, j = 0, 0
        n, m = len(sorted_list), len(add_list)
        result = []        
        while i < n and j < m:
            if sorted_list[i] < add_list[j]:
                result.append(sorted_list[i])
                i += 1
            elif sorted_list[i] > add_list[j]:
                result.append(add_list[j])
                j += 1
            else:
                # They are equal, so just add one of them and advance both
                result.append(sorted_list[i])
                i += 1
                j += 1
        # Add the remaining elements from either list
        while i < n:
            result.append(sorted_list[i])
            i += 1
        while j < m:
            result.append(add_list[j])
            j += 1
        return result  

    def old_step(self):
        """
        Naive implementation: recompute bin every step
        """
        logger.info(f"Old Step {self._step} start")
        step_start_time = time.time()       
        count, key = heapq.heappop(self._pqueue)
        key_dict = json.loads(key)
        length = Tokenizer.num_bonds(key_dict)
        inds = [0 for _ in range(self.n)]
        # define new token
        n = len(self._tokens)
        self._tokens[n] = key_dict          
        for i, index in self._geo_dict[key]: # _geo_dict[key] is sorted
            t = self.tokenizers[i]
            i1 = t.token_pos[index-1]
            l1 = index-i1
            i2 = index
            l2 = length-l1
            if (l1 == 0) or (l2 == 0) or (l1 + l2 != length):
                continue            
            geo_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), i)
            if geo_key != key:
                continue
            # update tokens, token_pos, bond_to_token
            while t.tokens[inds[i]] is None or t.tokens[inds[i]][0] < i1: # find where it is
                inds[i] += 1
            assert t.tokens[inds[i]][0] == i1
            t.tokens[inds[i]] = (i1, n, length)
            t.tokens[inds[i]+1] = None
            for j in range(i2, i2+l2):
                t.token_pos[j] = i1
            t.bond_to_token.pop(i2)
            t.bond_to_token[i1] = (i1, n, length)
            # standardize occurrence
            t.set_token_geo(i1, l1+l2, self._bin_val(key_dict))        
        for i in range(self.n):
            t = self.tokenizers[i]
            t.tokens = list(filter(None, t.tokens))            
        self._step += 1
        self.old_bin()        
        logger.info(f"Old Step {self._step-1} took {time.time()-step_start_time}")
    

    def rmsd_partition(self, k):
        logger.info(f"Start partitioning {k}")
        super_res = self.rmsd_super_res if hasattr(self, "rmsd_super_res") else False
        key_dict = json.loads(k)
        length = Tokenizer.num_bonds(key_dict)
        all_pos = list(self._geo_dict[k]) # for indexing        
        N = len(all_pos)
        if N > self.max_num_strucs:
            active_inds = self.rng.choice(N, self.max_num_strucs, replace=False)
        else:
            active_inds = np.arange(N)        
        active_occurs = []
        active_coords = []
        for i in tqdm(active_inds, desc=f"gathering {len(active_inds)} occurrences"):
            ti, index = all_pos[i]
            t = self.tokenizers[ti]
            i1 = t.token_pos[index-1]
            coords = t.compute_coords(i1, length, orig=super_res)
            active_occurs.append((ti, i1))
            active_coords.append(coords)
        
        # run k-medoids
        medoid_inds = k_medoids(active_coords, self.num_partitions[length], rng=self.rng)
        medoids = [active_inds[m] for m in medoid_inds]
        assignments = [None for _ in range(N)]
        max_workers = _effective_cpus()
        if max_workers == 0:
            for idx, (ti, index) in enumerate(all_pos):
                assignments[idx] = BPE._compute_assignment((self.tokenizers[ti], self.tokenizers[ti].token_pos[index-1], length), active_coords, medoid_inds, super_res)
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=BPE._init_assignment_worker,
                initargs=(active_coords, medoid_inds, super_res, self.tokenizers)
            ) as pool:
                futures = {
                    pool.submit(
                        BPE._compute_assignment,
                        (ti, self.tokenizers[ti].token_pos[index-1], length)
                    ): idx
                    for idx, (ti, index) in enumerate(all_pos)
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                            desc="computing assignments"):
                    idx = futures.pop(future)
                    assignments[idx] = future.result()

        # _sphere_dict
        self._sphere_dict[k] = []
        for p, m in enumerate(medoid_inds):
            ti, i1 = active_occurs[m]
            t = self.tokenizers[ti]
            struc = t.token_geo(i1, length, orig=super_res)
            logger.info(f"Partition {p}: {struc}")
            self._sphere_dict[k].append(struc)
            key_vis_path = os.path.join(self.save_dir, f"key_iter={self._step}_{p}.png")
            t.visualize_bonds(i1, length, key_vis_path, orig=super_res)
        return assignments


    def step(self):
        logger.info(f"Step {self._step} start")
        step_start_time = time.time()        
        max_workers = _effective_cpus()
        key_tup = self._priority_dict.peekitem(0)
        if self.compute_sec_structs:
            (exists, num_sec, count, key), _ = key_tup
            recurring_key = not exists # whether already defined
            count = -count       
            assert count == len(self._geo_dict[key])
            # self._bin_side_chain(key) # later add to _tokens
            logger.info(f"Pop {key} as most frequent key, {count} times, {num_sec} secondary structures")        
        else:
            (exists, count, key), _ = key_tup
            recurring_key = not exists
            if key in self._tokens.values():
                breakpoint()
            count = -count
            assert count == len(self._geo_dict[key])
            logger.info(f"Pop {key} as most frequent key, {count} times")  
            # TODO: Should never pop again
        key_dict = json.loads(key)
        length = Tokenizer.num_bonds(key_dict)        
        if not recurring_key: # new key
            # visualizations
            vis_start_time = time.time()            
            key_vis_path = os.path.join(self.save_dir, f"key_iter={self._step}.png")
            self.visualize(key, key_vis_path)
            vis_time = time.perf_counter()-vis_start_time
        # --- Step 0: Do RMSD packing if |token| >= rmsd_partition_min_size ---
        if Tokenizer.num_bonds(key_dict) >= self.rmsd_partition_min_size:
            if recurring_key:
                assert key in self._sphere_dict # very rare for this to happen
                # compute assignments directly without partitioning
                active_coords = [Tokenizer.key_coords(kk) for kk in self._sphere_dict[key]]
                medoid_inds = list(range(len(active_coords)))
                super_res = self.rmsd_super_res if hasattr(self, "rmsd_super_res") else False
                all_pos = list(self._geo_dict[key])
                N = len(all_pos)
                assignments = [None for _ in range(N)]
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=BPE._init_assignment_worker,
                    initargs=(active_coords, medoid_inds, super_res, self.tokenizers)
                ) as pool:
                    futures = {
                        pool.submit(
                            BPE._compute_assignment,
                            (ti, self.tokenizers[ti].token_pos[index-1], length)
                        ): idx
                        for idx, (ti, index) in enumerate(all_pos)
                    }
                    for future in tqdm(as_completed(futures), total=len(futures),
                                desc="computing assignments"):
                        idx = futures.pop(future)
                        assignments[idx] = future.result()                
            else:
                # Step 0.1: populate _sphere_dict
                # Step 0.2: update geo
                # Step 0.3: update _geo_dict
                assignments = self.rmsd_partition(key)
            rmsd_key = key
        else:
            rmsd_key = None
        # Define new token                        
        n = len(self._tokens)
        if rmsd_key is None:
            binned_key_dict = self._bin_val(key_dict) # get the values
            self._tokens[n] = key_dict        
        else: # create a collection of tokens as long as key isn't already present
            if recurring_key:                
                n_ind = list(self._sphere_dict).index(key) # find the n that self._sphere_dict[key][0] is
                n = sorted(set(map(lambda x: x[0], filter(tuple, self._tokens))))[n_ind]
                for p in range(len(self._sphere_dict[list(self._sphere_dict)[n_ind]])):
                    assert self._tokens[(n, p)] == self._sphere_dict[list(self._sphere_dict)[n_ind]][p]
            else:
                for p, token_p in enumerate(self._sphere_dict[key]):
                    self._tokens[(n, p)] = token_p                
        # standardize and replace all occurrences
        # update 3 data structures: t.token_pos, _geo_dict, and pqueue
        # we will track which keys to remove/add, and later infer change to pqueue
        diff_count = defaultdict(list)
        # time
        step_times = {
            "sort": 0.0,
            "step1": 0.0,      # Decrement key.
            "step2": 0.0,      # Update t.token_pos and t.tokens.
            "step3_4": 0.0,    # Decrement left and right key pair freq.
            "step5": 0.0,      # Increment new pairs.
            "step6": 0.0,      # Standardize occurrence by binning key_dict.
            # "update_tokens": 0.0,  # Update t.tokens for each tokenizer.
            "step7": 0.0,      # Update priority queue.
            "step8": 0.0
        }        
        # we will running update token_pos
        start_time = time.perf_counter()
        vals = list(self._geo_dict[key])
        uniq_idxes = set((v[0] for v in vals))
        # if rmsd_key is not None and not self.rmsd_only and self.glue_opt and self.glue_opt_method == "all":
        #     for idx in uniq_idxes:
        #         t = self.tokenizers[idx]
        #         R_occs, t_occs = t.exit_frame(3, 3*t.n-4, ret_all=True)
        #         t.cached_all_frames = (R_occs, t_occs)        
        sort_val_idxes = sorted(range(len(vals)), key=lambda i: vals[i])
        step_times["sort"] += time.perf_counter() - start_time
        
        last_i, last_i1 = None, None
        for idx in sort_val_idxes:  # _geo_dict is sorted
            (i, index) = vals[idx]
            if rmsd_key is not None:
                p = assignments[idx]
            t = self.tokenizers[i]
            i2 = index  # start of second token
            i1 = t.token_pos[index-1]  # start of prev token
            l1 = i2 - i1
            l2 = length - l1
            # double-check that token pair is still valid (not valid if prev (i, index) overlaps)
            overlaps = (last_i is not None) and (last_i == i) and (last_i1+length > i1) # overlaps
            not_present = (i, index) not in self._geo_dict[key]
            if overlaps != not_present:
                breakpoint()
            if overlaps:
                continue
            else:
                assert ((l1 > 0) and (l2 > 0) and (l1 + l2 == length))                
            geo_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), i)
            if geo_key != key: # should never happen
                breakpoint()
                continue

            # --- Step 1: Decrement key ---
            start_time = time.perf_counter()
            self._geo_dict[key].remove((i, index))
            diff_count[key].append((i, i1, length, "remove"))
            step_times["step1"] += time.perf_counter() - start_time

            # --- Step 3 and 4: Decrement left and right key pair freq. ---
            start_time = time.perf_counter()
            if i1:
                i0 = t.token_pos[i1 - 1]
                l0 = i1 - i0
                left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1)), i)
            else:
                left_key = None
            if i2 + l2 < len(t.token_pos):
                i3 = i2 + l2
                l3 = 0
                while i3 + l3 < len(t.token_pos) and t.token_pos[i3 + l3] == i3:
                    l3 += 1
                right_key = self.compute_geo_key(((i2, None, l2), (i3, None, l3)), i) # 
            else:
                right_key = None

            if left_key:
                self._geo_dict[left_key].remove((i, i1))
                diff_count[left_key].append((i, i0, l0+l1, "remove"))
            if right_key:
                self._geo_dict[right_key].remove((i, i3))
                diff_count[right_key].append((i, i2, l2+l3, "remove"))
            step_times["step3_4"] += time.perf_counter() - start_time

            # --- Step 2: Update t.token_pos and t.bond_to_token ---
            start_time = time.perf_counter()
            for j in range(i2, i2 + l2):
                t.token_pos[j] = i1                 
            # assert t.tokens[ind][0] == i1
            # rmsd_left_key = isinstance(t.tokens[prev_token_ind][1], tuple) or isinstance(t.tokens[ind][1], tuple)
            # save_tokens = deepcopy(t.tokens)
            # rmsd_right_key = isinstance(t.tokens[ind+1][1], tuple) or isinstance(t.tokens[ind+2][1], tuple)
            # if rmsd_key is not None:
            #     t.tokens[ind] = (i1, (n, p), length)
            # else:
            # t.tokens[ind] = (i1, n, length)
            # t.tokens[ind + 1] = None
            t.bond_to_token.pop(i2)
            if rmsd_key is not None:                
                t.bond_to_token[i1] = (i1, (n, p), length)
            else:
                t.bond_to_token[i1] = (i1, n, length)
            step_times["step2"] += time.perf_counter() - start_time            

            # --- Step 6: For RMSD keys, set geo now
            if rmsd_key is not None:
                start_time = time.perf_counter()
                i1 = t.token_pos[index-1]
                if not self.rmsd_only:
                    if i1 > 0 and self.glue_opt and self.glue_opt_method == "each":
                        # update now
                        R_occ, t_occ = t.exit_frame(i1, 3*((length-2)//3)+2)
                        t.set_token_geo(i1, length, self._sphere_dict[key][p])
                        BPE.opt_glue(t, i1, 3*((length-2)//3)+2, R_occ, t_occ, self.glue_opt_prior, self._bin_centers, self._thresholds)
                    else:
                        t.set_token_geo(i1, length, self._sphere_dict[key][p])
                step_times["step6"] += time.perf_counter() - start_time    
            
            # if i == 0 and t.bond_to_token[235] == (235, (41, 2), 6) and t.bond_to_token[241] == (241, 103, 24) and t.bond_to_token[265] == (265, 11, 3):
            #     breakpoint()

            # --- Step 5: Increment new pairs ---                     
            start_time = time.perf_counter()
            if left_key:
                # if l0+l1+l2 >= self.rmsd_partition_min_size:
                #     new_left_key = self.hash_geo(t.token_geo(i0, l0+l1+l2))
                # else:
                new_left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1 + l2)), i)
                self._geo_dict[new_left_key].add((i, i1))
                diff_count[new_left_key].append((i, i0, l0+l1+l2, "add"))
            if right_key:
                # if l1+l2+l3 >= self.rmsd_partition_min_size:
                #     new_right_key = self.hash_geo(t.token_geo(i1, l1+l2+l3))
                # else:
                assert i1+l1+l2 == i3
                new_right_key = self.compute_geo_key(((i1, None, l1 + l2), (i3, None, l3)), i)
                self._geo_dict[new_right_key].add((i, i3))
                diff_count[new_right_key].append((i, i1, l1+l2+l3, "add"))
                           
            step_times["step5"] += time.perf_counter() - start_time

            # --- Step 6: Standardize occurrence by binning key_dict ---            
            if rmsd_key is None:
                start_time = time.perf_counter()
                t.set_token_geo(i1, l1 + l2, binned_key_dict)
                step_times["step6"] += time.perf_counter() - start_time

            last_i = i
            last_i1 = i1

        # # --- Update t.tokens for each tokenizer (Step 2 cont.) ---
        # start_time = time.perf_counter()
        # for i in range(self.n):
        #     t = self.tokenizers[i]
        #     t.tokens = list(filter(None, t.tokens))
        # step_times["update_tokens"] += time.perf_counter() - start_time

        # -- Step 6 (cont.) for opt_glue
        if rmsd_key is not None and not self.rmsd_only and self.glue_opt and self.glue_opt_method == "all" and (self._step % self.glue_opt_every == 0):
            if max_workers == 0:
                global BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR
                BIN_CENTERS, THRESHOLDS, GLUE_OPT_PRIOR = self._bin_centers, self._thresholds, self.glue_opt_prior
                for idx in uniq_idxes:
                    t = self.tokenizers[idx]                    
                    t_new = BPE._opt_glue_worker(pickle_copy(t))
                    # find all the changed (dihedral, omega, C-N-CA) triples
                    for i1 in t.bond_to_token:
                        _, _, l1 = t.bond_to_token[i1]
                        if i1+l1 == 3*t.n-1:
                            continue
                        i2, _, l2 = t.bond_to_token[i1+l1]
                        new_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t_new)
                        old_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t)
                        if new_key != old_key:
                            self._geo_dict[old_key].remove((idx, i2))
                            diff_count[old_key].append((idx, i1, l1+l2, "remove"))
                            self._geo_dict[new_key].add((idx, i2))
                            diff_count[new_key].append((idx, i1, l1+l2, "add"))
                        # in some rare instances, new_key is an existing token, so we max out its priority later

                    self.tokenizers[idx] = t_new
            else:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=BPE._init_opt_glue_worker,
                    initargs=(self._bin_centers, self._thresholds, self.glue_opt_prior)
                ) as pool:
                    for idx, t_new in zip(uniq_idxes, tqdm(pool.map(BPE._opt_glue_worker, [self.tokenizers[idx] for idx in uniq_idxes], chunksize=10), total=len(uniq_idxes), desc="optimizing all glues")):
                        t = self.tokenizers[idx]
                        # find all the changed (dihedral, omega, C-N-CA) triples
                        for i1 in t.bond_to_token:
                            _, _, l1 = t.bond_to_token[i1]
                            if i1+l1 == 3*t.n-1:
                                continue
                            i2, _, l2 = t.bond_to_token[i1+l1]
                            new_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t_new)
                            old_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), t)
                            if new_key != old_key:
                                self._geo_dict[old_key].remove((idx, i2))
                                diff_count[old_key].append((idx, i1, l1+l2, "remove"))
                                self._geo_dict[new_key].add((idx, i2))
                                diff_count[new_key].append((idx, i1, l1+l2, "add"))                    
                        self.tokenizers[idx] = t_new
                    

        if not recurring_key:
            self._step += 1

        # --- Step 7: Update priority queue ---
        start_time = time.perf_counter()
        new_keys = []
        for k in diff_count:
            # if diff_count[k] > 0:
                # logger.info(f"+{diff_count[k]}={len(self._geo_dict[k])} occurrences of {k}")
                # heapq.heappush(self._pqueue, (-len(self._geo_dict[k]), k, self._step))
                # self._geo_step[k] = self._step  # later use to discard old counts
            # elif diff_count[k] < 0:
                # logger.info(f"{diff_count[k]}={len(self._geo_dict[k])} occurrences of {k}")
                # heapq.heappush(self._pqueue, (-len(self._geo_dict[k]), k, self._step))
                # self._geo_step[k] = self._step
            if k in self._key_to_priority:
                priority = self._key_to_priority[k]
                self._key_to_priority.pop(k)
                if self.compute_sec_structs:
                    sec_memb, exists, count, kk = priority 
                else:
                    exists, count, kk = priority
                count = -count
                exists = not exists
                self._priority_dict.pop(priority)   
                assert k == kk
            else:
                if self.compute_sec_structs:
                    priority = (True, 0, 0, k)
                    sec_memb, count = 0, 0
                else:
                    priority = (True, 0, k)
                    count = 0
            length = Tokenizer.num_bonds(json.loads(k))
            for (i, i1, length, action) in diff_count[k]:
                t = self.tokenizers[i]
                if action == "add":
                    count += 1
                    if self.compute_sec_structs:
                        sec_memb += t.is_secondary(i1, length)
                else:
                    count -= 1
                    if self.compute_sec_structs:
                        sec_memb -= t.is_secondary(i1, length)
            assert count == len(self._geo_dict[k])            
            # update priority (or not)
            if count:     
                exists = False
                if hasattr(self, "_sphere_dict") and k in self._sphere_dict:
                    exists = True
                if self.compute_sec_structs:
                    new_priority = (not exists, -sec_memb, -count, k)
                else:
                    new_priority = (not exists, -count, k)
                # if k not in self._key_to_priority and length >= self.rmsd_partition_min_size: # new key, common, need rmsd
                #     if len(self._geo_dict[k]) > 10:
                #         new_keys.append(k)
                self._key_to_priority[k] = new_priority
                self._priority_dict[new_priority] = None
                # logger.info(f"{priority}->{new_priority}")
            else:
                self._geo_dict.pop(k)
                # logger.info(f"remove {priority}")
            
        step_times["step7"] += time.perf_counter() - start_time
    
        # --- Step 8: RMSD packing for new keys above self._rmsd_partition_min_size
        start_time = time.perf_counter()
        # for k in new_keys:
            # self.rmsd_partition(k) # will update _geo_dict and priorities
        step_times["step8"] += time.perf_counter() - start_time

        # Log total time spent in each step.
        for step, t_elapsed in step_times.items():
            logger.info(f"Total time for {step}: {t_elapsed:.6f} seconds")
        time_elapsed = time.time()-step_start_time
        try:
            time_elapsed -= vis_time
        except:
            pass
        self._times.append(time_elapsed)
        # TODO: make more efficient
        if self.plot_iou_with_sec_structs:
            self.compute_iou()
        
        if recurring_key:
            logger.info(f"Step {self._step} repeat took {time_elapsed}")
        else:
            logger.info(f"Step {self._step-1} took {time_elapsed}")

        if not self._priority_dict.peekitem(0)[0][0]:
            logger.info(f"Repeating step {self._step-int(not recurring_key)}...")
            self.step()


def debug():
    from foldingdiff.datasets import FullCathCanonicalCoordsDataset
    d = "1755847933.8217926"
    i = 100
    bpe = pickle.load(open(f"/n/netscratch/mzitnik_lab/Lab/msun415/{d}/bpe_iter={i}.pkl", "rb"))
    dataset = FullCathCanonicalCoordsDataset("pretrain", 
                                            use_cache=False,
                                            zero_center=False)    
    cleaned_structures = []
    for i, struc in enumerate(dataset.structures):
        if (struc['angles']['psi']==struc['angles']['psi']).sum() < len(struc['angles']['psi'])-1:
            print(f"skipping {i}, {struc['fname']} because of missing dihedrals")
        else:
            cleaned_structures.append(struc)

    for index in range(len(bpe.tokenizers)):
        t = Tokenizer(cleaned_structures[index])
        bpe.tokenize(t)


if __name__ == "__main__":
    breakpoint()
    debug()
