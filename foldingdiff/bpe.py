from foldingdiff.tokenizer import *
from foldingdiff.plotting import *
from foldingdiff.algo import *
import heapq
import json
import logging
from collections import defaultdict
from sortedcontainers import SortedDict
logger = logging.getLogger(__name__)

class BPE():
    def __init__(self, structures, bins, 
        bin_strategy="histogram",
        save_dir="./plots/bpe",
        compute_sec_structs=False, 
        plot_iou_with_sec_structs=False,
        res_init=False,
        rmsd_partition_min_size=4,
        num_partitions=3,
        glue_opt=False
    ):
        """
        structures: list of dataset objects
        bins: resolution of discretization, dict from size to num_bins
        bin_strategy: how to bin values
        compute_sec_structs: whether to use secondary structure count to define token pair priority
        rmsd_packing_min_size: when to start using rmsd partitioning, if 0 run with special setting
        num_partitions: how many partitions
        """
        self.tokenizers = []
        for structure in structures:
            tokenizer = Tokenizer(deepcopy(structure), compute_sec_structs=compute_sec_structs | plot_iou_with_sec_structs)
            self.tokenizers.append(tokenizer)
        self.compute_sec_structs = compute_sec_structs
        self.plot_iou_with_sec_structs = plot_iou_with_sec_structs
        self.rmsd_partition_min_size = rmsd_partition_min_size
        self.rmsd_only = rmsd_partition_min_size == 0
        self.glue_opt = glue_opt
        self.num_partitions = num_partitions
        self.res_init = res_init
        self.bins = bins
        self.bin_strategy = bin_strategy
        print(self.bin_strategy)
        self.n = len(self.tokenizers)
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
        self._init_tokens()        
        logger.info(f"_init_tokens took {time.perf_counter()-start_time}")
        logger.info(f"Initialize finish")
    
    def _init_tokens(self):
        """
        Here we treat each bond as an inital "bond-type" token, then standardizing the length of the bond to a fixed value
        """
        self._tokens = {}
        for i in range(3):
            self._tokens[i] = {Tokenizer.BOND_TYPES[i]: [0]}
            self._thresholds[Tokenizer.BOND_TYPES[i]] = [(Tokenizer.BOND_LENGTHS[i], Tokenizer.BOND_LENGTHS[i])]  
        label_dict = {}
        if self.rmsd_only:
            breakpoint()
        for t in self.tokenizers:
            # update avg bond lengths
            for i in range(3*t.n-1):
                tt = t.bond_labels[i]
                dic = self._tokens[tt]                
                try:
                    t.set_token_geo(i, 1, self._bin_val(dic))
                except:
                    breakpoint()
            if self.res_init:      
                if self.rmsd_only:
                    # get the quantized residue for each
                    breakpoint()
                else:     
                    # update binned residue geo                
                    labels = []
                    for i in range(t.n):
                        start = 3*i
                        length = 3 if i < t.n-1 else 2
                        geo = t.token_geo(start, length)
                        for k in geo:
                            if k not in Tokenizer.BOND_ANGLES+Tokenizer.DIHEDRAL_ANGLES:
                                continue
                            quant_vals = []
                            for i, v in enumerate(geo[k]):
                                lookup = self._thresholds[length]
                                v = (v+2*np.pi) % (2*np.pi) # Convert to [0, 2*pi]
                                ind = BPE.get_ind(v, lookup[k])
                                quant_vals.append(ind)      
                            geo[k] = quant_vals

                        key = self._bin_val(geo)
                        key_str = BPE.hash_geo(key)
                        if key_str not in label_dict:
                            n = len(label_dict)
                            label_dict[key_str] = n
                        else:
                            n = label_dict[key_str]
                        t.set_token_geo(start, length, key)
                        labels.append(n)
                # 3*(t.n-1) + 2 bonds
                new_tokens = [(3*i, labels[i], 3) for i in range(t.n-1)] + [(3*t.n-3, labels[t.n-1], 2)]
                bond_to_token = {t[0]: t for t in new_tokens} # start bond : token
                token_pos = [3*(i//3) for i in range(3*t.n-1)]                
                t.token_pos = token_pos
                t.tokens = new_tokens
                t.bond_to_token = bond_to_token
            else:
                new_tokens = [(i,t.bond_labels[i],1) for i in range(3*t.n-1)]
                bond_to_token = {t[0]: t for t in new_tokens}
                token_pos = [i for i in range(3*t.n-1)]
                t.token_pos = token_pos
                t.tokens = new_tokens
                t.bond_to_token = bond_to_token

        if self.res_init:
            self._tokens = {n: json.loads(key_str) for key_str, n in label_dict.items()}
            logger.info(f"initialized {len(self._tokens)} residue-level tokens")

    
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
        These threshold determine statistical significance of a motif, so we should choose them carefully
        We make these depend on |token|
        Due to circular angular data, we use circular histogram
        Thresholds: Dict{|token|: List[(bin_start, bin_end)]}
        """
        self._thresholds = ThresholdDict()
        self._bin_counts = {}
        last_size = 0
        for size, num_bins in self.bins.items():
            if last_size:
                for s in range(last_size+1, size):
                    self._thresholds[s] = self._thresholds[last_size]
                    self._bin_counts[s] = self._bin_counts[last_size]
            _thresholds = {}
            _bin_counts = {}
            # we will fix the bond lengths
            vals = {}
            for t in self.tokenizers:
                angles = t.angles_and_dists
                for key in t.BOND_ANGLES+t.DIHEDRAL_ANGLES: # these are mostly fixed
                    t_vals = angles[key][angles[key].fillna(0)!=0.].tolist()
                    vals[key] = vals.get(key, []) + t_vals
            if path is not None:
                path = Path(path).with_name(name)
            for key in t.BOND_ANGLES+t.DIHEDRAL_ANGLES:
                name = f"{key}_{self.bin_strategy}_{num_bins}.png"
                if self.bin_strategy == "histogram":
                    starts, ends, widths, counts = save_circular_histogram(vals[key], path=path, bins=self.bins[size], title=f"{self.bin_strategy} {key}, {num_bins} bins")
                elif self.bin_strategy == "uniform":
                    starts, ends, widths, counts = save_circular_histogram_equal_counts(vals[key], path=path, bins=self.bins[size], title=f"{self.bin_strategy} {key}, {num_bins} bins")
                else:
                    raise NotImplementedError
                logger.info(f"# bins: {len(counts)}, bin starts: {starts}, bin ends: {ends}, counts: {counts}")
                
                for start, end, width, count in zip(starts, ends, widths, counts):
                    _thresholds[key] = _thresholds.get(key, []) + [(float(start), float(end))]
                    _bin_counts[key] = _bin_counts.get(key, []) + [count]                 
            self._thresholds[size] = _thresholds
            self._bin_counts[size] = _bin_counts
            last_size = size
    

    @property
    def vocab_size(self):
        vocab = len(self._tokens)
        vocab += self.cum_bin_count()
        return vocab
        
    
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

    def compute_geo_key(self, token_pair, i, ignore_left=False, ignore_right=False):
        """
        Compute the quantized geo key for token_pair
        token_pair: (idx1, _, l1) and (idx2, _, l2)
        can be old tokens  
        ignore_left: whether to ignore the fact the left token is in a partitioned token   
        ignore_left: same for right   
        """
        t = self.tokenizers[i]
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
                    if pt1:
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
                    if k in Tokenizer.BOND_TYPES:
                        lookup = self._thresholds
                    else:
                        v = (v+2*np.pi) % (2*np.pi) # Convert to [0, 2*pi]
                        lookup = self._thresholds[l1+l2]
                    ind = BPE.get_ind(v, lookup[k])
                    quant_vals.append(ind)
                else:
                    quant_vals.append(v)
            geo[k] = quant_vals
        geo_key = self.hash_geo(geo)
        return geo_key

    def bin(self):
        """
        Loop over all tokenizers' current token pairs from scratch
        Bin each of their geometries
        Insert (binned_geometry, count) into pqueue
        """
        self._geo_dict = defaultdict(set)
        self._sphere_dict = {} # for rmsd sphere packin
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
                # priority: (#(secondary membership), #(membership), key)
                sec_memb = 0
                for i, i2 in self._geo_dict[key]:
                    t = self.tokenizers[i]
                    i1 = t.token_pos[i2-1]
                    sec_memb += t.is_secondary(i1, length)
                priority = (-sec_memb, -len(self._geo_dict[key]), key)
            else:
                priority = (-len(self._geo_dict[key]), key)
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
            if k in Tokenizer.BOND_TYPES:
                relv_thresholds = self._thresholds
            else:
                relv_thresholds = self._thresholds[size]
            key_dict_copy[k] = [sum(relv_thresholds[k][v])/2 if isinstance(v, int) else v for v in vals]
        return key_dict_copy
    

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
        plot_backbone(coords, output_path, bts, title=key, zoom_factor=1.0)  

    
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
        key_dict = json.loads(k)
        length = Tokenizer.num_bonds(key_dict)
        all_occurs = []
        all_coords = []
        for i, index in self._geo_dict[k]:
            t = self.tokenizers[i]
            i1 = t.token_pos[index-1]
            all_occurs.append((i, i1))
            coords = t.compute_coords(i1, length)
            all_coords.append(coords)
        medoids, assignments = k_medoids(all_coords, self.num_partitions)
        # _sphere_dict
        self._sphere_dict[k] = []
        for p, m in enumerate(medoids):
            i, i1 = all_occurs[m]         
            t = self.tokenizers[i]
            struc = t.token_geo(i1, length)            
            logger.info(f"Partition {p}: {struc}")
            self._sphere_dict[k].append(struc)
            key_vis_path = os.path.join(self.save_dir, f"key_iter={self._step}_{p}.png")
            t.visualize_bonds(i1, length, key_vis_path)
        return assignments


    def step(self):
        logger.info(f"Step {self._step} start")
        step_start_time = time.time()
        # pick the most frequent token pair
        # while True:
        #     count, key, step = heapq.heappop(self._pqueue)
        #     logger.info(f"Pop {key} as most frequent key, {-count} times")
        #     if self._geo_step[key] == step:
        #         break
        # loop to find most frequent token pair
        # key = None
        # count = float("-inf")
        # for k, vals in self._geo_dict.items():
        #     if len(vals) > count:
        #         key = k
        #         count = len(vals)
        key_tup = self._priority_dict.peekitem(0)
        if self.compute_sec_structs:
            (num_sec, count, key), _ = key_tup
            count = -count       
            assert count == len(self._geo_dict[key])
            # self._bin_side_chain(key) # later add to _tokens
            logger.info(f"Pop {key} as most frequent key, {count} times, {num_sec} secondary structures")        
        else:
            (count, key), _ = key_tup
            if key in self._tokens.values():
                breakpoint()
            count = -count
            assert count == len(self._geo_dict[key])
            logger.info(f"Pop {key} as most frequent key, {count} times")  
            # TODO: Should never pop again
        # visualizations
        vis_start_time = time.time()
        # tokens_by_freq = sorted(self._geo_dict.items(), key=lambda t: len(t[1]))
        # counts = []
        # for k, vals in tokens_by_freq:
        #     count = len(vals)
        #     logger.info("%s: %d occurrences", k, count)
        #     counts.append(count)
        # plot_path = os.path.join(self.save_dir, f"counts_iter={self._step}.png")
        # sorted_bar_plot(counts, title=f"Counts by Binned Geometry, iter={self._step}", 
        #                 ylabel="Count", save_path=plot_path)
        # key, _ = tokens_by_freq[-1]
        key_dict = json.loads(key)
        key_vis_path = os.path.join(self.save_dir, f"key_iter={self._step}.png")
        self.visualize(key, key_vis_path)
        vis_time = time.perf_counter()-vis_start_time
        # --- Step 0: Do RMSD packing if |token| >= rmsd_partition_min_size ---
        if Tokenizer.num_bonds(key_dict) >= self.rmsd_partition_min_size \
        and key not in self._sphere_dict \
        and len(self._geo_dict[key]) > self.num_partitions:
            # Step 0.1: populate _sphere_dict
            # Step 0.2: update geo
            # Step 0.3: update _geo_dict
            assignments = self.rmsd_partition(key)
            rmsd_key = key
        else:
            rmsd_key = None
        # Define new token        
        binned_key_dict = self._bin_val(key_dict) # get the values
        length = Tokenizer.num_bonds(key_dict)
        n = len(self._tokens)
        if rmsd_key is None:
            self._tokens[n] = key_dict        
        else: # create a collection of tokens
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
                if not self.rmsd_only: # don't set if rmsd_only                                        
                    t.set_token_geo(i1, length, self._sphere_dict[key][p])
                    if self.glue_opt:
                        t.opt_glue(i1, length)
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
                    sec_memb, count, kk = priority 
                else:
                    count, kk = priority
                count = -count
                self._priority_dict.pop(priority)   
                assert k == kk
            else:
                if self.compute_sec_structs:
                    priority = (0, 0, k)
                    sec_memb, count = 0, 0
                else:
                    priority = (0, k)
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
                if self.compute_sec_structs:
                    new_priority = (-sec_memb, -count, k)
                else:
                    new_priority = (-count, k)
                # if k not in self._key_to_priority and length >= self.rmsd_partition_min_size: # new key, common, need rmsd
                #     if len(self._geo_dict[k]) > 10:
                #         new_keys.append(k)
                self._key_to_priority[k] = new_priority
                self._priority_dict[new_priority] = None
                logger.info(f"{priority}->{new_priority}")
            else:
                self._geo_dict.pop(k)
                logger.info(f"remove {priority}")
            
        step_times["step7"] += time.perf_counter() - start_time
    
        # --- Step 8: RMSD packing for new keys above self._rmsd_partition_min_size
        start_time = time.perf_counter()
        # for k in new_keys:
            # self.rmsd_partition(k) # will update _geo_dict and priorities
        step_times["step8"] += time.perf_counter() - start_time

        # Log total time spent in each step.
        for step, t_elapsed in step_times.items():
            logger.info(f"Total time for {step}: {t_elapsed:.6f} seconds")
        time_elapsed = time.perf_counter()-step_start_time
        time_elapsed -= vis_time
        self._times.append(time_elapsed)
        # TODO: make more efficient
        if self.plot_iou_with_sec_structs:
            self.compute_iou()
        logger.info(f"Step {self._step-1} took {time_elapsed}")                     
        # logger.info(f"Step {self._step-1} took {time_elapsed}")                     

