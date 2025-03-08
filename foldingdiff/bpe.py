from foldingdiff.tokenizer import *
from foldingdiff.plotting import *
import heapq
import json
import logging
logger = logging.getLogger(__name__)

class BPE():
    def __init__(self, structures, bins=100, save_dir="./plots/bpe"):
        """
        structures: list of dataset objects
        bins: resolution of discretization
        """
        self.tokenizers = []
        for structure in structures:
            tokenizer = Tokenizer(deepcopy(structure))
            self.tokenizers.append(tokenizer)
        self.bins = bins
        self.n = len(self.tokenizers)
        self.save_dir = save_dir
        self._step = 0
    
    def initialize(self):
        self._init_thresholds() # call this before _init_bond_tokens
        self._init_bond_tokens()
    
    def _init_bond_tokens(self):
        """
        Here we treat each bond as an inital "bond-type" token, then standardizing the length of the bond to a fixed value
        """
        self._tokens = {}
        for i in range(3):
            self._tokens[i] = {Tokenizer.BOND_TYPES[i]: [0]}
            self._thresholds[Tokenizer.BOND_TYPES[i]] = [(Tokenizer.BOND_LENGTHS[i], Tokenizer.BOND_LENGTHS[i])]
        for t in self.tokenizers:
            new_tokens = [(i,t.bond_labels[i],1) for i in range(3*t.n-1)]
            token_pos = [i for i in range(3*t.n-1)]
            # update avg bond lengths
            for i,tt,l in new_tokens:
                dic = self._tokens[tt]                
                t.set_token_geo(i, l, self._bin_val(dic))
            t.token_pos = token_pos
            t.tokens = new_tokens


    def _init_thresholds(self):
        """
        We obtain the thresholds in a data-driven way
        These threshold determine statistical significance of a motif, so we should choose them carefully
        Due to circular angular data, we use the rmse
        Thresholds: List[(mean, std)]
        """
        self._thresholds = {}
        self._bin_counts = {}
        # we will fix the bond lengths
        vals = {}
        for t in self.tokenizers:
            angles = t.angles_and_dists
            for key in t.BOND_ANGLES+t.DIHEDRAL_ANGLES: # these are mostly fixed
                t_vals = angles[key][angles[key].fillna(0)!=0.].tolist()
                vals[key] = vals.get(key, []) + t_vals
        for key in t.BOND_ANGLES+t.DIHEDRAL_ANGLES:
            bin_centers, widths, counts = save_circular_histogram(vals[key], path=None, bins=self.bins)
            # save_circular_histogram(vals[key], path=f"/n/home02/msun415/foldingdiff/hist_{key}.png", bins=self.bins, title=f"Histogram {key}")
            for center, width, count in zip(bin_centers, widths, counts):
                self._thresholds[key] = self._thresholds.get(key, []) + [(center-width/2, center+width/2)]
                self._bin_counts[key] = self._bin_counts.get(key, []) + [count]
    
    @staticmethod
    def hash_geo(geo):
        return json.dumps(geo, sort_keys=True)

    def compute_geo_key(self, token_pair, i):
        try:
            (idx1, _, l1), (idx2, _, l2) = token_pair
        except:
            return None
        assert idx1+l1 == idx2
        geo = self.tokenizers[i].token_geo(idx1, l1+l2)
        for k in geo:
            if k in Tokenizer.BOND_TYPES:
                geo[k] = [0 for _ in geo[k]] # fixed
            else:
                geo[k] = ((np.array(geo[k])+2*np.pi) % (2*np.pi)).tolist() # Convert to [0, 2*pi]
                inds = []
                for v in geo[k]:
                    for l, (start, end) in enumerate(self._thresholds[k]):
                        if v >= start and v<end:
                            inds.append(l)
                            break
                if len(inds) != len(geo[k]):
                    breakpoint()
                geo[k] = inds
        geo_key = self.hash_geo(geo)        
        return geo_key

    def bin(self):
        """
        Loop over all tokenizers' current token pairs from scratch
        Bin each of their geometries
        Insert (binned_geometry, count) into pqueue
        """
        self._geo_dict = {}
        self._geo_step = {}
        for i in range(self.n):
            for j in range(len(self.tokenizers[i].tokens)-1):
                token_pair = self.tokenizers[i].tokens[j:j+2]
                geo_key = self.compute_geo_key(token_pair, i)
                (i1, _, l1), (i2, _, l2) = token_pair
                self._geo_dict[geo_key] = self._geo_dict.get(geo_key, []) + [(i, i2)] # store the start of the second token
        for key in self._geo_dict:
            self._geo_step[key] = 0
        self._pqueue = [(-len(self._geo_dict[key]), key, 0) for key in self._geo_dict]
        heapq.heapify(self._pqueue)
    

    def _bin_val(self, key_dict):
        key_dict_copy = {}
        for k, vals in key_dict.items():
            assert np.all([isinstance(v, int) for v in vals])
            key_dict_copy[k] = [sum(self._thresholds[k][v])/2 for v in vals]
        return key_dict_copy
    

    def visualize(self, key, output_path):
        """
        Visualize a key, take the average of the bin
        """
        vals = self._geo_dict[key] # check ValueError
        key_dict = json.loads(key)
        key_dict = self._bin_val(key_dict)
        # since geo_nerf expects 3*n-1 bonds, we do a similar procedure as in Tokenizer.compute_coords, rounding and offsetting
        i, index = vals[0]
        length = sum([len(key_dict[bt]) for bt in Tokenizer.BOND_TYPES if bt in key_dict])
        start = 3*(index//3)
        end = 3*(((index+length-1)+1)//3)+1 # end bond id, but we round it up so it's 1 (mod 3)
        off_start = index-start
        off_end = end-(index+length-1)
        geo = self.tokenizers[i].token_geo(start, end-start+1) # round from nearest residues        
        for k in geo:
            if k in key_dict:
                n = len(key_dict[k])
                if k in Tokenizer.BOND_TYPES:                
                    bt_index = Tokenizer.BOND_TYPES.index(k)                        
                    geo[k] = [Tokenizer.BOND_LENGTHS[bt_index] for _ in geo[k]] # default values
                    if index%3 > bt_index:
                        assert np.abs(len(geo[k])-n) <= 2.
                        for j in range(n):
                            geo[k][1+j] = key_dict[k][j]
                    else:
                        assert np.abs(len(geo[k])-n) <= 1.
                        for j in range(n):
                            geo[k][j] = key_dict[k][j]
                else:
                    assert n == len(geo[k])
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
        count, key, step = heapq.heappop(self._pqueue)
        key_dict = json.loads(key)
        for i, index in self._geo_dict[key]: # _geo_dict is sorted
            t = self.tokenizers[i]
            geo_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), i)
            if geo_key != key:
                continue
            # update tokens and token_pos
            breakpoint()
            # standardize
            t.set_token_geo(i1, l1+l2, self._bin_val(key_dict))        
        self.bin()


    def step(self):
        logger.info(f"Step {self._step}")
        # pick the most frequent token pair
        while True:
            count, key, step = heapq.heappop(self._pqueue)
            logger.info(f"Pop {key} as most frequent key, {-count} times")
            if self._geo_step[key] == step:
                break
        # visualizations
        tokens_by_freq = sorted(self._geo_dict.items(), key=lambda t: len(t[1]))
        counts = []
        for k, vals in tokens_by_freq:
            count = len(vals)
            logger.info("%s: %d occurrences", k, count)
            counts.append(count)
        plot_path = os.path.join(self.save_dir, f"counts_iter={self._step}.png")
        sorted_bar_plot(counts, title=f"Counts by Binned Geometry, iter={self._step}", 
                        ylabel="Count", save_path=plot_path)
        key, _ = tokens_by_freq[-1]
        key_vis_path = os.path.join(self.save_dir, f"key_iter={self._step}.png")
        self.visualize(key, key_vis_path)    
        # define new token
        key_dict = json.loads(key)
        length = Tokenizer.num_bonds(key_dict)
        n = len(self._tokens)
        self._tokens[n] = key_dict        
        # standardize and replace all occurrences
        # update 3 data structures: t.token_pos, _geo_dict, and pqueue
        # we will track which keys to remove/add, and later infer change to pqueue
        add_geo_dict = {} # add keys for _geo_dict
        rem_geo_dict = {} # remove keys for _geo_dict
        inds = [0 for _ in range(self.n)]
        # we will running update token_pos
        for i, index in self._geo_dict[key]: # _geo_dict is sorted
            t = self.tokenizers[i]
            i2 = index # start of second token
            i1 = t.token_pos[index-1] # start of prev token
            l1 = i2-i1
            l2 = length-l1
            # do we need to update, double check it's still there
            geo_key = self.compute_geo_key(((i1, None, l1), (i2, None, l2)), i)
            if geo_key != key:
                continue
            # 1. Decrement key.
            rem_geo_dict[key] = rem_geo_dict.get(key, []) + [(i, index)]
            # 2. Update t.token_pos and t.tokens
            for j in range(i2, i2+l2):
                t.token_pos[j] = i1
            while t.tokens[inds[i]] is None or t.tokens[inds[i]][0] < i1: # find where it is
                inds[i] += 1
            assert t.tokens[inds[i]][0] == i1
            t.tokens[inds[i]] = (i1, n, length)
            t.tokens[inds[i]+1] = None
            # 3 and 4. Decrement left and right key pair freq.
            if i1:
                i0 = t.token_pos[i1-1]
                l0 = i1-i0
                left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1)), i)
            else:
                left_key = None
            if i2+l2 < len(t.token_pos):
                i3 = i2+l2
                l3 = 0
                while i3+l3 < len(t.token_pos) and t.token_pos[i3+l3] == i3:
                    l3 += 1
                right_key = self.compute_geo_key(((i2, None, l2), (i3, None, l3)), i)
            else:
                right_key = None            
            if left_key:
                rem_geo_dict[left_key] = rem_geo_dict.get(left_key, []) + [(i, i1)]
            if right_key:
                rem_geo_dict[right_key] = rem_geo_dict.get(right_key, []) + [(i, i3)]
            # 5. Increment new pairs.
            if left_key:
                new_left_key = self.compute_geo_key(((i0, None, l0), (i1, None, l1+l2)), i)
                add_geo_dict[new_left_key] = add_geo_dict.get(new_left_key, []) + [(i, i1)]
            if right_key:
                new_right_key = self.compute_geo_key(((i1, None, l1+l2), (i3, None, l3)), i)
                add_geo_dict[new_right_key] = add_geo_dict.get(new_right_key, []) + [(i, i3)]                
            # 6. Standardize occurrence by binning key_dict
            t.set_token_geo(i1, l1+l2, self._bin_val(key_dict))
        # 2 (cont). update t.tokens
        for i in range(self.n):
            t = self.tokenizers[i]
            t.tokens = list(filter(None, t.tokens))
        # 7. update _geo_dict by iterating
        diff_count = {k:0 for k in rem_geo_dict}
        diff_count.update({k:0 for k in add_geo_dict})
        for k in add_geo_dict:
            logger.info(f"Add {len(add_geo_dict[k])}+{len(self._geo_dict.get(k, []))} occurrences of {k}")
            self._geo_dict[k] = BPE.add_values(self._geo_dict.get(k, []), add_geo_dict[k])
            diff_count[k] += len(add_geo_dict[k])
        for k in rem_geo_dict:
            logger.info(f"Remove {len(self._geo_dict[k])}-{len(rem_geo_dict[k])} occurrences of {k}")
            self._geo_dict[k] = BPE.remove_values(self._geo_dict[k], rem_geo_dict[k])
            diff_count[k] -= len(rem_geo_dict[k])
        self._step += 1
        # 8. update priority queue with step
        for k in diff_count:
            if diff_count[k] == 0: # no change
                continue
            heapq.heappush(self._pqueue, (-len(self._geo_dict[k]), k, self._step))            
            self._geo_step[k] = self._step # later use to discard old counts
