import numpy as np
from foldingdiff.angles_and_coords import *
from foldingdiff.data_structures import *
from foldingdiff.algo import compute_rmsd, kabsch
from foldingdiff.nerf import *
from foldingdiff.plotting import plot_backbone
from types import SimpleNamespace
from itertools import starmap
from copy import deepcopy
import tempfile
import imageio
import time
import logging
import pickle
from pathlib import Path
logger = logging.getLogger(__name__)

class Tokenizer:
    BOND_TYPES = ["N:CA","CA:C","0C:1N"]
    ATOM_TYPES = ["N","CA","C"]
    BOND_ANGLES = ["tau","CA:C:1N","C:1N:1CA"]
    DIHEDRAL_ANGLES = ["psi","omega","phi"]
    BOND_LENGTHS = [N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH]
    def __init__(self, structure, compute_sec_structs=False):
        for col in structure['angles'].columns:
            structure['angles'][col] = pd.Series(
                [float(x) for x in structure['angles'][col]],
                index=structure['angles'].index,
                dtype=object
            )
        self._angles_and_dists = structure['angles'] # controls coords
        self._coords = structure['coords']
        self.beta_coords = structure['c_beta']
        idxes = structure['full_idxes']
        self._idxes = idxes
        self._res_idx_map = dict(zip(idxes[0::3], range(0, len(idxes), 3)))
        self._full_coords = structure['full_coords']
        self.compute_sec_structs = compute_sec_structs
        if self.compute_sec_structs:
            self._sec = structure['sec']   
        else:
            self._sec = None
        self._side_chains = structure['side_chain']
        self.aa = structure['aa']
        self.fname = structure['fname']
        self.n = len(self._angles_and_dists) # fixed
        # fixed
        self.bond_labels = sum([[0,1,2] for _ in range(self.n-1)]+[[0,1]], []) # 0,1,2,...,0,1
        self.atom_labels = np.tile([0,1,2], self.n)
        self.edges = [[j,j+1,0] for j in range(1,3*self.n)]
        self._bond_to_token = None
        # init
        self._init_coords()
        if self.compute_sec_structs:
            try:
                self._init_secondary_pos()
            except Exception as e:
                logger.error("%s: _init_secondary_pos", self.fname)
                logger.error(str(e))
                raise  # re-raise the caught exception to terminate the program
    

    @property
    def bond_to_token(self):
        return self._bond_to_token
    

    @bond_to_token.setter
    def bond_to_token(self, new_value):
        self._bond_to_token = TokenHierarchy(self, new_value) # new instance
    

    def _init_coords(self):
        self._init_n_ca = np.linalg.norm(N_INIT-CA_INIT)
        self._init_ca_c = np.linalg.norm(CA_INIT-C_INIT)
        self._init_bond_angle = angle_between(N_INIT-CA_INIT, C_INIT-CA_INIT)


    def _init_secondary_pos(self):
        """
        Index each unique secondary structure
        Map each bond position to the index (if part of secondary structure)
        or -1 if not part of any
        """
        self.sec_pos = [-1 for _ in self.bond_labels]
        self.sec_types = []
        self.sec_bond_range = {}
        _rim = self._res_idx_map
        for sec_type, i, j in self._sec:
            # map residue k to an atom
            # alias _rim as self._res_idx_map
            # residues i,i+1,...,j covers bonds
            # _rim[ind], _rim[ind]+1, _rim[ind]+2
            # for ind=i,...j-1
            # _rim[ind], _rim[ind]+1 for ind=j           
            sec_id = len(self.sec_types)
            self.sec_types.append(sec_type)
            start = None
            end = None
            for ind in range(i, j):
                if ind in _rim:
                    if start is None:
                       start =  _rim[ind]
                    self.sec_pos[_rim[ind]] = sec_id
                    self.sec_pos[_rim[ind]+1] = sec_id
                    self.sec_pos[_rim[ind]+2] = sec_id
                    end = _rim[ind]+2
            if j in _rim:
                if start is None:
                    start = _rim[j]
                self.sec_pos[_rim[j]] = sec_id
                self.sec_pos[_rim[j]+1] = sec_id
                end = _rim[j]+1
            self.sec_bond_range[sec_id] = (start, end)


    def is_secondary(self, i1, length):
        assert self.compute_sec_structs
        return self.sec_pos[i1] != -1 and self.sec_pos[i1] == self.sec_pos[i1+length-1]
    
    
    def _add_tokens(self, tokens):        
        for idx, t, l in self._tokens: # TODO: use just the pointers
            if t in tokens: # new token
                self.reps[t] = self.reps.get(t, []) + [self.token_geo(idx,l)]
        for t in self.reps:
            self.reps[t] = np.mean(self.reps[t])


    def _bond_length(self, idx):
        if idx == 0:
            return self._init_n_ca            
        elif idx == 1:
            return self._init_ca_c            
        else:
            return self._angles_and_dists[Tokenizer.BOND_TYPES[idx%3]][(idx-2)//3]
        
    def _set_bond_length(self, idx, value):
        if idx == 0:
            self._init_n_ca = value
        elif idx == 1:
            self._init_ca_c = value
        else:
            # max is idx=3*n-2, which is .iloc[n-2], idx=3*n-1 would be n-1
            self._angles_and_dists[Tokenizer.BOND_TYPES[idx%3]].iloc[(idx-2)//3] = value
    
    def _bond_angle(self, idx):
        if idx == 0:
            return self._init_bond_angle
        else:
            # max is idx=3*n-3, which is 3*n-4 or n-2
            return self._angles_and_dists[Tokenizer.BOND_ANGLES[idx%3]][(idx-1)//3]
    
    def _set_bond_angle(self, idx, value):
        if idx == 0:
            self._init_bond_angle = value
        else:
            self._angles_and_dists[Tokenizer.BOND_ANGLES[idx%3]].iloc[(idx-1)//3] = value
        
    def _dihedral_angle(self, idx):
        # max is idx=3*n-4, which is (3*n-3)//3=n-1
        return self._angles_and_dists[Tokenizer.DIHEDRAL_ANGLES[idx%3]][(idx+1)//3]

    def _set_dihedral_angle(self, idx, value):
        # max is idx=3*n-4, which is (3*n-3)//3=n-1
        self._angles_and_dists[Tokenizer.DIHEDRAL_ANGLES[idx%3]].iloc[(idx+1)//3] = value

    def token_geo(self, idx, l):
        """
        Here we want the geometry of bonds idx:idx+l
        To standardize, we always use the dists and angles representation
        The output consists of:
            1. l bond dists, in order of bond idx,idx+1,...,idx+l-1
            2. l-1 bond angles
            3. l-2 dihedral angles  
        For example, if l=1, and idx points to a N-CA bond, then output is {"N:CA": [length of bond idx]}
        Edge cases:
            Always start with N_INIT,CA_INIT,C_INIT so:
            a) no dihedrals for first phi,psi,omega angles
            b) no tau, CA:C:1N, C:1N:CA for first angles
            c) no bond lengths for first N:CA, CA:C bonds
        """
        if idx+l-1 > 3*self.n-1: 
            raise ValueError(f"idx+l cannot exceed {3*self.n-1}")
        ans = {}
        # Bond dists
        for j in range(idx, idx+l):
            bt = Tokenizer.BOND_TYPES[j%3]
            ans[bt] = ans.get(bt, []) + [self._bond_length(j)]
        # Bond angles
        for j in range(idx, idx+l-1):
            # bond j to j+1
            ang = Tokenizer.BOND_ANGLES[j%3]
            ans[ang] = ans.get(ang, []) + [self._bond_angle(j)]
        # Dihedral angles
        for j in range(idx, idx+l-2):
            # dihedral around j+1, aka between plane formed by bonds j,j+1 and j+1,j+2
            di = Tokenizer.DIHEDRAL_ANGLES[j%3]
            ans[di] = ans.get(di, []) + [self._dihedral_angle(j)]
        return ans
    
    def visualize_bonds(self, i1, length, output_path, **kwargs):
        coords = self.compute_coords(i1, length)
        # ATOM_TYPES[i1%3], ATOM_TYPES[i1%3+1], ..., ATOM_TYPES[i1%3+length]
        bts = [Tokenizer.ATOM_TYPES[(i1%3+i)%3] for i in range(length+1)]
        plot_backbone(coords, output_path, bts, title=f"{Path(self.fname).stem} bonds {i1}-{i1+length-1}", zoom_factor=1.0, **kwargs)
    
    def visualize(self, output_path, **kwargs):
        coords = self.compute_coords() # (3*N)
        if "ref_coords" in kwargs:
            ref_coords = kwargs.pop("ref_coords") # (3*N, 3)
            # before, align with kabsch
            coords, _, _ = kabsch(ref_coords, coords)
        if self.bond_to_token:
            tokens = [self.bond_to_token[i] for i in sorted(self.bond_to_token)]
        else:
            tokens = None
        return plot_backbone(coords, output_path, atom_types=np.tile(Tokenizer.ATOM_TYPES, len(coords)//3), tokens=tokens, **kwargs)
        

    def set_token_geo(self, idx, l, vals):
        """
        Here we want the geometry of bonds idx:idx+l
        To standardize, we always use the dists and angles representation
        The output consists of:
            1. l bond dists, in order of bond idx,idx+1,...,idx+l-1
            2. l-1 bond angles
            3. l-2 dihedral angles  
        For example, if l=1, and idx points to a N-CA bond, then output is {"N:CA": [length of bond idx]}
        Edge cases:
            Always start with N_INIT,CA_INIT,C_INIT so:
            a) no dihedrals for first phi,psi,omega angles
            b) no tau, CA:C:1N, C:1N:CA for first angles
            c) no bond lengths for first N:CA, CA:C bonds
        """
        rev_vals = deepcopy(vals)
        for k in rev_vals:
            rev_vals[k] = rev_vals[k][::-1]
        # Bond dists
        for j in range(idx, idx+l):
            bt = Tokenizer.BOND_TYPES[j%3]
            self._set_bond_length(j, rev_vals[bt].pop(-1))            
        # Bond angles
        for j in range(idx, idx+l-1):
            # bond j to j+1
            ang = Tokenizer.BOND_ANGLES[j%3]
            self._set_bond_angle(j, rev_vals[ang].pop(-1))
        # Dihedral angles
        for j in range(idx, idx+l-2):
            # dihedral around j+1, aka between plane formed by bonds j,j+1 and j+1,j+2
            di = Tokenizer.DIHEDRAL_ANGLES[j%3]
            self._set_dihedral_angle(j, rev_vals[di].pop(-1))
        for k in rev_vals:
            assert len(rev_vals[k]) == 0      


    def _standardize_res(self):
        breakpoint()
        # self.res = {0:[],1:[],2:[]}
        # for i,t,l in new_tokens:
        #     self.res[t].append(self.token_geo(i, l))        
        # for k in self.res:
        #     self.res[k] = {
        #         key: list(
        #             starmap(lambda *vals: sum(vals) / len(vals), zip(*(d[key] for d in self.res[k])))
        #         )
        #         for key in self.res[k][0]
        #     }        

    @staticmethod
    def num_bonds(geo):
        return (len(geo.get('N:CA', []))+len(geo.get('CA:C', []))+len(geo.get('0C:1N', [])))


    @property
    def angles_and_dists(self):
        return self._angles_and_dists
    # plot_backbone([N_INIT,CA_INIT,C_INIT],'/n/home02/msun415/foldingdiff/test_before.png')
    # plot_backbone(list(update_backbone_positions(N_INIT, CA_INIT, C_INIT, geo['CA:C'][0], geo['N:CA'][0], 0.0)),'/n/home02/msun415/foldingdiff/test_after_after.png')
    @staticmethod
    def geo_nerf(geo):
        """
        Given 3n-1 bonds, we use NERFBuilder by fixing the N_INIT-CA_INIT-C_INIT plane
        We infer the corrected N_INIT, CA_INIT by fixing C_INIT using the first bond angle and two bond dists
        Then call NERFBuilder with the remaining args
        """
        assert len(geo['N:CA']) == len(geo['CA:C'])
        assert len(geo['CA:C']) == len(geo.get('0C:1N', []))+1
        num_bonds = Tokenizer.num_bonds(geo)
        assert num_bonds%3 == 2
        n_init, ca_init, c_init = update_backbone_positions(N_INIT, CA_INIT, C_INIT, geo['CA:C'][0], geo['N:CA'][0], geo['tau'][0])
        if num_bonds == 2:
            nerf = SimpleNamespace()
            setattr(nerf, "cartesian_coords", np.array([n_init, ca_init, c_init]))
        else:        
            nerf = NERFBuilder(
                phi_dihedrals=np.array([np.nan]+geo['phi']),
                psi_dihedrals=np.array(geo['psi']+[np.nan]),
                omega_dihedrals=np.array(geo['omega']+[np.nan]),
                bond_len_n_ca=np.array(geo['N:CA'][1:]), 
                bond_len_ca_c=np.array(geo['CA:C'][1:]), 
                bond_len_c_n=np.array(geo['0C:1N']),
                bond_angle_n_ca=np.array(geo['C:1N:1CA']), 
                bond_angle_ca_c=np.array(geo['tau'][1:]), 
                bond_angle_c_n=np.array(geo['CA:C:1N']),
                init_coords=[n_init,ca_init,c_init]
            )
        return nerf
    

    def compute_coords(self, index=0, length=float("inf")):
        """
        Compute coords for length atoms from position index
        We call token_geo to get the angular information
        Round to the nearest residues
        Then return the coords
        """
        length = min(length, 3*self.n-1-index) 
        start = 3*(index//3)
        end = 3*(((index+length-1)+1)//3)+1 # end bond id, but we round it up so it's 1 (mod 3)
        off_start = index-start
        off_end = end-(index+length-1)
        geo = self.token_geo(start, end-start+1) # round from nearest residues
        geo_nerf = Tokenizer.geo_nerf(geo)
        # assert np.all(nerf.cartesian_coords == geo_nerf)
        coords = geo_nerf.cartesian_coords
        return coords[off_start: len(coords)-off_end] # offset
    
    # @angles_and_dists.setter
    # def angles_and_dists(self, i, attr, value): # use this from other classes
    #     breakpoint()
    #     self._update_coords()

    @property
    def g(self):
        return {
            'nodelabels': np.array(labels, dtype=np.uint32)[:, None],
            'nodepos': np.array(coords, dtype=np.float64),
            'edges': np.array(edges, dtype=np.uint32)
        }        


    def tokenize(self):
        tokenized = []
        for (start, bt, length) in self.bond_to_token.values():
            tokenized.append(("MOTIF", bt))
            # find the dihedral
            b = start+length # find dihedral around this bond        
            if b < 3*self.n-1:
                dt = Tokenizer.DIHEDRAL_ANGLES[(b-2)%3]
                tokenized.append(("DIHEDRAL", dt, self._dihedral_angle(b-2)))
                dt = Tokenizer.DIHEDRAL_ANGLES[(b-1)%3]
                tokenized.append(("DIHEDRAL", dt, self._dihedral_angle(b-1)))
                bt = Tokenizer.BOND_ANGLES[(b-1)%3]
                tokenized.append(("BOND_ANGLE", bt, self._bond_angle(b-1)))
        return tokenized

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

    
    def get_glue_left(self, idx):
        """
        Given bond idx (mult of 3), the N-CA bond of residue s, get (omega_{s-1}, theta_CNCA_s, phi_s)
        """
        if idx % 3:
            raise ValueError("must be multiple of 3")
        if idx < 3:
            raise ValueError("no left glues for first residue")
        return (self._dihedral_angle(idx-2), self._bond_angle(idx-1), self._dihedral_angle(idx-1))


    def set_glue_left(self, idx, tup):
        """
        Given bond idx (mult of 3), the N-CA bond of residue s, get (omega_{s-1}, theta_CNCA_s, phi_s)
        """
        if idx % 3:
            raise ValueError("must be multiple of 3")
        if idx < 3:
            raise ValueError("no left glues for first residue")
        if len(tup) != 3:
            raise ValueError("tup needs to be size 3")
        self._set_dihedral_angle(idx-2, tup[0])
        self._set_bond_angle(idx-1, tup[1])
        self._set_dihedral_angle(idx-1, tup[2])
    

    def entry_frame(self, idx, length):
        """
        Begin building coords from residue before idx
        Return entry frame of first residue (that idx+length belongs to)
        """
        if idx % 3:
            raise ValueError(f"idx={idx} has to be start of residue")
        if length % 3 != 2:
            raise ValueError(f"idx+length-1 must end the last residue")        
        return frame_from_triad(*list(self.compute_coords(idx-3, 2)))


    def exit_frame(self, idx, length, ret_all=False):
        """
        Begin building coords from residue before idx
        Return exit frame of final residue (that idx+length belongs to)
        """
        if idx % 3:
            raise ValueError(f"idx={idx} has to be start of residue")
        if length % 3 != 2:
            raise ValueError(f"idx+length-1 must end the last residue")
        coords = self.compute_coords(idx-3, length+3)
        if ret_all:
            assert coords.shape[0] % 3 == 0
            R_occs, t_occs = [], []
            for i in range((length+1)//3):
                R_occ, t_occ = frame_from_triad(*list(coords[3*i:3*(i+1)]))                
                R_occs.append(R_occ)
                t_occs.append(t_occ)
            return R_occs, t_occs
        else:            
            return frame_from_triad(*list(coords[-3:]))



def vis_subspans(t1, t2, folder):
    df = t1.angles_and_dists.iloc[(t1.n-9):]
    n = len(df)
    # remove rows 0 through 173 in this df
    struc = Tokenizer.init_structure(n)
    struc["angles"] = df.reset_index()
    t1 = Tokenizer(struc|{"fname":t1.fname})
    t1.visualize(f"./{folder}/test1.png", vis_dihedral=False)

    df = t2.angles_and_dists.iloc[(t2.n-9):]
    n = len(df)
    # remove rows 0 through 173 in this df
    struc = Tokenizer.init_structure(n)
    struc["angles"] = df.reset_index()
    t2 = Tokenizer(struc|{"fname":t2.fname})
    t2.visualize(f"./{folder}/test2.png", vis_dihedral=False)
    
    for start in range(0, 3*t1.n, 3):
        for length in range(3, 3*t1.n - start - 1, 3):
            t1.visualize_bonds(start, length, f"./{folder}/test1_{start}-{start+length}.png", vis_dihedral=False)

    # Visualize all subspans for t2
    for start in range(0, 3*t2.n, 3):
        for length in range(3, 3*t2.n - start - 1, 3):
            t2.visualize_bonds(start, length, f"./{folder}/test2_{start}-{start+length}.png", vis_dihedral=False)    


def debug():
    folder = "ckpts/1753714998.318021"
    bpe = pickle.load(open(f"./{folder}/bpe_init.pkl", "rb"))
    index_0 = 0
    index_1 = 6
    t1 = bpe.tokenizers[index_0]
    t2 = bpe.tokenizers[index_1]
    # vis_subspans(t1, t2, folder)

    # GOAL: replace 3-12 from t2 into 3-12 from t1
    start1 = 3
    start2 = 3
    length = 9
    orig_coords = t1.compute_coords()
    
    # snap internal angles for [s,e] from t2 into t1 (already done)
    t1.set_token_geo(start1, length, t2.token_geo(start2, length))    
    t1.visualize(f"./{folder}/test1_after.png", vis_dihedral=False)
    after_coords = t1.compute_coords()
    error = compute_rmsd(orig_coords, after_coords)    

    t1.opt_glue(start1, length)
    t1.visualize(f"./{folder}/test1_opt.png", vis_dihedral=False)
    after_coords = t1.compute_coords()

    # sanity check: anchored or global RMSD goes down
    err_after  = compute_rmsd(orig_coords, after_coords)                    # after glue opt
    print("RMSD before:", error, "after:", err_after)    


if __name__ == "__main__":
    breakpoint()
    debug()
