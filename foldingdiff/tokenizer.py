import numpy as np
from foldingdiff.angles_and_coords import *
from foldingdiff.nerf import *
from foldingdiff.plotting import plot_backbone
from types import SimpleNamespace
from itertools import starmap
from copy import deepcopy
import tempfile
import nglview as nv
import imageio
import time

class Tokenizer:
    BOND_TYPES = ["N:CA","CA:C","0C:1N"]
    ATOM_TYPES = ["N","CA","C"]
    BOND_ANGLES = ["tau","CA:C:1N","C:1N:1CA"]
    DIHEDRAL_ANGLES = ["psi","omega","phi"]
    BOND_LENGTHS = [N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH]
    def __init__(self, structure):
        self._angles_and_dists = structure['angles'] # controls coords
        self._coords = structure['coords']
        self._full_coords = structure['full_coords']   
        self.n = len(self._angles_and_dists) # fixed
        # fixed
        self.bond_labels = sum([[0,1,2] for _ in range(self.n-1)]+[[0,1]], []) # 0,1,2,...,0,1
        self.atom_labels = np.tile([0,1,2], self.n)
        self.edges = [[j,j+1,0] for j in range(1,3*self.n)]
        # init
        self._init_coords()
    

    def _init_coords(self):
        self._init_n_ca = np.linalg.norm(N_INIT-CA_INIT)
        self._init_ca_c = np.linalg.norm(CA_INIT-C_INIT)
        self._init_bond_angle = angle_between(N_INIT-CA_INIT, C_INIT-CA_INIT)        
    
    
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
    
    def visualize(self, output_path):
        coords = self.compute_coords()
        plot_backbone(coords, output_path, atom_types=np.tile(Tokenizer.ATOM_TYPES, len(coords)//3), tokens=self.tokens)
        

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
        num_bonds = (len(geo['N:CA'])+len(geo['CA:C'])+len(geo.get('0C:1N', [])))
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
