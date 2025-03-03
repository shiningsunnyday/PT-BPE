import numpy as np
from foldingdiff.angles_and_coords import *
from foldingdiff.nerf import *
from itertools import starmap
from copy import deepcopy

class Tokenizer:
    BOND_TYPES = ["N:CA","CA:C","0C:1N"]
    ATOM_TYPES = ["N","CA","C"]
    BOND_ANGLES = ["tau","CA:C:1N","C:1N:1CA"]
    DIHEDRAL_ANGLES = ["psi","omega","phi"]

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
        self._init_tokens()
    

    def _init_coords(self):
        self._init_n_ca = np.linalg.norm(N_INIT-CA_INIT)
        self._init_ca_c = np.linalg.norm(CA_INIT-C_INIT)
        self._init_bond_angle = angle_between(N_INIT-CA_INIT, C_INIT-CA_INIT)        
    
    
    def _add_tokens(self, tokens):        
        for idx, t, l in self._tokens: # TODO: use just the pointers
            if t in tokens: # new token
                self.reps[t] = self.reps.get(t, []) + [self._token_geo(idx,l)]
        for t in self.reps:
            self.reps[t] = np.mean(self.reps[t])
    

    def build_nerf(self):
        """
        Here we take...
        """
        breakpoint()


    def _bond_length(self, idx):
        if idx == 0:
            return self._init_n_ca            
        elif idx == 1:
            return self._init_ca_c            
        else:
            return self._angles_and_dists[Tokenizer.BOND_TYPES[idx%3]][idx//3]
        
    def _set_bond_length(self, idx, value):
        if idx == 0:
            self._init_n_ca = value
        elif idx == 1:
            self._init_ca_c = value
        else:
            self._angles_and_dists[Tokenizer.BOND_TYPES[idx%3]].iloc[idx//3] = value
    
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

    def _token_geo(self, idx, l):
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

    def _set_token_geo(self, idx, l, vals):
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

    
    def _init_tokens(self):
        """
        Here we treat each bond as an inital "bond-type" token, then standardizing the length of the bond to a fixed value
        """
        new_tokens = [(i,self.bond_labels[i],1) for i in range(3*self.n-1)]
        self.res = {0:[],1:[],2:[]}
        for i,t,l in new_tokens:
            self.res[t].append(self._token_geo(i, l))
        for k in self.res:
            self.res[k] = {
                key: list(
                    starmap(lambda *vals: sum(vals) / len(vals), zip(*(d[key] for d in self.res[k])))
                )
                for key in self.res[k][0]
            }
        # update avg bond lengths
        for i,t,l in new_tokens:
            dic = self.res[t]
            self._set_token_geo(i, l, dic)


    @property
    def angles_and_dists(self):
        return self._angles_and_dists
    
    @angles_and_dists.setter
    def angles_and_dists(self, i, attr, value):
        breakpoint()
        self._update_coords()

    @property
    def g(self):
        return {
            'nodelabels': np.array(labels, dtype=np.uint32)[:, None],
            'nodepos': np.array(coords, dtype=np.float64),
            'edges': np.array(edges, dtype=np.uint32)
        }        
