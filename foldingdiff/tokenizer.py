import numpy as np

class Tokenizer:
    BOND_TYPES: ["N-CA","CA-C","C-N"]
    ATOM_TYPES: ["N","CA","C"]
    def __init__(self, structure):
        self.angles = structure['angles']
        self.coords = structure['coords']
        self.full_coords = structure['full_coords']   
        self.n = len(self.angles)
        self.bond_labels = sum([[0,1,2] for _ in range(self.n-1)]+[[0,1]], []) # 0,1,2,...,0,1
        self.atom_labels = np.tile([0,1,2], n)
        self.edges = [[j,j+1,0] for j in range(1,3*self.n)]
        self._init_tokens()
    
    
    def _add_tokens(self, tokens):
        self.reps = {}
        for i in range(self._tokens):
            idx, t, l = self._tokens[i]
            if t in tokens:
                self.reps[t] = self.reps.get(t, []) + [self._token_geo(idx,l)]
        for t in self.reps:
            self.reps[t] = np.mean(self.reps[t])


    def _token_geo(self, idx, l):
        """
        Here we want the geometry of bonds idx:idx+l
        To standardize, we always use the internal bond and dihedral angles
        """
        breakpoint()

    
    def _init_tokens(self):
        """
        Here we treat each bond as an inital "bond-type" token, then standardizing the length of the bond to a fixed value
        """
        self._tokens = [(i,self.bond_labels[i],1) for i in range(3*n-1)]
        self._add_tokens([0,1,2])
    
    @property
    def g(self):
        return {
            'nodelabels': np.array(labels, dtype=np.uint32)[:, None],
            'nodepos': np.array(coords, dtype=np.float64),
            'edges': np.array(edges, dtype=np.uint32)
        }        
