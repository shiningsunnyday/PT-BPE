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
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        left_val = self.left.value if self.left else None
        right_val = self.right.value if self.right else None
        return f"Node({self.value}, left={left_val}, right={right_val})"


class BinaryTreeBuilder:
    def __init__(self):
        # Use a regular dict mapping keys to nodes.
        self.nodes = {}

    def add_leaf(self, key):
        """
        Add a new leaf node with a unique key.
        """
        if key in self.nodes:
            raise ValueError(f"Key '{key}' already exists.")
        self.nodes[key[0]] = Node(key)
        print(f"Added leaf with key '{key}'.")

    def combine(self, left_key, right_key, parent_key):
        """
        Combine the two nodes identified by left_key and right_key into a new parent node.
        
        - left_key: the key of the node to be used as the left child.
        - right_key: the key of the node to be used as the right child.
        - parent_key: the key for the new parent node.
        
        Both keys must exist in the dictionary.
        After combining, the two nodes are removed, and the new parent node is added.
        """
        if left_key[0] not in self.nodes or right_key[0] not in self.nodes:
            raise ValueError("Both keys must exist in the nodes dictionary.")
        if parent_key in self.nodes:
            print(f"Popping Parent key '{parent_key}'.")
            self.nodes.pop(parent_key)

        left_node = self.nodes[left_key[0]]
        right_node = self.nodes[right_key[0]]
        assert left_node.value == left_key
        assert right_node.value == right_key
        parent = Node(parent_key, left_node, right_node)

        # Remove the two child nodes.
        del self.nodes[left_key[0]]
        del self.nodes[right_key[0]]
        # Insert the new parent node.
        self.nodes[parent_key[0]] = parent
        print(f"Combined nodes '{left_key}' and '{right_key}' into new parent with key '{parent_key}'.")

    def get_tree(self):
        """
        Return the sole remaining node if the tree has been fully combined;
        otherwise, return the current dictionary of nodes.
        """
        if len(self.nodes) == 1:
            return next(iter(self.nodes.values()))
        return self.nodes

    def visualize(self, path, node_radius=0.3, horizontal_gap=1.0, vertical_gap=1.5, margin=1.0, font_size=12):
        """
        Visualize the tree (or forest) by recursively assigning positions so that:
          - In each tree, all leaves are equally spaced along the bottom.
          - Each internal node's x position is the average of its children.
          - The y position is set by the node's depth.
        If there are multiple trees (i.e. the forest is not fully combined), all leaves
        are aligned at the bottom. The figure size is dynamically adjusted based on the extent
        of the tree.
        The resulting image is saved to the file at the given path.
        """
        # Get the structure. It might be a single Node or a dictionary (a forest).
        tree_structure = self.get_tree()
        if isinstance(tree_structure, Node):
            trees = [tree_structure]
        else:
            trees = sorted(list(self.nodes.values()), key=lambda node: node.value)

        positions = {}  # Final mapping of each node to its (x, y) coordinate.
        x_counter = [0]  # Global counter for leaf x positions.
        tree_max_depths = {}  # To store maximum depth for each tree.

        def assign_positions(node, depth, pos_dict):
            """
            Recursively assign positions:
              - For a leaf, assign the next available x value using x_counter.
              - For an internal node, the x position is the average of its children.
            Returns a tuple (x_position, max_depth) for the subtree rooted at node.
            """
            if node.left is None and node.right is None:
                x = x_counter[0] * horizontal_gap
                x_counter[0] += 1
                pos_dict[node] = (x, -depth * vertical_gap)
                return x, depth
            else:
                left_x, left_depth = assign_positions(node.left, depth + 1, pos_dict)
                right_x, right_depth = assign_positions(node.right, depth + 1, pos_dict)
                x = (left_x + right_x) / 2.0
                pos_dict[node] = (x, -depth * vertical_gap)
                return x, max(left_depth, right_depth)

        all_positions = {}
        for tree in trees:
            pos_dict = {}
            _, max_depth = assign_positions(tree, 0, pos_dict)
            tree_max_depths[tree] = max_depth
            all_positions.update(pos_dict)

        # Compute global maximum depth.
        global_max_depth = max(tree_max_depths.values()) if tree_max_depths else 0

        # Helper to check if target is in subtree rooted at root.
        def in_subtree(root, target):
            if root is target:
                return True
            if root is None:
                return False
            return in_subtree(root.left, target) or in_subtree(root.right, target)

        # Adjust vertical positions so all trees' leaves align.
        for tree in trees:
            delta_y = -global_max_depth * vertical_gap - (-tree_max_depths[tree] * vertical_gap)
            for node in all_positions:
                if in_subtree(tree, node):
                    x, y = all_positions[node]
                    all_positions[node] = (x, y + delta_y)

        positions = all_positions

        # Calculate extents to dynamically adjust the figure size.
        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        fig_width = (max_x - min_x + 2 * margin)
        fig_height = (max_y - min_y + 2 * margin)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        def draw_edges(node):
            x, y = positions[node]
            if node.left:
                child_x, child_y = positions[node.left]
                ax.plot([x, child_x], [y, child_y], 'k-', lw=1)
                draw_edges(node.left)
            if node.right:
                child_x, child_y = positions[node.right]
                ax.plot([x, child_x], [y, child_y], 'k-', lw=1)
                draw_edges(node.right)

        # Draw edges for each tree.
        for tree in trees:
            draw_edges(tree)

        # Draw nodes as circles with labels.
        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), node_radius, color='skyblue', ec='black', zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, str(node.value), fontsize=font_size, ha='center', va='center', zorder=4)

        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Tree visualization saved to {path}")

class TokenHierarchy(dict):
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        super().__init__(*args, **kwargs)
        self.init()
    
    def init(self):
        # check is sorted
        prev = None
        for n in self:
            if prev:
                assert prev[0]+prev[-1] == self[n][0]
            prev = self[n]
        self.tree = BinaryTreeBuilder()
        for n in self:
            self.tree.add_leaf(self[n])
    
    def __setitem__(self, k, v):      
        super().__setitem__(k, v)
        # update parent hierarchy
        try:
            left_key = self.tree.nodes[k].value
        except:
            return
        right_key = self.tree.nodes[left_key[0]+left_key[2]].value
        assert right_key[0]+right_key[2] == v[0]+v[2]
        self.tree.combine(left_key, right_key, v)
        

    def __delitem__(self, key):
        breakpoint()
        # Custom behavior when deleting an item.
        self.parent.last_action = f"__delitem__: {key}"
        print(f"[TokenHierarchy] Deleting key {key}")
        return super().__delitem__(key)
    
    def pop(self, key):        
        # Custom behavior when popping an item.        
        value = super().pop(key)
        print(f"[TokenHierarchy] Popped key {key} with value {value}")
        return value
    
    def popitem(self):
        breakpoint()
        # Custom behavior for popitem.
        key, value = super().popitem()
        self.parent.last_action = f"popitem: removed {key} with value {value}"
        print(f"[TokenHierarchy] Popped item ({key}: {value})")
        return key, value
    
    def update(self, *args, **kwargs):
        breakpoint()
        # Custom behavior for updating the dictionary.
        super().update(*args, **kwargs)
        self.parent.last_action = "update: dictionary updated"
        print(f"[TokenHierarchy] Dictionary updated with args {args} and kwargs {kwargs}")

    def clear(self):
        raise NotImplementedError    
    
    def setdefault(self, key, default=None):
        raise NotImplementedError   


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


# [(460, '-'), (461, 'H'), (462, 'H'), (463, 'H'), (464, 'H'), (465, 'H'), (466, 'H'), (467, 'H'), (468, 'H'), (469, 'H'), (470, 'H'), (471, 'H'), (472, 'H'), (473, 'H'), (474, 'H'), (475, 'S'), (476, '-'), (477, '-'), (478, 'S'), (479, 'S'), (480, '-'), (481, 'H'), (482, 'H'), (483, 'H'), (484, 'H'), (485, 'H'), (486, 'H'), (487, 'H'), (488, 'H'), (489, 'H'), (490, 'H'), (491, 'H'), (492, 'H'), (493, 'H'), (494, 'H'), (495, '-'), (496, 'T'), (497, 'T'), (498, '-'), (499, 'S'), (500, '-'), (501, 'H'), (502, 'H'), (503, 'H'), (504, 'H'), (505, 'H'), (506, 'H'), (507, 'H'), (508, 'H'), (509, 'H'), (510, 'H'), (511, 'H'), (512, 'H'), (513, 'H'), (514, 'H'), (515, 'H'), (516, 'H'), (517, 'H'), (518, 'H'), (519, 'T'), (520, 'S'), (521, 'S'), (522, '-'), (523, '-'), (524, 'S'), (525, 'S'), (526, 'S'), (527, '-'), (528, '-'), (529, 'H'), (530, 'H'), (531, 'H'), (532, 'H'), (533, 'H'), (534, 'H'), (535, 'H'), (536, 'H'), (537, 'H'), (538, 'H'), (539, 'H'), (540, 'H'), (541, 'H'), (542, 'H'), (543, 'H'), (544, 'H'), (545, '-'), (546, 'H'), (547, 'H'), (548, 'H'), (549, 'H'), (550, 'H'), (551, 'H'), (552, 'H'), (553, 'H'), (554, 'H'), (555, 'H'), (556, '-')]
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
    
    def visualize_bonds(self, i1, length, output_path):
        coords = self.compute_coords(i1, length)
        # ATOM_TYPES[i1%3], ATOM_TYPES[i1%3+1], ..., ATOM_TYPES[i1%3+length]
        bts = [Tokenizer.ATOM_TYPES[(i1%3+i)%3] for i in range(length+1)]
        plot_backbone(coords, output_path, bts, title=f"{Path(self.fname).stem} bonds {i1}-{i1+length-1}", zoom_factor=1.0)        
    
    def visualize(self, output_path, **kwargs):
        coords = self.compute_coords()
        tokens = [self.bond_to_token[i] for i in sorted(self.bond_to_token)]
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
