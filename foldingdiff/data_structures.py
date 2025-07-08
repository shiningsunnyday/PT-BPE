import matplotlib.pyplot as plt


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
        self.leaves = {}

    def add_leaf(self, key):
        """
        Add a new leaf node with a unique key.
        """
        if key in self.nodes:
            raise ValueError(f"Key '{key}' already exists.")
        self.nodes[key[0]] = Node(key)
        self.leaves[key[0]] = Node(key)
        # print(f"Added leaf with key '{key}'.")

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
            # print(f"Popping Parent key '{parent_key}'.")
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
        # print(f"Combined nodes '{left_key}' and '{right_key}' into new parent with key '{parent_key}'.")
    
    def split(self, parent_key, left_key, right_key):
        """
        Split parent_key into left_key, right_key

        - left: the key of the new left node
        - right: the key of the new right node
        """
        assert self.nodes[parent_key[0]].left is None
        assert self.nodes[parent_key[0]].right is None
        left_node = Node(left_key)
        right_node = Node(right_key)
        self.leaves.pop(parent_key[0])
        self.leaves[left_key[0]] = left_node
        self.leaves[right_key[0]] = right_node
        self.nodes[parent_key[0]].left = left_node
        self.nodes[parent_key[0]].right = right_node

        

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


class ThresholdDict(dict): # only add, no remove
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.max_key = -1
        return instance

    def __init__(self):
        super().__init__()
        self.max_key = -1

    def __setitem__(self, key, val):
        if isinstance(key, int) and key > self.max_key:
            self.max_key = key
        super().__setitem__(key, val)
    
    def __getitem__(self, key):
        if isinstance(key, int) and key > self.max_key:
            key = self.max_key        
        return super().__getitem__(key)