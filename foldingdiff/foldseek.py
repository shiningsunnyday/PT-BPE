import numpy as np

foldseek_feat_labels = [
    "cos_phi12",            # cos(u1, u2): backbone angle at residue i
    "cos_phi34",            # cos(u3, u4): backbone angle at partner residue j
    "cos_phi15",            # cos(u1, u5): angle between i’s backbone and the i–j vector
    "cos_phi35",            # cos(u3, u5): angle between j’s backbone and the i–j vector
    "cos_phi14",            # cos(u1, u4): angle between i’s first backbone vector and j’s second
    "cos_phi23",            # cos(u2, u3): angle between i’s second backbone vector and j’s first
    "cos_phi13",            # cos(u1, u3): angle between first backbone directions of i and j
    "ca_ca_distance",       # Euclidean distance |Cα_i – Cα_j|
    "seq_sep_clamped",      # signed sequence separation min(|i–j|,4)
    "seq_sep_log"           # signed log(|i–j|+1) sequence separation
]

# --- Basic vector operations ---

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def scale(a: np.ndarray, f: float) -> np.ndarray:
    return a * f


def degree_to_radians(deg: float) -> float:
    return deg * (np.pi / 180.0)

# --- Constants ---
DISTANCE_ALPHA_BETA = 1.53
VIRTUAL_CENTER_ALPHA = 270.0
VIRTUAL_CENTER_BETA = 0.0
VIRTUAL_CENTER_D = 2.0
FEATURE_COUNT = 10

# --- Cβ reconstruction ---

def approx_cbeta_position(ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1 = norm(sub(c, ca))
    v2 = norm(sub(n, ca))
    b1 = add(v2, scale(v1, 1/3.0))
    b2 = cross(v1, b1)
    u1 = norm(b1)
    u2 = norm(b2)
    v4 = add(
        scale(v1, -1/3.0),
        scale(
            sub(scale(u1, -0.5), scale(u2, np.sqrt(3)/2.0)),
            np.sqrt(8)/3.0
        )
    )
    return add(ca, scale(v4, DISTANCE_ALPHA_BETA))

# --- Virtual center computation ---

def calc_virtual_center(
    ca: np.ndarray,
    cb: np.ndarray,
    n: np.ndarray,
    alpha: float = VIRTUAL_CENTER_ALPHA,
    beta: float = VIRTUAL_CENTER_BETA,
    d: float = VIRTUAL_CENTER_D
) -> np.ndarray:
    alpha_r = degree_to_radians(alpha)
    beta_r = degree_to_radians(beta)
    v = sub(cb, ca)
    a = sub(cb, ca)
    b = sub(n, ca)
    k = norm(cross(a, b))
    v = add(
        scale(v, np.cos(alpha_r)),
        add(
            scale(cross(k, v), np.sin(alpha_r)),
            scale(k, dot(k, v)*(1 - np.cos(alpha_r)))
        )
    )
    k = norm(sub(n, ca))
    v = add(
        scale(v, np.cos(beta_r)),
        add(
            scale(cross(k, v), np.sin(beta_r)),
            scale(k, dot(k, v)*(1 - np.cos(beta_r)))
        )
    )
    return add(ca, scale(v, d))

# --- Distance ---

def calc_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# --- Replace CB with virtual center ---

def replace_cb_with_virtual_center(
    ca: np.ndarray,
    n: np.ndarray,
    c: np.ndarray,
    cb: np.ndarray
) -> None:
    L = ca.shape[0]
    for i in range(L):
        if np.isnan(cb[i]).any():
            cb[i] = approx_cbeta_position(ca[i], n[i], c[i])
        cb[i] = calc_virtual_center(ca[i], cb[i], n[i])

# --- Mask creation ---

def create_residue_mask(
    ca: np.ndarray,
    n: np.ndarray,
    c: np.ndarray
) -> np.ndarray:
    return ~(
        np.isnan(ca).any(axis=1) |
        np.isnan(n).any(axis=1) |
        np.isnan(c).any(axis=1)
    )

# --- Nearest-neighbor partner selection ---

def find_residue_partners(
    cb: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    L = cb.shape[0]
    partner_idx = -np.ones(L, dtype=int)
    for i in range(1, L-1):
        if not mask[i]:
            continue
        best_d = np.inf
        best_j = -1
        for j in range(1, L-1):
            if j == i or not mask[j]:
                continue
            d = calc_distance(cb[i], cb[j])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j < 0:
            mask[i] = False
        else:
            partner_idx[i] = best_j
    return partner_idx

# --- Feature calculation ---

def calc_features(ca: np.ndarray, i: int, j: int) -> np.ndarray:
    u1 = norm(ca[i] - ca[i-1])
    u2 = norm(ca[i+1] - ca[i])
    u3 = norm(ca[j] - ca[j-1])
    u4 = norm(ca[j+1] - ca[j])
    u5 = norm(ca[j] - ca[i])
    f = np.zeros(FEATURE_COUNT)
    f[0] = dot(u1, u2)
    f[1] = dot(u3, u4)
    f[2] = dot(u1, u5)
    f[3] = dot(u3, u5)
    f[4] = dot(u1, u4)
    f[5] = dot(u2, u3)
    f[6] = dot(u1, u3)
    f[7] = calc_distance(ca[i], ca[j])
    delta = j - i
    f[8] = np.sign(delta) * min(abs(delta), 4)
    f[9] = np.sign(delta) * np.log(abs(delta) + 1)
    return f

# --- Conformation descriptors ---

def calc_conformation_descriptors(
    ca: np.ndarray,
    partner_idx: np.ndarray,
    mask: np.ndarray
) -> (np.ndarray, np.ndarray):
    L = ca.shape[0]
    feats = np.zeros((L, FEATURE_COUNT))
    mask_copy = mask.copy()
    for i in range(1, L-1):
        j = partner_idx[i]
        if (mask_copy[i-1] and mask_copy[i] and mask_copy[i+1] and
            mask_copy[j-1] and mask_copy[j] and mask_copy[j+1]):
            feats[i] = calc_features(ca, i, j)
        else:
            mask[i] = False
    mask[0] = False
    mask[-1] = False
    return feats, mask

# --- Main interface up to descriptor ---

def structure2features(
    ca: np.ndarray,
    n_arr: np.ndarray,
    c: np.ndarray,
    cb: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Converts backbone atom coordinates to 3Di features.

    Returns:
      feats: (L, FEATURE_COUNT) descriptor array
      mask: boolean mask of valid residues
      partner_idx: int array of nearest neighbor indices
    """
    # 1. Virtual center
    replace_cb_with_virtual_center(ca, n_arr, c, cb)
    # 2. Mask
    mask = create_residue_mask(ca, n_arr, c)
    # 3. Partners
    partner_idx = find_residue_partners(cb, mask)
    # 4. Descriptors
    feats, mask = calc_conformation_descriptors(ca, partner_idx, mask)
    return feats, mask, partner_idx
