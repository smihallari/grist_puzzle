import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.distance import cdist


def extract_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blurs, grads, coarses, smalls = [], [], [], []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read frame: {p}")

        img64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        lab64 = cv2.cvtColor(img64, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray64 = cv2.cvtColor(img64, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Blurred Lab (smooth linear response to small pixel motion)
        blurs.append(cv2.GaussianBlur(lab64, (0, 0), 1.5).ravel())

        # 2. Z-scored Sobel gradients (invariant to exposure/brightness flicker; safe on blank frames)
        gx = cv2.Sobel(gray64, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray64, cv2.CV_32F, 0, 1, ksize=3)
        g = np.concatenate([gx.ravel(), gy.ravel()])
        std = float(g.std())
        if std > 1e-5:
            g = (g - float(g.mean())) / std
        else:
            g.fill(0.0)
        grads.append(g)

        # 3. Coarse 16x16 Lab grid (survives large camera pans)
        coarses.append(cv2.resize(lab64, (16, 16), interpolation=cv2.INTER_AREA).ravel())

        # 4. Blurred 32x32 grayscale for 2nd-order acceleration penalty
        g32 = cv2.resize(gray64, (32, 32), interpolation=cv2.INTER_AREA)
        smalls.append(cv2.GaussianBlur(g32, (0, 0), 1.5).ravel())

    return np.asarray(blurs), np.asarray(grads), np.asarray(coarses), np.asarray(smalls)


def _norm_matrix(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    if n < 2:
        return m
    med = float(np.median(m[np.triu_indices(n, k=1)]))
    return m / med if med > 1e-9 else m


def build_distance_matrix(
    blurs: np.ndarray, grads: np.ndarray, coarses: np.ndarray, weights=(0.4, 0.4, 0.2)
) -> np.ndarray:
    d_blur = _norm_matrix(cdist(blurs, blurs, metric="cityblock"))
    d_grad = _norm_matrix(cdist(grads, grads, metric="cityblock"))
    d_coarse = _norm_matrix(cdist(coarses, coarses, metric="cityblock"))
    w1, w2, w3 = weights
    d = w1 * d_blur + w2 * d_grad + w3 * d_coarse
    np.fill_diagonal(d, 0.0)
    return d


def multi_fragment_path(d: np.ndarray) -> list[int]:
    n = d.shape[0]
    if n <= 2:
        return list(range(n))

    iu, iv = np.triu_indices(n, k=1)
    order = np.argsort(d[iu, iv])

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: list[list[int]] = [[] for _ in range(n)]
    edges = 0
    for idx in order:
        u, v = int(iu[idx]), int(iv[idx])
        if len(adj[u]) == 2 or len(adj[v]) == 2:
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        parent[ru] = rv
        adj[u].append(v)
        adj[v].append(u)
        edges += 1
        if edges == n - 1:
            break

    start = next(i for i in range(n) if len(adj[i]) == 1)
    path = [start]
    prev, curr = -1, start
    while len(path) < n:
        nxt = adj[curr][0] if adj[curr][0] != prev else adj[curr][1]
        path.append(nxt)
        prev, curr = curr, nxt
    return path


def two_opt_open(path: list[int], d: np.ndarray, k: int = 20) -> list[int]:
    n = len(path)
    if n < 4:
        return path

    # Append dummy node n with 0 cost to all frames so open path == closed cycle
    m = n + 1
    d_ext = np.zeros((m, m), dtype=np.float64)
    d_ext[:n, :n] = d
    dm = d_ext.tolist()

    k_eff = min(k, n - 1)
    cand = [
        [int(x) for x in np.argpartition(d[u], k_eff)[: k_eff + 1] if x != u] + [n]
        for u in range(n)
    ]

    cycle = [n] + list(path)
    pos = [0] * m
    for idx, node in enumerate(cycle):
        pos[node] = idx

    improved = True
    while improved:
        improved = False
        for i in range(m):
            a = cycle[i]
            b = cycle[(i + 1) % m]
            js: set[int] = set()
            if a != n:
                for nb in cand[a]:
                    js.add(pos[nb])
            if b != n:
                for nb in cand[b]:
                    js.add((pos[nb] - 1) % m)

            for j in js:
                if j == i or (i + 1) % m == j or (j + 1) % m == i:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                u1, v1 = cycle[lo], cycle[lo + 1]
                u2, v2 = cycle[hi], cycle[(hi + 1) % m]
                delta = (dm[u1][u2] + dm[v1][v2]) - (dm[u1][v1] + dm[u2][v2])
                if delta < -1e-9:
                    cycle[lo + 1 : hi + 1] = cycle[lo + 1 : hi + 1][::-1]
                    for idx in range(lo + 1, hi + 1):
                        pos[cycle[idx]] = idx
                    improved = True
                    break

    cut = cycle.index(n)
    return cycle[cut + 1 :] + cycle[:cut]


def or_opt_open(path: list[int], d: np.ndarray, k: int = 15) -> list[int]:
    n = len(path)
    if n < 5:
        return path

    dm = d.tolist()
    k_eff = min(k, n - 1)
    top_k = [
        [int(x) for x in np.argpartition(d[u], k_eff)[: k_eff + 1] if x != u]
        for u in range(n)
    ]

    p = list(path)
    for block_len in (1, 2, 3):
        improved = True
        passes = 0
        while improved and passes < 25:
            improved = False
            passes += 1
            pos = {node: idx for idx, node in enumerate(p)}
            i = 0
            while i <= n - block_len:
                head, tail = p[i], p[i + block_len - 1]
                left = p[i - 1] if i > 0 else -1
                right = p[i + block_len] if i + block_len < n else -1

                rem_gain = (
                    (dm[left][head] if left != -1 else 0.0)
                    + (dm[tail][right] if right != -1 else 0.0)
                    - (dm[left][right] if (left != -1 and right != -1) else 0.0)
                )

                slots = {0, n}
                for nb in top_k[head]:
                    idx = pos[nb]
                    slots.add(idx)
                    slots.add(idx + 1)
                for nb in top_k[tail]:
                    idx = pos[nb]
                    slots.add(idx)
                    slots.add(idx + 1)

                best_delta = -1e-9
                best_move = None
                for ins in slots:
                    if i <= ins <= i + block_len:
                        continue
                    ins_l = p[ins - 1] if ins > 0 else -1
                    ins_r = p[ins] if ins < n else -1
                    cut_cost = dm[ins_l][ins_r] if (ins_l != -1 and ins_r != -1) else 0.0

                    for rev in (False, True) if block_len > 1 else (False,):
                        h, t = (tail, head) if rev else (head, tail)
                        add_cost = (
                            (dm[ins_l][h] if ins_l != -1 else 0.0)
                            + (dm[t][ins_r] if ins_r != -1 else 0.0)
                            - cut_cost
                        )
                        delta = add_cost - rem_gain
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (ins, rev)

                if best_move is not None:
                    ins, rev = best_move
                    block = p[i : i + block_len]
                    if rev:
                        block = block[::-1]
                    if ins < i:
                        p = p[:ins] + block + p[ins:i] + p[i + block_len :]
                    else:
                        p = p[:i] + p[i + block_len : ins] + block + p[ins:]
                    pos = {node: idx for idx, node in enumerate(p)}
                    improved = True
                    break
                i += 1
    return p


def refine_second_order(
    path: list[int], d: np.ndarray, smalls: np.ndarray, alpha: float = 0.15, k: int = 15
) -> list[int]:
    n = len(path)
    if n < 4:
        return path

    steps = [d[path[i], path[i + 1]] for i in range(n - 1)]
    lam = alpha * float(np.median(steps))
    if lam <= 1e-9:
        return path

    dm = d.tolist()
    k_eff = min(k, n - 1)
    top_k = [
        [int(x) for x in np.argpartition(d[u], k_eff)[: k_eff + 1] if x != u]
        for u in range(n)
    ]

    cache: dict[tuple[int, int, int], float] = {}

    raw_steps = [float(np.abs(smalls[path[i + 1]] - smalls[path[i]]).sum()) for i in range(n - 1)]
    eps = 0.25 * float(np.median(raw_steps)) + 1e-6
    def triplet(a: int, b: int, c: int) -> float:
        key = (a, b, c) if a <= c else (c, b, a)
        val = cache.get(key)
        if val is None:
            ga, gb, gc = smalls[a], smalls[b], smalls[c]
            d1 = float(np.abs(gb - ga).sum())
            d2 = float(np.abs(gc - gb).sum())
            acc = float(np.abs(ga - 2.0 * gb + gc).sum())
            
            val = acc / (d1 + d2 + eps)
            cache[key] = val
        return val

    p = list(path)
    pos = [0] * n
    for idx, node in enumerate(p):
        pos[node] = idx

    def try_reverse(l: int, r: int) -> bool:
        if l < 0 or r >= n or l >= r:
            return False
        old_d = (dm[p[l - 1]][p[l]] if l > 0 else 0.0) + (
            dm[p[r]][p[r + 1]] if r < n - 1 else 0.0
        )
        new_d = (dm[p[l - 1]][p[r]] if l > 0 else 0.0) + (
            dm[p[l]][p[r + 1]] if r < n - 1 else 0.0
        )

        centers = {c for c in (l - 1, l, r, r + 1) if 1 <= c <= n - 2}

        def at(idx: int) -> int:
            return p[l + r - idx] if l <= idx <= r else p[idx]

        old_t = sum(triplet(p[c - 1], p[c], p[c + 1]) for c in centers)
        new_t = sum(triplet(at(c - 1), at(c), at(c + 1)) for c in centers)

        if (new_d - old_d) + lam * (new_t - old_t) < -1e-9:
            p[l : r + 1] = p[l : r + 1][::-1]
            for idx in range(l, r + 1):
                pos[p[idx]] = idx
            return True
        return False

    improved = True
    passes = 0
    while improved and passes < 20:
        improved = False
        passes += 1
        # Check adjacent swaps (fixes 1-frame jitter) and neighbor-guided 2-opt
        for i in range(n - 1):
            if try_reverse(i, i + 1):
                improved = True
        for i in range(n):
            for nb in top_k[p[i]]:
                j = pos[nb]
                if i + 1 < j and try_reverse(i + 1, j):
                    improved = True
                    break
                if j + 1 < i and try_reverse(j + 1, i):
                    improved = True
                    break
    return p

def estimate_direction_score(paths: list[Path], order: list[int], max_pairs: int = 40) -> float:
    """
    Returns positive score if `order` looks forward, negative if reversed.
    Magnitude roughly reflects confidence (near 0 == ambiguous).
    """
    n = len(order)
    if n < 3:
        return 0.0

    # Sample evenly across the sequence with a small temporal stride for cleaner flow
    stride = max(1, min(3, n // 20))
    step = max(1, (n - stride) // max_pairs)
    pair_indices = list(range(0, n - stride, step))

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    w, h = 160, 90

    # Precompute centered coordinate grid for least-squares affine fit: [x, y, 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx - w * 0.5) / (w * 0.5)
    yn = (yy - h * 0.5) / (h * 0.5)
    # Trim 10% border where optical flow wraps/breaks
    sl = (slice(9, h - 9), slice(16, w - 16))
    X = np.stack([xn[sl].ravel(), yn[sl].ravel(), np.ones(xn[sl].size, dtype=np.float32)], axis=1)
    pinv_X = np.linalg.pinv(X)

    gray_cache: dict[int, np.ndarray] = {}

    def get_gray(idx: int) -> np.ndarray:
        g = gray_cache.get(idx)
        if g is None:
            img = cv2.imread(str(paths[order[idx]]), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            gray_cache[idx] = g
        return g

    div_scores = []
    grav_scores = []

    for i in pair_indices:
        g1 = get_gray(i)
        g2 = get_gray(i + stride)
        flow = dis.calc(g1, g2, None)
        u = flow[sl][..., 0].ravel()
        v = flow[sl][..., 1].ravel()

        # 1. Affine expansion rate (a + d): positive = expanding / moving forward
        coef_u = pinv_X @ u
        coef_v = pinv_X @ v
        div = float(coef_u[0] + coef_v[1])
        div_scores.append(div)

        # 2. Foreground residual vertical velocity: positive = downward
        ru = u - (X @ coef_u)
        rv = v - (X @ coef_v)
        mag = np.hypot(ru, rv)
        fg = mag > max(0.5, float(np.percentile(mag, 85)))
        if np.any(fg):
            grav_scores.append(float(np.median(rv[fg])))

    # Normalize each cue gently; weight expansion higher than residual vertical bias
    div_med = float(np.median(div_scores)) if div_scores else 0.0
    grav_med = float(np.median(grav_scores)) if grav_scores else 0.0
    return 1.0 * div_med + 0.35 * grav_med

def solve_dir(frames_dir: Path, use_second_order: bool = True) -> list[str]:
    paths = sorted(frames_dir.glob("*.png"))
    if len(paths) <= 1:
        return [p.name for p in paths]

    blurs, grads, coarses, smalls = extract_features(paths)
    d = build_distance_matrix(blurs, grads, coarses)
    order = multi_fragment_path(d)
    order = two_opt_open(order, d)
    order = or_opt_open(order, d)
    if use_second_order:
        order = refine_second_order(order, d, smalls, alpha=0.5)
    if estimate_direction_score(paths, order) < 0.0:
        order.reverse()
    return [paths[i].name for i in order]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/frames")
    for name in solve_dir(target):
        print(name)
        