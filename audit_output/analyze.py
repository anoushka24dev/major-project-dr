"""Analyze image_stats.jsonl produced by scan_images.py (read-only)."""
import os, json, collections, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
recs = []
with open(os.path.join(HERE, "image_stats.jsonl"), encoding="utf-8") as f:
    for line in f:
        recs.append(json.loads(line))

print(f"records: {len(recs)}")
ok = [r for r in recs if "error" not in r]
bad = [r for r in recs if "error" in r]
print(f"readable: {len(ok)}  unreadable: {len(bad)}")
for r in bad[:10]:
    print("  ERR", r["path"], r["error"])

# ---------- integrity: dimensions, modes, formats ----------
dims = collections.Counter((r["width"], r["height"]) for r in ok)
modes = collections.Counter(r["mode"] for r in ok)
fmts = collections.Counter(r["fmt"] for r in ok)
print("\nformats:", dict(fmts))
print("modes:", dict(modes))
print(f"unique resolutions: {len(dims)}")
print("top resolutions:", dims.most_common(10))
ws = [r["width"] for r in ok]; hs = [r["height"] for r in ok]
print(f"width range: {min(ws)}..{max(ws)}  height range: {min(hs)}..{max(hs)}")
areas = [(r["width"], r["height"], r["path"]) for r in ok]
small = sorted(areas, key=lambda x: x[0] * x[1])[:15]
print("smallest images:", small)

# ---------- suspicious flags ----------
def flag(r):
    fl = []
    if r["width"] < 300 or r["height"] < 300: fl.append("tiny_dims")
    if r["lum_std"] < 12: fl.append("near_uniform_low_std")
    if r["edge_energy"] < 1.2: fl.append("low_texture")
    if r["sat_mean"] < 18: fl.append("near_grayscale")
    if r["frac_dark"] > 0.55: fl.append("mostly_black")
    if r["frac_bright"] > 0.35: fl.append("mostly_white")
    if r["center_mean"] > 10 and r["border_mean"] / max(r["center_mean"], 1) < 0.25: fl.append("black_border")
    if r["center_mean"] < 255 and r["border_mean"] / max(r["center_mean"], 1) > 2.5: fl.append("white_border")
    if r["lum_mean"] > 200: fl.append("very_bright")
    if r["lum_mean"] < 30: fl.append("very_dark")
    return fl

flagged = []
for r in ok:
    fl = flag(r)
    if fl: flagged.append((r, fl))
fc = collections.Counter(f for _, fl in flagged for f in fl)
print("\nsuspicious flags:", dict(fc))
print(f"flagged files total: {len(flagged)}")
# most-flagged examples per flag type
for f in fc:
    ex = [r["path"] for r, fl in flagged if f in fl][:6]
    print(f"  {f}: {ex}")

# ---------- exact duplicates ----------
byhash = collections.defaultdict(list)
for r in ok:
    byhash[r["sha256"]].append(r["path"])
dups = {h: ps for h, ps in byhash.items() if len(ps) > 1}
dup_files = sum(len(ps) for ps in dups.values())
print(f"\nEXACT DUPLICATES (sha256): {len(dups)} groups, {dup_files} files involved")
cross = {h: ps for h, ps in dups.items() if len(set(p.split('/')[0] for p in ps)) > 1}
print(f"  cross-class groups: {len(cross)}")
for h, ps in list(dups.items())[:8]:
    print("   grp:", ps)
if cross:
    print("  CROSS-CLASS examples:")
    for h, ps in list(cross.items())[:20]:
        print("   ", ps)

# duplicated within same class only
same = sum(1 for ps in dups.values() if len(set(p.split('/')[0] for p in ps)) == 1)
print(f"  within-class groups: {same}")

# ---------- near duplicates via perceptual hashes ----------
n = len(ok)
dhash = np.frombuffer(b"".join(bytes.fromhex(r["dhash"].rjust(16, "0")) for r in ok), dtype=">u8").astype(np.uint64)
ahash = np.frombuffer(b"".join(bytes.fromhex(r["ahash"].rjust(64, "0")) for r in ok), dtype=">u8").reshape(n, 4).astype(np.uint64)
paths = [r["path"] for r in ok]

def near_pairs(h1, h2, maxdist, name):
    pairs = []
    block = 512
    for i in range(0, n, block):
        a = h1[i:i + block, None] ^ h2[None, :]
        d = np.bitwise_count(a).min(axis=1)  # not min - need per-pair
        # per-pair distance: bitwise_count of XOR
        d = np.bitwise_count(a)  # (block, n)
        ii, jj = np.where(d <= maxdist)
        for a_, b_ in zip(ii, jj):
            gi, gj = i + a_, b_
            if gi < gj:
                pairs.append((gi, gj, int(d[a_, b_])))
    return pairs

# use combined: require both dhash dist <=4 and ahash dist <=6 OR sha differs
hp = near_pairs(dhash, dhash, 4, "dhash")
print(f"\nNEAR-DUP candidates (dHash Hamming <= 4): {len(hp)} pairs")
sha = [r["sha256"] for r in ok]
nd = []
for i, j, d in hp:
    if sha[i] == sha[j]:
        continue
    # 256-bit ahash distance as corroboration (4 words)
    ad = int(np.bitwise_count(ahash[i] ^ ahash[j]).sum())
    nd.append((i, j, d, ad))
print(f"  not byte-identical: {len(nd)}")
cross_nd = [t for t in nd if t[0] != t[1] and paths[t[0]].split('/')[0] != paths[t[1]].split('/')[0]]
print(f"  crossing class folders: {len(cross_nd)}")
print("  closest 15 by dhash dist (dhash, ahash256):")
for i, j, d, ad in sorted(nd, key=lambda x: (x[2], x[3]))[:15]:
    print(f"   d={d} a={ad}: {paths[i]}  <->  {paths[j]}")
print("  closest 15 CROSS-class:")
for i, j, d, ad in sorted(cross_nd, key=lambda x: (x[2], x[3]))[:15]:
    print(f"   d={d} a={ad}: {paths[i]}  <->  {paths[j]}")

# per-class exact duplicate stats
per_dup = collections.Counter()
for ps in dups.values():
    for p in ps: per_dup[p.split('/')[0]] += 1
print("\nfiles participating in exact-dup groups per class:", dict(per_dup))

# save flagged + dup lists
with open(os.path.join(HERE, "flagged.txt"), "w", encoding="utf-8") as f:
    for r, fl in flagged:
        f.write(f"{r['path']}\t{','.join(fl)}\tw={r['width']}\th={r['height']}\tlum={r['lum_mean']}/{r['lum_std']}\tsat={r['sat_mean']}\tedge={r['edge_energy']}\tdark={r['frac_dark']}\tbright={r['frac_bright']}\n")
with open(os.path.join(HERE, "exact_dup_groups.txt"), "w", encoding="utf-8") as f:
    for h, ps in dups.items():
        f.write(h + "\t" + "\t".join(ps) + "\n")
with open(os.path.join(HERE, "near_dup_pairs.txt"), "w", encoding="utf-8") as f:
    for i, j, d, ad in nd:
        f.write(f"{d}\t{ad}\t{paths[i]}\t{paths[j]}\n")
print("\nwrote flagged.txt, exact_dup_groups.txt, near_dup_pairs.txt")
