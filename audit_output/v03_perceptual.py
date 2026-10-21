# Perceptual (dhash+ahash) match of Zenodo V03 images against the audited dataset.
import zipfile, json, io, os, collections
import numpy as np
from PIL import Image

ZPATH = r"C:\Users\anous\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03.zip"
STATS = r"audit_output\image_stats.jsonl"

def pack_bits(bits):
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def dhash_arr(gray_small_arr):
    a = np.asarray(gray_small_arr.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return pack_bits((a[:, 1:] > a[:, :-1]).flatten())

def ahash16_arr(gray_small_arr):
    a = np.asarray(gray_small_arr.resize((16, 16), Image.LANCZOS), dtype=np.float64)
    return pack_bits((a > a.mean()).flatten())

# --- dataset stats ---
recs = [json.loads(l) for l in open(STATS, encoding="utf-8")]
n = len(recs)
dh = np.frombuffer(b"".join(bytes.fromhex(r["dhash"].rjust(16, "0")) for r in recs), dtype=">u8").astype(np.uint64)
ah = np.frombuffer(b"".join(bytes.fromhex(r["ahash"].rjust(64, "0")) for r in recs), dtype=">u8").reshape(n, 4).astype(np.uint64)
paths = [r["path"] for r in recs]
is_jpg = np.array([p.lower().endswith(".jpg") for p in paths])
print("dataset:", n, " (.jpg:", int(is_jpg.sum()), ")")

def popc(x):
    return np.bitwise_count(x).astype(np.int16)

# --- V03 images ---
zf = zipfile.ZipFile(ZPATH)
v03 = []
for i in zf.infolist():
    if i.filename.endswith('/'):
        continue
    if '/' not in i.filename:
        continue
    folder, base = i.filename.split('/')
    if not base.lower().endswith(('.jpg', '.jpeg')):
        continue
    with zf.open(i) as f:
        data = f.read()
    im = Image.open(io.BytesIO(data))
    im.load()
    rgb = im.convert("RGB")
    small = rgb.resize((128, 128), Image.LANCZOS)
    gray = small.convert("L")
    v03.append({
        "base": base, "folder": folder, "w": im.size[0], "h": im.size[1],
        "d": dhash_arr(gray), "a": ahash16_arr(gray), "bytes": len(data),
    })
print("V03 images:", len(v03))

vd = np.array([v["d"] for v in v03], dtype=np.uint64)
va = np.frombuffer(b"".join(bytes.fromhex(format(v["a"], "064x")) for v in v03),
                   dtype=">u8").astype(np.uint64).reshape(-1, 4)

# pairwise: process V03 in chunks against all dataset rows
best_d = np.full(len(v03), 99, dtype=np.int16)
best_i = np.full(len(v03), -1, dtype=np.int32)
CONF_DHASH, CONF_AHASH = 4, 6
matches = []          # (v03_idx, ds_idx, d_dist, a_dist)
for s in range(0, len(v03), 64):
    e = min(s + 64, len(v03))
    dd = popc(vd[s:e, None] ^ dh[None, :])                 # (chunk, n)
    aa = popc(va[s:e, None, :] ^ ah[None, :, :]).sum(axis=2)
    bd = dd.min(axis=1)
    bi = dd.argmin(axis=1)
    best_d[s:e] = bd
    best_i[s:e] = bi
    # candidates with dhash<=8 then confirm with ahash
    rows, cols = np.where(dd <= 8)
    for r, c in zip(rows, cols):
        if aa[r, c] <= CONF_AHASH and dd[r, c] <= CONF_DHASH:
            matches.append((s + r, int(c), int(dd[r, c]), int(aa[r, c])))

print("confirmed matches (dhash<=%d & ahash<=%d): %d" % (CONF_DHASH, CONF_AHASH, len(matches)))

matched_v03 = {m[0] for m in matches}
jpg_idx = [i for i in range(n) if is_jpg[i]]
jpeg_idx = [i for i in range(n) if not is_jpg[i]]
m_to_jpg = sum(1 for m in matches if is_jpg[m[1]])
m_to_jpeg = sum(1 for m in matches if not is_jpg[m[1]])
print("  matches -> dataset .jpg :", m_to_jpg, " (distinct v03:", len({m[0] for m in matches if is_jpg[m[1]]}), ")")
print("  matches -> dataset .jpeg:", m_to_jpeg, " (distinct v03:", len({m[0] for m in matches if not is_jpg[m[1]]}), ")")

# dataset .jpg files covered by V03
jpg_matched_ds = {m[1] for m in matches if is_jpg[m[1]]}
print("dataset .jpg files matched by some V03 image:", len(jpg_matched_ds), "of 757")

def folder_status(f):
    import re
    return re.sub(r"^\d+\.\s*", "", f)

# label consistency for matches into .jpg (folder of V03 vs dataset class folder)
conflicts = []
for vi, di, dd_, aa_ in matches:
    if not is_jpg[di]:
        continue
    vlab = folder_status(v03[vi]["folder"])
    dlab = paths[di].split("/")[0]
    norm = lambda s: s.lower().replace(" ", "").replace("(", "").replace(")", "")
    ok = norm(vlab) in norm(dlab) or norm(dlab) in norm(vlab)
    # special: V03 'No DR signs' vs dataset 'Normal Fundus' (treat as same meaning?)
    if not ok:
        conflicts.append((v03[vi]["base"], vlab, paths[di], dd_, aa_))
print("label conflicts on .jpg matches:", len(conflicts))
for c in conflicts[:40]:
    print("   ", c)

# jpeg matches: which dataset classes?
cc = collections.Counter(paths[di].split("/")[0] for vi, di, d_, a_ in matches if not is_jpg[di])
print("jpeg-match dataset classes:", dict(cc))
cc = collections.Counter(folder_status(v03[vi]["folder"]) for vi, di, d_, a_ in matches if not is_jpg[di])
print("jpeg-match V03 folders:", dict(cc))

# best-distance distribution for unmatched V03
un = [i for i in range(len(v03)) if i not in matched_v03]
print("V03 images with NO match:", len(un))
hist = collections.Counter(int(min(best_d[i], 32)) for i in un)
print("best-dhash dist histogram (unmatched, capped 32):", dict(sorted(hist.items())))
# examples of near misses
near = sorted(un, key=lambda i: int(best_d[i]))[:10]
for i in near:
    j = int(best_i[i])
    print("   near miss:", v03[i]["base"], folder_status(v03[i]["folder"]), "vs", paths[j], "d=", int(best_d[i]))

# save manifest of matches
out = [{"v03": v03[vi]["base"], "v03_folder": v03[vi]["folder"], "ds": paths[di],
        "dhash_d": dd_, "ahash_d": aa_} for vi, di, dd_, aa_ in matches]
with open(r"audit_output\v03_matches.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("wrote audit_output/v03_matches.json")

# V03 dims summary
sizes = collections.Counter((v["w"], v["h"]) for v in v03)
print("V03 distinct sizes:", len(sizes), "top:", sizes.most_common(8))
