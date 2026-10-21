# V03-internal: does Zenodo v0.3 itself contain the same image under two different labels?
import zipfile, json, io, re, collections
import numpy as np
from PIL import Image

ZPATH = r"C:\Users\anous\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03.zip"
zf = zipfile.ZipFile(ZPATH)
items = [(i, i.filename.split("/")[0], i.filename.split("/")[1]) for i in zf.infolist()
         if "/" in i.filename and not i.filename.endswith("/") and i.filename.lower().endswith((".jpg", ".jpeg"))]
print("V03 images:", len(items))

def pack_bits(bits):
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def dhash_arr(g):
    a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return pack_bits((a[:, 1:] > a[:, :-1]).flatten())

def ahash16_arr(g):
    a = np.asarray(g.resize((16, 16), Image.LANCZOS), dtype=np.float64)
    return pack_bits((a > a.mean()).flatten())

def fs(f): return re.sub(r"^\d+\.\s*", "", f)

imgs, meta, dh, ah = [], [], [], []
for idx, (info, folder, base) in enumerate(items):
    with zf.open(info) as f:
        im = Image.open(io.BytesIO(f.read())); im.load()
    rgb = im.convert("RGB")
    imgs.append(rgb)
    meta.append({"base": base, "folder": folder, "label": fs(folder)})
    g = rgb.resize((128, 128), Image.LANCZOS).convert("L")
    dh.append(dhash_arr(g))
    ah.append(ahash16_arr(g))
    if idx % 300 == 0:
        print(f"  hashed {idx}/{len(items)}", flush=True)

dh = np.array(dh, dtype=np.uint64)
ah = np.frombuffer(b"".join(bytes.fromhex(format(v, "064x")) for v in ah), dtype=">u8").astype(np.uint64).reshape(-1, 4)
n = len(items)

def popc(x): return np.bitwise_count(x).astype(np.int16)

# candidate pairs
cands = []
for s in range(0, n, 128):
    e = min(s + 128, n)
    dd = popc(dh[s:e, None] ^ dh[None, :])
    aa = popc(ah[s:e, None, :] ^ ah[None, :, :]).sum(axis=2)
    rows, cols = np.where((dd <= 4) & (aa <= 8))
    for r, c in zip(rows, cols):
        i, j = s + int(r), int(c)
        if j > i:
            cands.append((i, j))
print("candidate pairs (dhash<=4):", len(cands))

def cmp256(a, b):
    A = np.asarray(a.resize((256, 256), Image.LANCZOS), np.float32)
    B = np.asarray(b.resize((256, 256), Image.LANCZOS), np.float32)
    mae = float(np.abs(A - B).mean())
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    return mae, r

same_pairs = []
for k, (i, j) in enumerate(cands):
    mae, r = cmp256(imgs[i], imgs[j])
    if mae < 8 and r > 0.95:
        same_pairs.append((i, j, mae, r))
    if k % 200 == 0:
        print(f"  pixel-checked {k}/{len(cands)}", flush=True)
print("pixel-SAME pairs inside V03:", len(same_pairs))

# union-find clusters
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
for i, j, mae, r in same_pairs:
    union(i, j)

clusters = collections.defaultdict(list)
for i in range(n):
    clusters[find(i)].append(i)
multi = {k: v for k, v in clusters.items() if len(v) > 1}
print("V03 same-image clusters >1:", len(multi), "files involved:", sum(len(v) for v in multi.values()))

cross = {k: v for k, v in multi.items() if len({meta[i]["label"] for i in v}) > 1}
print("CROSS-LABEL clusters inside V03:", len(cross), "files:", sum(len(v) for v in cross.values()))
for k, v in sorted(cross.items(), key=lambda kv: -len(kv[1])):
    labs = collections.Counter(meta[i]["label"] for i in v)
    members = ", ".join(f"{meta[i]['base']}[{meta[i]['label']}]" for i in sorted(v, key=lambda i: meta[i]["label"]))
    print(f"   cluster ({len(v)} files, labels={dict(labs)}): {members}")

same_lab = {k: v for k, v in multi.items() if len({meta[i]["label"] for i in v}) == 1}
print("same-label clusters:", len(same_lab), "files:", sum(len(v) for v in same_lab.values()))

out = [{"cluster": k, "members": [{"base": meta[i]["base"], "label": meta[i]["label"]} for i in v],
        "cross_label": len({meta[i]["label"] for i in v}) > 1} for k, v in multi.items()]
json.dump(out, open(r"audit_output\v03_internal_clusters.json", "w", encoding="utf-8"), indent=1)
print("wrote audit_output/v03_internal_clusters.json")
