# Final quantification: pixel-verify ALL V03<->dataset pairs, exact coverage + label conflicts.
import zipfile, json, io, re, collections
import numpy as np
from PIL import Image

ZPATH = r"C:\Users\anous\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03.zip"
DATA = r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I"
STATS = r"audit_output\image_stats.jsonl"

matches = json.load(open(r"audit_output\v03_matches.json", encoding="utf-8"))
recs = {r["path"]: r for r in (json.loads(l) for l in open(STATS, encoding="utf-8"))}

SEM = {"No DR signs": "11.Normal Fundus", "Mild (or early) NPDR": "3.Mild DR",
       "Moderate NPDR": "4.Moderate DR", "Severe NPDR": "5.Severe DR",
       "Very Severe NPDR": "13. Very Severe NPDR", "PDR": "6.Proliferate DR",
       "Advanced PDR": "12. Advanced PDR"}
def fstatus(f): return re.sub(r"^\d+\.\s*", "", f)

zf = zipfile.ZipFile(ZPATH)
znames = {i.filename.split("/")[1]: i for i in zf.infolist()
          if "/" in i.filename and not i.filename.endswith("/") and i.filename.lower().endswith((".jpg", ".jpeg"))}

def load_v03(base):
    with zf.open(znames[base]) as f:
        im = Image.open(io.BytesIO(f.read())); im.load()
    return im.convert("RGB")

def load_ds(path):
    im = Image.open(rf"{DATA}\{path.replace('/', chr(92))}"); im.load()
    return im.convert("RGB")

def cmp256(a, b):
    A = np.asarray(a.resize((256, 256), Image.LANCZOS), dtype=np.float32)
    B = np.asarray(b.resize((256, 256), Image.LANCZOS), dtype=np.float32)
    mae = float(np.abs(A - B).mean())
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    return mae, r

SAME_MAE, SAME_R = 8.0, 0.95
pairs = []
cache_v, cache_d = {}, {}
for k, m in enumerate(matches):
    vb, dp = m["v03"], m["ds"]
    if vb not in cache_v: cache_v[vb] = load_v03(vb)
    if dp not in cache_d: cache_d[dp] = load_ds(dp)
    mae, r = cmp256(cache_v[vb], cache_d[dp])
    m["mae"], m["r"] = mae, r
    m["same"] = mae < SAME_MAE and r > SAME_R
    pairs.append(m)
    if k % 100 == 0:
        print(f"  verified {k}/{len(matches)}", flush=True)

n_same = sum(1 for m in pairs if m["same"])
print(f"pairs={len(pairs)} pixel-SAME={n_same} FALSE-POS={len(pairs)-n_same}")

same = [m for m in pairs if m["same"]]
# coverage of dataset .jpg by SAME pairs
cov = {m["ds"] for m in same}
print(f"dataset .jpg pixel-covered by V03: {len(cov)} of 757 ; uncovered: {757-len(cov)}")
alljpg = {p for p in recs if p.lower().endswith(".jpg")}
uncov = sorted(alljpg - cov)
print("uncovered dataset .jpg:", len(uncov))

# V03 side coverage
covv = {m["v03"] for m in same}
print(f"V03 images pixel-matched into dataset: {len(covv)} of 1437 ; new/absent from dataset: {1437-len(covv)}")
new_by_folder = collections.Counter(fstatus(znames[b].filename.split("/")[0]) for b in znames if b not in covv)
print("V03-absent-from-dataset by class:", dict(new_by_folder))

# label conflicts among SAME pairs
conf = [m for m in same if SEM[fstatus(m["v03_folder"])] != m["ds"].split("/")[0]]
print(f"\nSAME pairs with label conflict: {len(conf)}")
conf_ds = {m["ds"] for m in conf}
conf_v = {m["v03"] for m in conf}
print(f"distinct dataset files involved: {len(conf_ds)} ; distinct V03 images: {len(conf_v)}")
cc = collections.Counter((fstatus(m["v03_folder"]), m["ds"].split("/")[0]) for m in conf)
print("conflict pairs (V03 label -> dataset folder):")
for k, v in cc.most_common():
    print(f"   {k[0]:22s} -> {k[1]:24s} : {v}")

# does each conflict image also sit under 2 labels INSIDE the dataset? (check dataset-internal cross-label)
# build dataset same-image clusters among the involved files using audit's near-dup pairs
print("\nconflict files that are ALSO cross-label duplicated inside the dataset (from audit pairs):")
ds_cross = set()
try:
    for line in open(r"audit_output\strong_cross_class_pairs.txt", encoding="utf-8"):
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            ds_cross.update([parts[2], parts[3]])
except FileNotFoundError:
    pass
overlap = conf_ds & ds_cross
print(f"   conflict dataset files: {len(conf_ds)}, already known cross-label in-dataset: {len(overlap)}")
print("   new conflict files not in audit's cross-class list:", len(conf_ds - ds_cross))
for p in sorted(conf_ds - ds_cross)[:25]:
    print("      ", p)

# V03-internal: same underlying image under 2 V03 labels? cluster SAME pairs by v03 image
v03_labels = collections.defaultdict(set)
for m in same:
    v03_labels[m["v03"]].add(fstatus(m["v03_folder"]))
    # also the dataset file's V03 twin label vs its own: handled above
# dataset file mapped from multiple V03 labels:
ds_labels = collections.defaultdict(set)
for m in same:
    ds_labels[m["ds"]].add(fstatus(m["v03_folder"]))
multi = {d: ls for d, ls in ds_labels.items() if len(ls) > 1}
print(f"\ndataset files matched by >1 distinct V03 label: {len(multi)}")
for d, ls in list(multi.items())[:15]:
    print("   ", d, sorted(ls))

# strongest conflict examples for report
print("\nsame-image label conflicts pixel-strength (lowest MAE):")
for m in sorted(conf, key=lambda x: x["mae"])[:25]:
    print(f"   MAE={m['mae']:.3f} r={m['r']:.4f}  V03 {m['v03']} [{fstatus(m['v03_folder'])}] == {m['ds']}")

json.dump(pairs, open(r"audit_output\v03_pairs_verified.json", "w", encoding="utf-8"), indent=1)
print("\nwrote audit_output/v03_pairs_verified.json")

# ---- the 14 uncovered: best V03 candidate, pixel-checked ----
print("\n--- 14 dataset .jpg not covered: best V03 candidates ---")
import numpy as np
def pack_bits(bits):
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h
def dhash_arr(g):
    a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return pack_bits((a[:, 1:] > a[:, :-1]).flatten())
vbases = list(znames)
vd = []
for idx, b in enumerate(vbases):
    if b in cache_v:
        img = cache_v[b]
    else:
        img = load_v03(b)
        cache_v[b] = img
    g = img.resize((128, 128), Image.LANCZOS).convert("L")
    vd.append(dhash_arr(g))
    if idx % 300 == 0:
        print(f"  v03 dhash {idx}/{len(vbases)}", flush=True)
vd = np.array(vd, dtype=np.uint64)
for p in uncov:
    g = load_ds(p).resize((128, 128), Image.LANCZOS).convert("L")
    d = dhash_arr(g)
    dd = np.bitwise_count(d ^ vd).astype(np.int16)
    j = int(dd.argmin())
    mae, r = cmp256(load_ds(p), cache_v[vbases[j]])
    print(f"   {p}: best V03={vbases[j]} [{fstatus(znames[vbases[j]].filename.split('/')[0])}] d={int(dd[j])} MAE={mae:.2f} r={r:.4f}")
