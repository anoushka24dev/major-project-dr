# Pixel-level verification of V03<->dataset perceptual matches + semantic label audit.
import zipfile, json, io, re, collections
import numpy as np
from PIL import Image

ZPATH = r"C:\Users\anous\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03.zip"
DATA = r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I"
MATCHES = r"audit_output\v03_matches.json"
STATS = r"audit_output\image_stats.jsonl"

matches = json.load(open(MATCHES, encoding="utf-8"))
recs = {r["path"]: r for r in (json.loads(l) for l in open(STATS, encoding="utf-8"))}

# semantic label map V03 status -> dataset class keywords
SEM = {
    "No DR signs": ["11.Normal Fundus"],
    "Mild (or early) NPDR": ["3.Mild DR"],
    "Moderate NPDR": ["4.Moderate DR"],
    "Severe NPDR": ["5.Severe DR"],
    "Very Severe NPDR": ["13. Very Severe NPDR"],
    "PDR": ["6.Proliferate DR"],
    "Advanced PDR": ["12. Advanced PDR"],
}
def folder_status(f):
    return re.sub(r"^\d+\.\s*", "", f)

zf = zipfile.ZipFile(ZPATH)
znames = {i.filename.split("/")[1]: i for i in zf.infolist() if "/" in i.filename and not i.filename.endswith("/") and i.filename.lower().endswith((".jpg", ".jpeg"))}

def v03_img(base):
    with zf.open(znames[base]) as f:
        im = Image.open(io.BytesIO(f.read()))
        im.load()
    return im.convert("RGB")

def ds_img(path):
    im = Image.open(rf"{DATA}\{path.replace('/', chr(92))}")
    im.load()
    return im.convert("RGB")

def compare(a, b, size=(256, 256)):
    A = np.asarray(a.resize(size, Image.LANCZOS), dtype=np.float32)
    B = np.asarray(b.resize(size, Image.LANCZOS), dtype=np.float32)
    mae = float(np.abs(A - B).mean())
    x, y = A.ravel(), B.ravel()
    r = float(np.corrcoef(x, y)[0, 1])
    return mae, r

def sem_ok(vlab, dlab):
    return dlab in SEM.get(vlab, [])

# classify all matched pairs and pixel-verify the strong + conflicting ones
buckets = collections.Counter()
verified = []
conflicts_verified = []
for m in matches:
    vlab = folder_status(m["v03_folder"])
    dcls = m["ds"].split("/")[0]
    sem = sem_ok(vlab, dcls)
    strong = m["dhash_d"] <= 2 and m["ahash_d"] <= 2
    key = (sem, strong)
    buckets[key] += 1

print("match buckets (semantically_consistent, strong):", dict(buckets))

# pixel-verify: sample of consistent strong, consistent weak, all inconsistent
def sample(lst, k):
    return lst if len(lst) <= k else [lst[i] for i in np.linspace(0, len(lst) - 1, k).astype(int)]

cons_s = [m for m in matches if sem_ok(folder_status(m["v03_folder"]), m["ds"].split("/")[0]) and m["dhash_d"] <= 2 and m["ahash_d"] <= 2]
cons_w = [m for m in matches if sem_ok(folder_status(m["v03_folder"]), m["ds"].split("/")[0]) and not (m["dhash_d"] <= 2 and m["ahash_d"] <= 2)]
inc = [m for m in matches if not sem_ok(folder_status(m["v03_folder"]), m["ds"].split("/")[0])]
print("consistent strong:", len(cons_s), "consistent weak:", len(cons_w), "inconsistent:", len(inc))

def verify_group(name, group, k=40):
    out = []
    for m in sample(group, k):
        try:
            a = v03_img(m["v03"]); b = ds_img(m["ds"])
            mae, r = compare(a, b)
            out.append((mae, r, m))
        except Exception as e:
            print("ERR", m, e)
    if out:
        maes = [o[0] for o in out]; rs = [o[1] for o in out]
        same = sum(1 for mae, r, _ in out if mae < 8 and r > 0.95)
        print(f"{name}: n={len(out)} MAE med={np.median(maes):.2f} max={max(maes):.2f} | r med={np.median(rs):.4f} min={min(rs):.4f} | same-image(MAE<8,r>.95)={same}")
    return out

vs = verify_group("consistent-strong", cons_s, 60)
vw = verify_group("consistent-weak", cons_w, 60)
vi = verify_group("INCONSISTENT", inc, len(inc))

print("\n--- inconsistent pairs detail (pixel-checked) ---")
for mae, r, m in sorted(vi, key=lambda t: -t[0]):
    print(f"  MAE={mae:6.2f} r={r:.4f}  V03 {m['v03']} [{folder_status(m['v03_folder'])}]  vs  {m['ds']}  (d={m['dhash_d']},a={m['ahash_d']})")

# --- unmatched dataset .jpg files ---
matched_ds = {m["ds"] for m in matches}
jpgs = [p for p in recs if p.lower().endswith(".jpg")]
unm = sorted(set(jpgs) - matched_ds)
print("\ndataset .jpg with NO V03 perceptual match:", len(unm))
for p in unm:
    r = recs[p]
    print(f"   {p}  {r['width']}x{r['height']} sha={r['sha256'][:12]}")

# --- unmatched V03, with best dataset neighbor ---
# reload near-miss info quickly using stats arrays
import numpy as np
n = len(recs)
paths = list(recs.keys())
dh = np.frombuffer(b"".join(bytes.fromhex(recs[p]["dhash"].rjust(16, "0")) for p in paths), dtype=">u8").astype(np.uint64)
ah = np.frombuffer(b"".join(bytes.fromhex(recs[p]["ahash"].rjust(64, "0")) for p in paths), dtype=">u8").reshape(n, 4).astype(np.uint64)

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

matched_v03 = {m["v03"] for m in matches}
unmatched_bases = [b for b in znames if b not in matched_v03]
print("\nV03 unmatched:", len(unmatched_bases))
# for each, compute best dhash distance and pixel-verify the closest ones
close = []
for base in unmatched_bases:
    im = v03_img(base)
    g = im.convert("RGB").resize((128, 128), Image.LANCZOS).convert("L")
    d = dhash_arr(g)
    dd = np.bitwise_count(d ^ dh).astype(np.int16)
    j = int(dd.argmin())
    close.append((int(dd[j]), base, paths[j]))
close.sort()
print("unmatched-V03 best-dhash histogram:", dict(sorted(collections.Counter(min(c[0], 16) for c in close).items())))
print("\nverifying 12 closest unmatched V03 pairs pixel-wise:")
real = 0
for dist, base, p in close[:12]:
    a = v03_img(base); b = ds_img(p)
    mae, r = compare(a, b)
    flag = "SAME" if (mae < 8 and r > 0.95) else "diff"
    if flag == "SAME":
        real += 1
    print(f"  d={dist} V03 {base} [{folder_status(znames[base].filename.split('/')[0])}] vs {p}: MAE={mae:.2f} r={r:.4f} -> {flag}")
print(f"closest-12 that are actually same image: {real}")
print("\nfarthest unmatched V03 (truly new content) examples:")
for dist, base, p in close[-8:]:
    print(f"  d={dist} V03 {base} [{folder_status(znames[base].filename.split('/')[0])}] best-neighbor {p}")
