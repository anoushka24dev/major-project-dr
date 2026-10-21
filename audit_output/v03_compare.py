# Compare Zenodo V03 zip ("Dataset from fundus images for the study of diabetic
# retinopathy_V03.zip") against the audited Mendeley dataset.
import zipfile, hashlib, json, os, collections, re
import openpyxl

ZPATH = r"C:\Users\anous\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03.zip"
XLSX = r"C:\Users\anous\Downloads\zenodo_v03_annotations.xlsx"
DATA = r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I"

def sha(stream):
    h = hashlib.sha256()
    while True:
        b = stream.read(1 << 20)
        if not b:
            break
        h.update(b)
    return h.hexdigest()

# --- 1. V03 manifest (hash straight out of the zip, no extraction) ---
zf = zipfile.ZipFile(ZPATH)
v03 = {}            # filename -> dict
folder_counts = collections.Counter()
for i in zf.infolist():
    n = i.filename
    if n.endswith('/'):
        continue
    parts = n.split('/')
    if len(parts) != 2:
        print("UNEXPECTED PATH:", n)
        continue
    folder, base = parts
    if not base.lower().endswith(('.jpg', '.jpeg')):
        print("NON-IMAGE:", n)
        continue
    with zf.open(i) as f:
        d = sha(f)
    folder_counts[folder] += 1
    v03[base] = {"folder": folder, "sha": d, "size": i.file_size}
print("V03 images:", len(v03))
for k, v in sorted(folder_counts.items()):
    print("  folder", repr(k), v)

# --- 2. annotations xlsx ---
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
hdr = rows[0]
ann = {}
dup_ann = []
for r in rows[1:]:
    num, fmt, status = r
    key = f"{int(num)}.{fmt}" if isinstance(num, (int, float)) else f"{num}.{fmt}"
    if key in ann:
        dup_ann.append(key)
    ann[key] = status
print("xlsx header:", hdr, "rows:", len(ann), "dup rows:", len(dup_ann))

# folder label -> xlsx label mapping check
def folder_to_status(folder):
    # '1. No DR signs' -> 'No DR signs'
    return re.sub(r'^\d+\.\s*', '', folder)

mismatch, missing_ann, extra_ann = [], [], []
for base, m in v03.items():
    st = ann.get(base)
    if st is None:
        missing_ann.append(base)
        continue
    if str(st).strip() != folder_to_status(m["folder"]):
        mismatch.append((base, m["folder"], st))
for key in ann:
    if key not in v03:
        extra_ann.append(key)
print("folder-vs-xlsx label mismatches:", len(mismatch), mismatch[:10])
print("images missing from xlsx:", len(missing_ann), sorted(missing_ann, key=lambda s: int(re.sub(r'\D', '', s) or 0))[:20])
print("xlsx rows with no image:", len(extra_ann), extra_ann[:20])

# xlsx status counts
print("xlsx status counts:", dict(collections.Counter(str(v) for v in ann.values())))

# --- 3. hash the audited dataset files: all 22,453 ---
ds = {}   # sha -> list of relpaths
ds_by_file = {}  # (folder, base) -> sha
n = 0
for cls in sorted(os.listdir(DATA)):
    p = os.path.join(DATA, cls)
    if not os.path.isdir(p):
        continue
    for base in sorted(os.listdir(p)):
        fp = os.path.join(p, base)
        with open(fp, 'rb') as f:
            d = sha(f)
        ds.setdefault(d, []).append(cls + '/' + base)
        ds_by_file[(cls, base)] = d
        n += 1
print("dataset files hashed:", n, "unique shas:", len(ds))

# --- 4. V03 vs dataset ---
v03_by_sha = {m["sha"]: base for base, m in v03.items()}
matched, matched_jpg, matched_jpeg = 0, 0, 0
label_conflicts = []
match_rows = []
for base, m in v03.items():
    hits = ds.get(m["sha"], [])
    if hits:
        matched += 1
        for h in hits:
            cls = h.split('/')[0]
            ds_base = h.split('/', 1)[1]
            v03_cls = m["folder"]
            same = folder_to_status(v03_cls).lower().replace(' ', '') in cls.lower().replace(' ', '') or cls.lower().replace(' ', '') in folder_to_status(v03_cls).lower().replace(' ', '')
            if ds_base.lower().endswith('.jpg'):
                matched_jpg += 1
            else:
                matched_jpeg += 1
            match_rows.append({"v03": base, "v03_folder": v03_cls, "ds": h})
            if not same:
                label_conflicts.append((base, v03_cls, h))
print("V03 images found in dataset:", matched, "of", len(v03))
print("  matches on .jpg files:", matched_jpg, " on .jpeg files:", matched_jpeg)
print("label conflicts V03 vs dataset folder:", len(label_conflicts))
for c in label_conflicts[:30]:
    print("   ", c)

# how many dataset .jpg files are NOT in V03
jpg_files = [(c, b) for (c, b) in ds_by_file if b.lower().endswith('.jpg')]
jpg_total = len(jpg_files)
jpg_in_v03 = sum(1 for (c, b) in jpg_files if ds_by_file[(c, b)] in v03_by_sha)
print("dataset .jpg files:", jpg_total, " matched to V03 by sha:", jpg_in_v03)

# unmatched dataset .jpg (were they in old v0.2 zip only?)
unmatched_jpg = [(c, b) for (c, b) in jpg_files if ds_by_file[(c, b)] not in v03_by_sha]
print("dataset .jpg NOT in V03:", len(unmatched_jpg))
cc = collections.Counter(c for c, b in unmatched_jpg)
for k, v in cc.most_common():
    print("   ", repr(k), v)

# V03 images not present in the dataset at all
v03_missing_from_ds = [base for base, m in v03.items() if m["sha"] not in ds]
print("V03 images absent from dataset:", len(v03_missing_from_ds))
cc = collections.Counter(v03[b]["folder"] for b in v03_missing_from_ds)
for k, v in cc.most_common():
    print("   ", repr(k), v)

# --- 5. exact duplicates inside V03 ---
by_sha = collections.defaultdict(list)
for base, m in v03.items():
    by_sha[m["sha"]].append(base)
dupgroups = {s: bs for s, bs in by_sha.items() if len(bs) > 1}
print("V03 exact-dup groups:", len(dupgroups), "files involved:", sum(len(v) for v in dupgroups.values()))
for s, bs in list(dupgroups.items())[:20]:
    print("   ", [(b, folder_to_status(v03[b]["folder"])) for b in bs])

# cross-label dups inside V03
cross = 0
for s, bs in dupgroups.items():
    labs = {folder_to_status(v03[b]["folder"]) for b in bs}
    if len(labs) > 1:
        cross += 1
        print("   CROSS-LABEL:", [(b, folder_to_status(v03[b]["folder"])) for b in bs])
print("V03 cross-label dup groups:", cross)

with open(r"audit_output\v03_manifest.json", "w") as f:
    json.dump({"v03": v03, "ann": ann}, f)
print("wrote audit_output/v03_manifest.json")
