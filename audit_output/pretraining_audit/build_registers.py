"""Build pre-training audit registers from EXISTING audit artifacts (read-only).

Inputs (all in audit_output/, produced by the earlier audit):
  image_stats.jsonl, exact_dup_groups.txt, near_dup_pairs.txt,
  strong_cross_class_pairs.txt, flagged.txt, v03_pairs_verified.json
plus a re-hash of the Zenodo v0.2 zip to confirm .jpg provenance per file.

Outputs (in audit_output/pretraining_audit/):
  LABEL_CONFLICT_REGISTER.csv   one row per duplicate pair edge
  QUALITY_REVIEW_REGISTER.csv   one row per quality-flagged image
  PROVENANCE_MANIFEST.csv       one row per dataset image (22,453)
  build_summary.json            counts used by the written reports
"""
import os, re, json, csv, hashlib, zipfile, collections

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)                      # audit_output/
DATA = r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I"
V02ZIP = r"C:\Users\anous\Downloads\Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02.zip"

# ---------------------------------------------------------------- load stats
recs = [json.loads(l) for l in open(os.path.join(AUD, "image_stats.jsonl"), encoding="utf-8")]
by_path = {r["path"]: r for r in recs}
print("stats rows:", len(recs))

# ------------------------------------------------- filename family / grouping
PREF = re.compile(r"^(.+?)(\d+)_(left|right)_(\d+)_(\d+)\.jpeg$")
PLAIN = re.compile(r"^(\d+)(?:_(left|right))?\.jpeg$")

def family(path):
    folder, base = path.split("/", 1)
    if base.lower().endswith(".jpg"):
        return "zenodo_v02"
    m = PREF.match(base)
    if m:
        return "prefixed"
    m = PLAIN.match(base)
    if m:
        return "plain"
    return "unknown"

def group_key(path):
    folder, base = path.split("/", 1)
    fam = family(path)
    if fam == "zenodo_v02":
        return "NONE"
    if fam == "prefixed":
        m = PREF.match(base)
        return f"prefixed:{folder}|ID={m.group(2)}"
    if fam == "plain":
        m = PLAIN.match(base)
        return f"plain:{folder}|ID={m.group(1)}"
    return "UNKNOWN"

fam_counts = collections.Counter(family(p) for p in by_path)
print("families:", dict(fam_counts))

# ------------------------------------------- Zenodo v0.2 per-file sha confirm
zf = zipfile.ZipFile(V02ZIP)
def sha_stream(f):
    h = hashlib.sha256()
    while True:
        b = f.read(1 << 20)
        if not b:
            break
        h.update(b)
    return h.hexdigest()
v02_sha = {}
for i in zf.infolist():
    if i.filename.endswith("/") or "/" not in i.filename:
        continue
    with zf.open(i) as f:
        v02_sha[sha_stream(f)] = i.filename
zen_confirmed = 0
zen_by_sha = {}
for p, r in by_path.items():
    if family(p) == "zenodo_v02":
        hit = v02_sha.get(r["sha256"])
        if hit:
            zen_confirmed += 1
            zen_by_sha[p] = hit
print(f"zenodo .jpg confirmed byte-identical to v0.2 zip: {zen_confirmed}/757")

# --------------------------------------------------- duplicate edges (union)
exact_groups = []   # (sha, [paths])
for line in open(os.path.join(AUD, "exact_dup_groups.txt"), encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    exact_groups.append((parts[0], parts[1:]))
print("exact groups:", len(exact_groups), "files:", sum(len(p) for _, p in exact_groups))

strong = []   # (dhash_d, ahash_d, pa, pb)
for line in open(os.path.join(AUD, "near_dup_pairs.txt"), encoding="utf-8"):
    d, a, pa, pb = line.rstrip("\n").split("\t")
    d, a = int(d), int(a)
    if d <= 4 and a <= 6:
        strong.append((d, a, pa, pb))
print("strong near-dup pairs (d<=4 & a<=6):", len(strong))
strong_files = {p for _, _, pa, pb in strong for p in (pa, pb)}
print("strong near-dup unique files:", len(strong_files))

# Extra pair pixel-verified during the original audit but below the ahash
# corroboration threshold; kept in the register with explicit weaker evidence.
EXTRA = [(1, 12, "10.Glaucoma/Glaucoma2041_left_0_8337.jpeg",
          "2.Wet AMD/Wet AMD2041_left_0_7926.jpeg",
          "dhash=1 but ahash=12 (outside audit corroboration threshold); "
          "original audit pixel-verified r=0.923 (colour-shifted copy)")]
cross_strong = [t for t in strong if t[2].split("/")[0] != t[3].split("/")[0]]
print("cross-class strong pairs:", len(cross_strong),
      "files:", len({p for _, _, a, b in cross_strong for p in (a, b)}))

# same-patient vs cross-patient breakdown of strong pairs
same_patient = cross_patient = unknown_patient = within_class = 0
for d, a, pa, pb in strong:
    if pa.split("/")[0] == pb.split("/")[0]:
        within_class += 1
    ka, kb = group_key(pa), group_key(pb)
    if ka == "NONE" or kb == "NONE":
        unknown_patient += 1
    elif ka == kb:
        same_patient += 1
    else:
        cross_patient += 1
print(f"strong pairs: within_class={within_class} same_patient={same_patient} "
      f"cross_patient={cross_patient} unknown(zenodo)={unknown_patient}")

# verify the user-cited example pairs
examples = [("4.Moderate DR/243.jpg", "6.Proliferate DR/577.jpg"),
            ("4.Moderate DR/243.jpg", "6.Proliferate DR/579.jpg"),
            ("4.Moderate DR/243.jpg", "6.Proliferate DR/580.jpg"),
            ("11.Normal Fundus/110.jpg", "12. Advanced PDR/743.jpg"),
            ("13. Very Severe NPDR/547.jpg", "5.Severe DR/443.jpg")]
strong_set = {(pa, pb) for _, _, pa, pb in strong} | {(pb, pa) for _, _, pa, pb in strong}
scc_set = set()
for line in open(os.path.join(AUD, "strong_cross_class_pairs.txt"), encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 4:
        scc_set.add((parts[2], parts[3]))
scc_set |= {(b, a) for a, b in scc_set}
for a, b in examples:
    print(f"example {a} <-> {b}: in strong tier={((a, b) in strong_set)}  in cross-class file={((a, b) in scc_set)}")

# union-find over exact members + strong edges
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra
for _, paths in exact_groups:
    for p in paths[1:]:
        union(paths[0], p)
for _, _, pa, pb in strong:
    union(pa, pb)
for _, _, pa, pb, _ev in EXTRA:
    union(pa, pb)
comps = collections.defaultdict(set)
for p in parent:
    comps[find(p)].add(p)
comps = {k: v for k, v in comps.items() if len(v) > 1}
# deterministic group ids ordered by smallest path in component
order = sorted(comps.values(), key=lambda s: sorted(s)[0])
gid = {p: f"DUP-G{i+1:03d}" for i, s in enumerate(order) for p in s}
print("duplicate components:", len(order), "files:", len(gid))
cross_label_comps = [s for s in order if len({p.split("/")[0] for p in s}) > 1]
print("cross-label components:", len(cross_label_comps),
      "files:", len({p for s in cross_label_comps for p in s}))

# ------------------------------------------------ v03 evidence per dataset file
v03_labels = collections.defaultdict(set)
v03_pairs = json.load(open(os.path.join(AUD, "v03_pairs_verified.json"), encoding="utf-8"))
def fstatus(f):
    return re.sub(r"^\d+\.\s*", "", f)
for m in v03_pairs:
    if m.get("same") and m["mae"] < 1.0 and m["r"] > 0.99:
        v03_labels[m["ds"]].add(fstatus(m["v03_folder"]))
# two files found same-image to v0.3 in the 14-uncovered follow-up check
v03_labels["11.Normal Fundus/102.jpg"].add("No DR signs")
v03_labels["13. Very Severe NPDR/537.jpg"].add("Very Severe NPDR")
# broader same-image set (audit criterion MAE<8, r>0.95) for manifest v0.3 status
v03_same_ds = {m["ds"] for m in v03_pairs if m.get("same")}
v03_same_ds |= {"11.Normal Fundus/102.jpg", "13. Very Severe NPDR/537.jpg"}
# 12 dataset files verified absent from v0.3 (norm-corr check) + 2 found late
absent_v03 = {"13. Very Severe NPDR/449.jpg", "13. Very Severe NPDR/450.jpg",
              "13. Very Severe NPDR/451.jpg", "13. Very Severe NPDR/452.jpg",
              "13. Very Severe NPDR/453.jpg", "13. Very Severe NPDR/454.jpg",
              "13. Very Severe NPDR/455.jpg", "13. Very Severe NPDR/456.jpg",
              "13. Very Severe NPDR/457.jpg", "13. Very Severe NPDR/458.jpg",
              "5.Severe DR/445.jpg", "5.Severe DR/446.jpg"}

# ------------------------------------------------------ quality flags
qflags = {}
for line in open(os.path.join(AUD, "flagged.txt"), encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    path = parts[0]
    flags = parts[1].split(",")
    metrics = dict(kv.split("=", 1) for kv in parts[2:] if "=" in kv)
    qflags[path] = (flags, metrics)
print("quality-flagged files:", len(qflags))
qfc = collections.Counter(f for fl, _ in qflags.values() for f in fl)
print("flag tallies:", dict(qfc))

# ------------------------------------------------------ LABEL_CONFLICT REGISTER
def v03_str(p):
    return "|".join(sorted(v03_labels[p])) if p in v03_labels else ""

rows = []
# exact-group edges: enumerate all unordered pairs within each exact group
for sha, paths in exact_groups:
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pa, pb = paths[i], paths[j]
            la, lb = pa.split("/")[0], pb.split("/")[0]
            agree = la == lb
            rows.append(dict(
                group_id=gid[pa], relationship="exact_duplicate",
                evidence=f"sha256 identical ({sha[:16]}...)",
                path_a=pa, label_a=la, path_b=pb, label_b=lb,
                labels_agree="yes" if agree else "NO",
                patient_group_a=group_key(pa), patient_group_b=group_key(pb),
                source_family_a=family(pa), source_family_b=family(pb),
                v03_label_evidence_a=v03_str(pa), v03_label_evidence_b=v03_str(pb),
                visual_review_status="not_performed (numeric + sha evidence only)",
                proposed_disposition=("keep_together_same_split_group" if agree
                                      else "QUARANTINE_both_pending_adjudication"),
                disposition_reason=("byte-identical copies, same label; redundant copy - "
                                    "candidate for later single-copy exclusion, needs approval"
                                    if agree else
                                    "byte-identical pixels carry contradictory labels; "
                                    "unresolvable without specialist/source-author review")))
for d, a, pa, pb in strong:
    la, lb = pa.split("/")[0], pb.split("/")[0]
    agree = la == lb
    ka, kb = group_key(pa), group_key(pb)
    if ka == "NONE" or kb == "NONE":
        pat = "unknown (zenodo: no patient linkage)"
    else:
        pat = "same" if ka == kb else "different"
    rows.append(dict(
        group_id=gid[pa], relationship="near_duplicate",
        evidence=f"dhash_hamming={d}, ahash256_hamming={a} (audit thresholds <=4/<=6)",
        path_a=pa, label_a=la, path_b=pb, label_b=lb,
        labels_agree="yes" if agree else "NO",
        patient_group_a=ka, patient_group_b=kb,
        source_family_a=family(pa), source_family_b=family(pb),
        v03_label_evidence_a=v03_str(pa), v03_label_evidence_b=v03_str(pb),
        visual_review_status="not_performed (perceptual-hash evidence; sample pixel-verified in audit)",
        proposed_disposition=("keep_together_same_split_group" if agree
                              else "QUARANTINE_both_pending_adjudication"),
        disposition_reason=("same underlying image, same label; must not straddle splits"
                            if agree else
                            "same underlying image, contradictory labels; quarantine both "
                            "copies from affected experiments until adjudicated")))
for d, a, pa, pb, ev_note in EXTRA:
    la, lb = pa.split("/")[0], pb.split("/")[0]
    rows.append(dict(
        group_id=gid[pa], relationship="audit_noted_weak_pair",
        evidence=f"dhash_hamming={d}, ahash256_hamming={a}; {ev_note}",
        path_a=pa, label_a=la, path_b=pb, label_b=lb,
        labels_agree="NO",
        patient_group_a=group_key(pa), patient_group_b=group_key(pb),
        source_family_a=family(pa), source_family_b=family(pb),
        v03_label_evidence_a=v03_str(pa), v03_label_evidence_b=v03_str(pb),
        visual_review_status="numeric only (audit correlation 0.923; not visually confirmed)",
        proposed_disposition="QUARANTINE_both_pending_adjudication",
        disposition_reason="cross-label pair below automatic threshold; human/specialist "
                           "confirmation required before any use in affected tasks"))
reg_path = os.path.join(HERE, "LABEL_CONFLICT_REGISTER.csv")
with open(reg_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
n_pairs = len(rows)
n_files_reg = len(gid)
n_quar = sum(1 for r in rows if r["labels_agree"] == "NO")
conflict_files = sorted({r["path_a"] for r in rows if r["labels_agree"] == "NO"} |
                        {r["path_b"] for r in rows if r["labels_agree"] == "NO"})
print(f"register: {n_pairs} pair rows, {n_files_reg} unique files, "
      f"{len(order)} components, conflict rows={n_quar}, conflict files={len(conflict_files)}")

# ------------------------------------------------------ QUALITY register
PRI = {"near_uniform_low_std": "P1 - near-blank; strongest exclusion candidate after visual confirmation",
       "near_grayscale": "P2 - desaturated; confirm whether genuine B/W or corruption",
       "low_texture": "P2 - low-structure centre; confirm retinal content present",
       "very_dark": "P3 - dark exposure; often genuine fundus, confirm usability",
       "mostly_black": "P3 - large black area; confirm FOV coverage",
       "mostly_white": "P3 - large white area; confirm artifact vs exudation",
       "black_border": "P4 - framing anomaly; usually benign, confirm",
       "white_border": "P4 - framing anomaly; usually benign, confirm",
       "tiny_dims": "P4 - unusually small; confirm usable resolution",
       "very_bright": "P3 - overexposure; confirm usability"}
qrows = []
for path in sorted(qflags):
    fl, m = qflags[path]
    pri = min((PRI[f].split(" ")[0] for f in fl), key=lambda s: s)
    qrows.append(dict(
        image_path=path, source_family=family(path), original_label=path.split("/")[0],
        quality_flags="|".join(fl), flag_count=len(fl), flags_overlap="yes" if len(fl) > 1 else "no",
        numerical_evidence=(f"w={m.get('w')} h={m.get('h')} lum_mean={m.get('lum')} "
                            f"sat_mean={m.get('sat')} edge_energy={m.get('edge')} "
                            f"frac_dark={m.get('dark')} frac_bright={m.get('bright')}"),
        visual_review_performed="no",
        review_outcome="",
        proposed_status="quarantine_for_review",
        review_priority=pri.split(" ")[0],
        notes=pri))
qpath = os.path.join(HERE, "QUALITY_REVIEW_REGISTER.csv")
with open(qpath, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(qrows[0].keys()))
    w.writeheader()
    w.writerows(qrows)
print("quality register rows:", len(qrows))

# ------------------------------------------------------ PROVENANCE manifest
mpath = os.path.join(HERE, "PROVENANCE_MANIFEST.csv")
FAM_EV = {
    "zenodo_v02": ("confirmed",
                   "sha256 byte-identical to Castillo Benitez Zenodo v0.2 zip (record 4647952)"),
    "prefixed": ("likely",
                 "Kaggle 'retinal-fundus-images'-style class-prefixed filenames; "
                 "count-match evidence only, no source documentation"),
    "plain": ("likely",
              "Kaggle EyePACS-style <id>_left/right.jpeg filenames; count-match evidence only"),
    "unknown": ("unknown", "filename matched no known pattern"),
}
mrows = []
for p in sorted(by_path):
    r = by_path[p]
    fam = family(p)
    status, ev = FAM_EV[fam]
    if fam == "zenodo_v02":
        ev = ev if p in zen_by_sha else "sha256 NOT found in v0.2 zip - INVESTIGATE"
    mrows.append(dict(
        image_path=p, original_label=p.split("/")[0], source_family=fam,
        source_status=status, source_evidence=ev,
        resolution=f"{r['width']}x{r['height']}",
        patient_group_key=group_key(p),
        duplicate_group_id=gid.get(p, ""),
        in_label_conflict_register="yes" if p in gid else "no",
        quality_flags="|".join(qflags[p][0]) if p in qflags else "",
        v03_status=("absent_in_v0.3 (dropped by source authors)" if p in absent_v03
                    else ("identical_visual_twin_in_v0.3" if p in v03_same_ds else
                          ("not_applicable_non_zenodo" if fam != "zenodo_v02" else
                            "not_matched_in_v0.3 (review)"))),
        sha256=r["sha256"]))
with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
    w.writeheader()
    w.writerows(mrows)
print("manifest rows:", len(mrows))

# ------------------------------------------------------ per-class group stats (split readiness)
def split_group(p):
    """Split-group unit: patient/eye key for .jpeg families; each zenodo file is
    its own singleton (no patient linkage exists) - still must respect dup clusters."""
    k = group_key(p)
    return k if k != "NONE" else f"zenodo_singleton:{p}"

PRIMARY = {"11.Normal Fundus": "No apparent DR", "3.Mild DR": "Mild NPDR",
           "4.Moderate DR": "Moderate NPDR", "5.Severe DR": "Severe NPDR",
           "13. Very Severe NPDR": "Severe NPDR (merged candidate)",
           "6.Proliferate DR": "PDR", "12. Advanced PDR": "PDR (merged candidate)"}
stats = {}
conflict_set = set(conflict_files)
for folder in PRIMARY:
    paths = [p for p in by_path if p.split("/")[0] == folder]
    flagged = [p for p in paths if p in qflags]
    conflicted = [p for p in paths if p in conflict_set]
    eligible = [p for p in paths if p not in qflags and p not in conflict_set]
    groups = collections.Counter(split_group(p) for p in eligible)
    multi = sum(1 for v in groups.values() if v > 1)
    both_eye = 0
    sides = collections.defaultdict(set)
    for p in eligible:
        fam = family(p)
        base = p.split("/", 1)[1]
        if fam == "prefixed":
            m = PREF.match(base)
            sides[(p.split("/")[0], m.group(2))].add(m.group(3))
        elif fam == "plain":
            m = PLAIN.match(base)
            if m.group(2):
                sides[(p.split("/")[0], m.group(1))].add(m.group(2))
    both_eye = sum(1 for v in sides.values() if len(v) == 2)
    stats[folder] = dict(primary_label=PRIMARY[folder], total=len(paths),
                         quality_flagged=len(flagged), conflict_involved=len(conflicted),
                         provisional_eligible=len(eligible), unique_groups=len(groups),
                         multi_image_groups=multi, both_eye_groups=both_eye,
                         zenodo_ungrouped=len([p for p in eligible if family(p) == "zenodo_v02"]))
NONDR = ["1.Dry AMD", "2.Wet AMD", "7.Cataract", "8.Hypertensive Retinopathy",
         "9.Pathological Myopia", "10.Glaucoma"]
nondr_stats = {}
for folder in NONDR:
    paths = [p for p in by_path if p.split("/")[0] == folder]
    flagged = [p for p in paths if p in qflags]
    conflicted = [p for p in paths if p in conflict_set]
    eligible = [p for p in paths if p not in qflags and p not in conflict_set]
    groups = len({split_group(p) for p in eligible})
    nondr_stats[folder] = dict(total=len(paths), quality_flagged=len(flagged),
                               conflict_involved=len(conflicted),
                               provisional_eligible=len(eligible), unique_groups=groups)
print(json.dumps(stats, indent=1))

summary = dict(
    register_pair_rows=n_pairs, register_unique_files=n_files_reg,
    components=len(order), conflict_pair_rows=n_quar,
    conflict_unique_files=len(conflict_files),
    cross_label_components=len(cross_label_comps),
    exact_groups=len(exact_groups),
    exact_files=sum(len(p) for _, p in exact_groups),
    strong_pairs=len(strong), strong_files=len(strong_files),
    cross_class_strong_pairs=len(cross_strong),
    strong_same_patient=same_patient, strong_cross_patient=cross_patient,
    strong_unknown_patient=unknown_patient, strong_within_class=within_class,
    quality_flagged=len(qflags), zenodo_confirmed=zen_confirmed,
    flag_tallies=dict(qfc), families=dict(fam_counts), primary_stats=stats, nondr_stats=nondr_stats)
with open(os.path.join(HERE, "build_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
print("wrote build_summary.json")
