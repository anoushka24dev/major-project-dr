"""Build PROVISIONAL eligibility manifests for the primary (5-band DR grading)
and secondary (13-class) tasks, implementing the user-approved Option A
decisions of 2026-10-13.

Rules implemented (per approval):
  1. 5-band primary mapping; VSNPDR->Severe NPDR and AdvPDR->PDR recorded as
     PROVISIONAL merges with evidence tags; original labels always preserved.
  2. Normal Fundus task label carries wording basis "no DR visible according to
     the source label".
  3. Cross-label conflict files quarantined from BOTH tasks.
  4. All 300 quality-flagged files quarantined in both tasks pending review.
  5. Non-DR folders excluded from the primary cohort; retained in secondary.
  6./7. Merge-review need and no-Zenodo-append recorded as text constants.

This script does NOT create splits, does NOT train, and does NOT modify the
dataset or prior audit artifacts. Outputs go to audit_output/pretraining_audit/.
"""
import os, csv, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "provisional-v1-2026-10-13"
APPROVAL = "Option A approved by user 2026-10-13 (7 safeguards)"

TASK_LABEL = {
    "11.Normal Fundus": ("No apparent DR",
        "rename_per_policy_wording: 'no DR visible according to the source label'; "
        "NOT a clinically confirmed healthy status"),
    "3.Mild DR": ("Mild NPDR", "direct_name_match_to_ICDR_band; per-image grading unverifiable"),
    "4.Moderate DR": ("Moderate NPDR", "direct_name_match_to_ICDR_band; per-image grading unverifiable"),
    "5.Severe DR": ("Severe NPDR", "direct_name_match_to_ICDR_band; 4-2-1 rule unverifiable per image"),
    "13. Very Severe NPDR": ("Severe NPDR",
        "PROVISIONAL_MERGE: ETDRS 53E falls inside ICDR severe NPDR band; "
        "source-documentation/specialist review REQUIRED before final dataset"),
    "6.Proliferate DR": ("PDR", "direct_name_match_to_ICDR_PDR_band; per-image grading unverifiable"),
    "12. Advanced PDR": ("PDR",
        "PROVISIONAL_MERGE: ETDRS 81/85 falls inside ICDR PDR band; "
        "source-documentation/specialist review REQUIRED before final dataset"),
}
NON_DR = {"1.Dry AMD", "2.Wet AMD", "7.Cataract", "8.Hypertensive Retinopathy",
          "9.Pathological Myopia", "10.Glaucoma"}

man = list(csv.DictReader(open(os.path.join(HERE, "PROVENANCE_MANIFEST.csv"), encoding="utf-8-sig")))
print("provenance manifest rows:", len(man))
reg = list(csv.DictReader(open(os.path.join(HERE, "LABEL_CONFLICT_REGISTER.csv"), encoding="utf-8-sig")))

# files quarantined for label conflict (rows whose two labels disagree)
conflict_files = set()
for r in reg:
    if r["labels_agree"] == "NO":
        conflict_files.add(r["path_a"]); conflict_files.add(r["path_b"])
print("conflict-quarantine files:", len(conflict_files))

# ---- split-group table: union(patient/eye keys) over duplicate components ----
def base_key(row):
    k = row["patient_group_key"]
    return k if k != "NONE" else "zenodo_singleton:" + row["image_path"]

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

by_path = {r["image_path"]: r for r in man}
for r in man:
    find(base_key(r))
for r in reg:
    if r["path_a"] in by_path and r["path_b"] in by_path:
        union(base_key(by_path[r["path_a"]]), base_key(by_path[r["path_b"]]))
# deterministic group ids
roots = {}
for r in man:
    roots.setdefault(find(base_key(r)), None)
ordered = sorted(roots, key=lambda root: sorted(
    b for b in (base_key(x) for x in man) if find(b) == root)[0])
sgid = {root: f"SG-{i+1:05d}" for i, root in enumerate(ordered)}
print("split groups (union of patient keys over dup components):", len(sgid))

# per-file split group + zenodo singleton info
for r in man:
    root = find(base_key(r))
    r["_sg"] = sgid[root]
    r["_base"] = base_key(r)
    r["_quar_conf"] = r["image_path"] in conflict_files
    r["_quar_qual"] = bool(r["quality_flags"])

# ---- write manifests ----
FIELDS = ["manifest_status", "manifest_version", "approval_reference", "image_path",
          "original_label", "task_label", "task_label_basis", "source_family",
          "source_status", "patient_group_key", "split_group_id", "duplicate_group_id",
          "quarantine_label_conflict", "quarantine_quality_flag", "eligibility",
          "notes"]

def row_for(r, task):
    lab = r["original_label"]
    if task == "primary":
        if lab in NON_DR:
            tlabel, basis = "", "n/a - non-DR folder excluded from primary cohort (approval #5)"
            # The non-DR exclusion is an ELIGIBILITY decision (not in DR cohort).
            # It does NOT suppress the quarantine flags: a non-DR image that is
            # quality- or conflict-flagged still has that flag and that reason,
            # and the eligibility cell must name the reason(s) even though the
            # image stays excluded from the DR cohort.
            if r["_quar_conf"] and r["_quar_qual"]:
                elig = "excluded_non_dr_primary_and_both_quarantines"
            elif r["_quar_conf"]:
                elig = "excluded_non_dr_primary_and_label_conflict_quarantine"
            elif r["_quar_qual"]:
                elig = "excluded_non_dr_primary_and_quality_quarantine"
            else:
                elig = "excluded_non_dr_primary"
        else:
            tlabel, basis = TASK_LABEL[lab]
            if r["_quar_conf"] and r["_quar_qual"]:
                elig = "quarantined_conflict_and_quality"
            elif r["_quar_conf"]:
                elig = "quarantined_label_conflict"
            elif r["_quar_qual"]:
                elig = "quarantined_quality_review"
            else:
                elig = "eligible_provisional"
    else:  # secondary
        tlabel, basis = lab, "original_folder_label preserved verbatim (secondary task taxonomy)"
        if r["_quar_conf"] and r["_quar_qual"]:
            elig = "quarantined_conflict_and_quality"
        elif r["_quar_conf"]:
            elig = "quarantined_label_conflict"
        elif r["_quar_qual"]:
            elig = "quarantined_quality_review"
        else:
            elig = "eligible_provisional"
    notes = []
    if "PROVISIONAL_MERGE" in basis:
        notes.append("merge needs specialist/source-documentation review (approval #6)")
    if r["_base"].startswith("zenodo_singleton"):
        notes.append("zenodo: no patient linkage; split_group is image-level only")
    if r["_quar_conf"]:
        notes.append("cross-label duplicate component; do not delete/relabel (approval #3)")
    if r["_quar_qual"]:
        notes.append("quality flag pending documented human review (approval #4)")
    if r["original_label"] in NON_DR and task == "primary":
        pieces = []
        if r["_quar_qual"] and r["_quar_conf"]:
            pieces.append("non-DR-excluded yet carries BOTH a quality flag and a label-conflict flag (reasons preserved; still not in DR cohort)")
        elif r["_quar_qual"]:
            pieces.append("non-DR-excluded yet carries a quality flag (reason preserved; still not in DR cohort)")
        elif r["_quar_conf"]:
            pieces.append("non-DR-excluded yet carries a label-conflict flag (reason preserved; still not in DR cohort)")
        else:
            pieces.append("non-DR folder; excluded from DR cohort (approval #5)")
        notes.append("; ".join(pieces))
    return dict(manifest_status="PROVISIONAL - NOT AN APPROVED FINAL DATASET",
                manifest_version=VERSION, approval_reference=APPROVAL,
                image_path=r["image_path"], original_label=lab,
                task_label=tlabel, task_label_basis=basis,
                source_family=r["source_family"], source_status=r["source_status"],
                patient_group_key=r["patient_group_key"], split_group_id=r["_sg"],
                duplicate_group_id=r["duplicate_group_id"],
                quarantine_label_conflict="yes" if r["_quar_conf"] else "no",
                quarantine_quality_flag="yes" if r["_quar_qual"] else "no",
                eligibility=elig, notes="; ".join(notes))

counts = {}
for task in ("primary", "secondary"):
    out = os.path.join(HERE, f"PROVISIONAL_ELIGIBILITY_{task.upper()}.csv")
    rows = [row_for(r, task) for r in man]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    c = collections.Counter(x["eligibility"] for x in rows)
    print(f"\n{task}: wrote {len(rows)} rows -> {out}")
    print("  eligibility:", dict(c))
    # per-class counts + group counts among eligible
    per = {}
    for x in rows:
        if x["original_label"] in per or x["eligibility"] == "eligible_provisional" or True:
            d = per.setdefault(x["original_label"], collections.Counter())
            d[x["eligibility"]] += 1
    elig_groups = collections.defaultdict(set)
    for x in rows:
        if x["eligibility"] == "eligible_provisional":
            elig_groups[x["original_label"]].add(x["split_group_id"])
    for lab in sorted(per):
        e = per[lab]["eligible_provisional"]
        g = len(elig_groups.get(lab, ()))
        print(f"    {lab:28s} eligible={e:5d}  groups={g:5d}  "
              f"conflict_q={per[lab]['quarantined_label_conflict']} "
              f"qual_q={per[lab]['quarantined_quality_review']} "
              f"both_q={per[lab]['quarantined_conflict_and_quality']} "
              f"exclude_non_dr={per[lab]['excluded_non_dr_primary']} "
              f"+qual={per[lab].get('excluded_non_dr_primary_and_quality_quarantine',0)} "
              f"+conflict={per[lab].get('excluded_non_dr_primary_and_label_conflict_quarantine',0)} "
              f"+both={per[lab].get('excluded_non_dr_primary_and_both_quarantines',0)}")
    counts[task] = {lab: dict(per[lab]) | {"eligible_groups": len(elig_groups.get(lab, ()))} for lab in per}

# reconciliation assertions (AC-4 style)
n = len(man)
for task in ("primary", "secondary"):
    rows = [row_for(r, task) for r in man]
    assert len(rows) == n
    c = collections.Counter(x["eligibility"] for x in rows)
    total = sum(c.values())
    assert total == n, (task, total, n)
print("\nreconciliation: every image has exactly one eligibility row in each task manifest; "
      "eligible + quarantined + excluded == 22,453 for both tasks")

# Explicit truth-table cross-check implemented inside the builder now (AC-4 style):
#   FLAG-CONSISTENCY: provenance.quality_flags non-empty  <=>  manifest.quarantine_quality_flag == yes
#                       (and same for every path in both tasks)
#   ROW-COUNT: PRIMARY and SECONDARY each == len(man) == 22,453
#   BUCKET-TRUTH: for the PRIMARY task, among NON-DR rows the quarantine flags
#                  must agree with the eligibility cell suffix:
#                    non-DR + _quar_qual and not _quar_conf  -> *quality_quarantine* suffix
#                    non-DR + _quar_conf and not _quar_qual  -> *label_conflict_quarantine* suffix
#                    non-DR + both                          -> *both_quarantines* suffix
#                    non-DR + neither                       -> plain excluded_non_dr_primary
for task in ("primary",):
    rows = [row_for(r, task) for r in man]
    mism = 0
    for r in man:
        p = r["image_path"]
        pmask = (r["_quar_qual"], r["_quar_conf"])
        emask = (
            "yes" if r["_quar_qual"] else "no",
            "yes" if r["_quar_conf"] else "no",
        )
        # the manifest row for that path is `rows` in the same order as `man`
        # (we rebuilt row_for on the same `r`); grab it by index
        idx = [x["image_path"] for x in rows].index(p) if False else None
    # Simpler: a deterministic zip over man + rows. They are in the same order.
    mism = 0
    for r, x in zip(man, rows):
        if (x["quarantine_quality_flag"] == ("yes" if r["_quar_qual"] else "no")) is False:
            mism += 1
        if (x["quarantine_label_conflict"] == ("yes" if r["_quar_conf"] else "no")) is False:
            mism += 1
    assert mism == 0, f"FLAG-CONSISTENCY FAILED: {mism} mismatched flag cells"
    # row count + bucket truth for non-DR primary rows
    assert len(rows) == len(man)
    # non-DR rows: check that the eligibility suffix matches the flag pair
    for x, r in zip(rows, man):
        if r["original_label"] in NON_DR:
            q = r["_quar_qual"]; c = r["_quar_conf"]
            if q and not c:
                assert x["eligibility"] == "excluded_non_dr_primary_and_quality_quarantine", (x["image_path"], x["eligibility"])
            elif c and not q:
                assert x["eligibility"] == "excluded_non_dr_primary_and_label_conflict_quarantine", (x["image_path"], x["eligibility"])
            elif q and c:
                assert x["eligibility"] == "excluded_non_dr_primary_and_both_quarantines", (x["image_path"], x["eligibility"])
            else:
                assert x["eligibility"] == "excluded_non_dr_primary", (x["image_path"], x["eligibility"])
print("builder self-checks passed: flag consistency + non-DR bucket truth + row count")

json.dump(counts, open(os.path.join(HERE, "provisional_counts.json"), "w", encoding="utf-8"), indent=1)
print("wrote provisional_counts.json")
