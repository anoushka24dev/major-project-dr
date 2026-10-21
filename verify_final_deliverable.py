import csv
from pathlib import Path
from collections import Counter
from dataset.manifest import load_manifest
from dataset.eligibility import filter_primary_eligible, is_any_quarantined
from dataset.split_validation import SplitValidator
from dataset.split_protocol import PrimarySplitProtocol, build_group_table_hash, build_report_from_assignment, _component_source_meta, write_split_csv
from dataset.grouping import group_membership_map
from collections import defaultdict

PRIMARY_CSV = Path("audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_PRIMARY.csv")
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")
OUT = Path("provisional_primary_split")
CSV_PATH = OUT / "provisional_primary_split.csv"
REPORT_PATH = OUT / "provisional_primary_split_report.md"

prim = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT, require_files_exist=True)
all_rows = list(prim)
pe = list(filter_primary_eligible(prim))
mg = group_membership_map(pe)
rows_by_path = {r["image_path"]: r for r in pe}

parent={}
def find(x):
    while parent.setdefault(x,x)!=x:
        parent[x]=parent[parent[x]]
        x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
for paths in mg.values():
    it=iter(paths); first=next(it)
    for p in it: union(first,p)
comp=defaultdict(set)
for p in parent: comp[find(p)].add(p)
all_paths={r["image_path"] for r in pe}
covered={p for c in comp.values() for p in c}
components=[tuple(sorted(c)) for c in (list(comp.values())+[(p,) for p in (all_paths-covered)])]
components.sort(key=lambda c:", ".join(sorted(c)))

group_table_hash=build_group_table_hash(components)
proto=PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10)
assign, _ = proto.allocate(components, rows_by_path)

manifest_entry=pe[0]
manifest_version=manifest_entry.get("manifest_version","")
approval_reference=manifest_entry.get("approval_reference","")

final_report=build_report_from_assignment(
    assign, rows_by_path, components, proto,
    manifest_version=manifest_version,
    manifest_path=PRIMARY_CSV,
    approval_reference=approval_reference,
    group_table_hash=group_table_hash,
)

items=[(k,v) for k,v in assign.items()]
validator=SplitValidator(pe, allowed_partitions=["train","validation","test"], exclude_quarantined=True, exclude_non_dr_primary=True)
try:
    vr=validator.validate(items)
    final_report["validation"]=vr
except Exception as e:
    final_report["validation_error"]=str(e)

pure_zen_by_part=defaultdict(int)
for c in components:
    part=assign[c[0]]
    meta=_component_source_meta(c, rows_by_path)
    if meta["zenodo_pure_singleton"]:
        pure_zen_by_part[part]+=1
final_report["zenodo_pure_singletons_by_partition"]=dict(pure_zen_by_part)

write_split_csv(assign, rows_by_path, components, proto, CSV_PATH)

lines=[]
lines.append("# Provisional Primary Five-Band Split Report")
lines.append("")
lines.append("**Status:** PROVISIONAL -- not an approved final dataset; generated from `PROVISIONAL_ELIGIBILITY_PRIMARY.csv`.")
lines.append("")
lines.append("## Provenance")
lines.append(f"- Manifest version: `{final_report['manifest_version']}`")
lines.append(f"- Manifest path: `{final_report['manifest_path']}`")
lines.append(f"- Manifest file SHA-256: `{final_report['manifest_file_hash']}`")
lines.append(f"- Approval reference: `{final_report['approval_reference']}`")
lines.append(f"- Generated split CSV: `{CSV_PATH}`")
lines.append(f"- Seed: `{final_report['seed']}`")
lines.append(f"- Target fractions: `{final_report['target_fractions']}`")
lines.append(f"- Soft Zenodo test cap: `{final_report['near_zenodo_test_frac']}`")
lines.append("")
lines.append("## Group atomicity")
lines.append(f"- Total components allocated: `{final_report['total_components']}`")
lines.append(f"- Group-table SHA-256: `{final_report['group_table_hash']}`")
lines.append(f"- Validation result from `SplitValidator`: `{final_report.get('validation') is not None}`")
if 'validation_error' in final_report:
    lines.append(f"- Validation error: `{final_report['validation_error']}`")
lines.append("")
lines.append("## Image counts")
lines.append("")
lines.append("| Partition | Images | Ratio |")
lines.append("|-----------|--------|-------|")
for p in ["train","validation","test"]:
    n=final_report['partition_counts'][p]
    r=final_report['partition_ratios'][p]
    lines.append(f"| {p} | {n} | {r:.4f} |")
lines.append("")
lines.append("## Per-class counts")
lines.append("")
lines.append("| Band | Images | Groups | Train | Validation | Test |")
lines.append("|------|--------|--------|-------|------------|------|")
for band in ["No apparent DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"]:
    cc=final_report['class_counts'][band]
    lines.append(f"| {band} | {cc['images']} | {cc['groups']} | {cc['train']} | {cc['validation']} | {cc['test']} |")
lines.append("")
lines.append("## Zenodo-derived images without verified patient linkage")
lines.append("")
lines.append(f"- Total Zenodo-eligible images in this split: `{sum(final_report['zenodo_by_partition'].values())}`")
lines.append("")
lines.append("| Partition | Zenodo images | Pure-Zenodo singletons |")
lines.append("|-----------|---------------|------------------------|")
for p in ["train","validation","test"]:
    zn=final_report['zenodo_by_partition'][p]
    pz=final_report['zenodo_pure_singletons_by_partition'][p]
    lines.append(f"| {p} | {zn} | {pz} |")
lines.append("")
lines.append("## Limitations")
for lim in final_report['limitations']:
    lines.append(f"- {lim}")
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- Ratios are approximate. Actual proportions are reported above; they are not claimed to be exact 70/15/15.")
lines.append("- Duplicate-connected components are the atomic unit for this split; no such component crosses partitions.")
lines.append("- Patient independence is **not** claimed for the Zenodo-derived images.")
lines.append("- The Severe NPDR and PDR taxonomy merges remain provisional and unsigned.")

REPORT_PATH.write_text("\n".join(lines)+"\n", encoding="utf-8")

csv_rows=list(csv.DictReader(CSV_PATH.read_text(encoding='utf-8').splitlines()))
text=REPORT_PATH.read_text(encoding='utf-8')

checks = []
checks.append(("CSV rows == 14201", len(csv_rows)==14201))
checks.append(("CSV unique paths == 14201", len({r['image_path'] for r in csv_rows})==14201))
checks.append(("all partitions present", {r['partition'] for r in csv_rows}=={"train","validation","test"}))
checks.append(("all rows primary task", {r['task'] for r in csv_rows}=={"primary"}))
checks.append(("5 bands present", {r['primary_class'] for r in csv_rows}=={"No apparent DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"}))
band_present_all = all(
    {r['partition'] for r in csv_rows if r['primary_class']==band}=={"train","validation","test"}
    for band in ["No apparent DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"]
)
checks.append(("each band in all 3 partitions", band_present_all))
quarantined_set = {r['image_path'] for r in all_rows if is_any_quarantined(r)}
csv_paths = {r['image_path'] for r in csv_rows}
checks.append(("no quarantined paths assigned", not quarantined_set & csv_paths))
excluded_set = {r['image_path'] for r in all_rows if r.get('eligibility','').startswith('excluded_non_dr_primary')}
checks.append(("no excluded-non-DR paths assigned", not excluded_set & csv_paths))
checks.append(("zenodo_unlinked count == 503", sum(1 for r in csv_rows if r['zenodo_unlinked']=='True')==503))
checks.append(("patient_key_available count == 13698", sum(1 for r in csv_rows if r['patient_key_available']=='True')==13698))
checks.append(("report markers present", all(k in text for k in ["503","393","110","patient independence cannot be claimed"])))
checks.append(("SplitValidator passed", final_report.get('validation',{}).get('valid') is True))
checks.append(("report has seed + manifest version + approval", all(k in text for k in ["12345","provisional-v1-2026-10-13","Option A approved by user 2026-10-13 (7 safeguards)"])))

print("=== FINAL DELIVERABLE CHECKS ===")
ok=True
for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = ok and passed
print("\nALL CHECKS PASS:", ok)

print("\n=== ARTIFACT PATHS ===")
print("CSV:", CSV_PATH)
print("REPORT:", REPORT_PATH)
