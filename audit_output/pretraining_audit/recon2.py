import csv, collections
prim_rows = list(csv.DictReader(open("PROVISIONAL_ELIGIBILITY_PRIMARY.csv", encoding="utf-8-sig")))
sec_rows = list(csv.DictReader(open("PROVISIONAL_ELIGIBILITY_SECONDARY.csv", encoding="utf-8-sig")))
man = {r["image_path"]: r for r in csv.DictReader(open("PROVENANCE_MANIFEST.csv", encoding="utf-8-sig"))}
prim_by = {r["image_path"]: r for r in prim_rows}
sec_by = {r["image_path"]: r for r in sec_rows}
prim_band = {"3.Mild DR","4.Moderate DR","5.Severe DR","6.Proliferate DR","12. Advanced PDR","13. Very Severe NPDR","11.Normal Fundus"}

def qf(r): return r["quarantine_quality_flag"]=="yes"
def cf(r): return r["quarantine_label_conflict"]=="yes"

print("PROVENANCE_MANIFEST quality_flags non-empty count =", sum(1 for r in man.values() if r["quality_flags"]))
print()
print("PRIMARY manifest eligibility buckets (as stored):", dict(collections.Counter(r["eligibility"] for r in prim_rows)))
print()
print("PRIMARY TRUE flag-derived counts:")
q_only_prim = sum(1 for r in prim_rows if qf(r) and not cf(r))
c_only_prim = sum(1 for r in prim_rows if not qf(r) and cf(r))
both_prim = sum(1 for r in prim_rows if qf(r) and cf(r))
union_prim = q_only_prim + c_only_prim + both_prim
print("  Q-only (quality yes, conflict no) =", q_only_prim)
print("  C-only (quality no, conflict yes) =", c_only_prim)
print("  both (both yes) =", both_prim)
print("  EXCLUSIVE union =", union_prim, " (must be 326)")
print()
print("SECONDARY manifest eligibility buckets (as stored):", dict(collections.Counter(r["eligibility"] for r in sec_rows)))
print()
print("SECONDARY TRUE flag-derived counts:")
q_only_sec = sum(1 for r in sec_rows if qf(r) and not cf(r))
c_only_sec = sum(1 for r in sec_rows if not qf(r) and cf(r))
both_sec = sum(1 for r in sec_rows if qf(r) and cf(r))
union_sec = q_only_sec + c_only_sec + both_sec
print("  Q-only =", q_only_sec)
print("  C-only =", c_only_sec)
print("  both =", both_sec)
print("  EXCLUSIVE union =", union_sec, " (must be 326)")
print()
print("PRIMARY: mis-bucketed images (true Q-only not in 'quarantined_quality_review')")
m_qonly_prim = [p for p,r in prim_by.items() if qf(r) and not cf(r) and r["eligibility"]!="quarantined_quality_review"]
print("  count =", len(m_qonly_prim))
for p in m_qonly_prim:
    print("   ", p, "elig=", prim_by[p]["eligibility"], "qflag=", prim_by[p]["quarantine_quality_flag"], "cflag=", prim_by[p]["quarantine_label_conflict"], "sg=", prim_by[p]["split_group_id"])
print()
print("PRIMARY: mis-bucketed images (true C-only not in 'quarantined_label_conflict')")
m_conly_prim = [p for p,r in prim_by.items() if not qf(r) and cf(r) and r["eligibility"]!="quarantined_label_conflict"]
print("  count =", len(m_conly_prim))
for p in m_conly_prim:
    print("   ", p, "elig=", prim_by[p]["eligibility"], "qflag=", prim_by[p]["quarantine_quality_flag"], "cflag=", prim_by[p]["quarantine_label_conflict"], "sg=", prim_by[p]["split_group_id"])
print()
print("PRIMARY: total mis-bucketed =", len(m_qonly_prim)+len(m_conly_prim))
print()
print("SECONDARY: mis-bucketed images (true Q-only not in 'quarantined_quality_review')")
m_qonly_sec = [p for p,r in sec_by.items() if qf(r) and not cf(r) and r["eligibility"]!="quarantined_quality_review"]
print("  count =", len(m_qonly_sec))
for p in m_qonly_sec[:20]:
    print("   ", p, "elig=", sec_by[p]["eligibility"], "qflag=", sec_by[p]["quarantine_quality_flag"], "cflag=", sec_by[p]["quarantine_label_conflict"])
print()
print("PRIMARY: TRUE eligible (primary-band + no flag) =", sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)))
print("  manifest 'eligible_provisional' =", sum(1 for r in prim_rows if r["eligibility"]=="eligible_provisional"))
print("  manifest 'excluded_non_dr' =", sum(1 for r in prim_rows if r["eligibility"]=="excluded_non_dr_primary"))
print("  TRUE total primary =", sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)) + union_prim + sum(1 for r in prim_rows if r["original_label"] not in prim_band))
print("  manifest total primary =", sum(1 for r in prim_rows), " (must = 22453)")
print()
print("SECONDARY: TRUE eligible (all rows with no flag) =", sum(1 for r in sec_rows if not qf(r) and not cf(r)))
print("  manifest 'eligible_provisional' =", sum(1 for r in sec_rows if r["eligibility"]=="eligible_provisional"))
print("  manifest 'excluded_non_dr' =", sum(1 for r in sec_rows if r["eligibility"]=="excluded_non_dr_primary"))
print("  manifest total secondary =", sum(1 for r in sec_rows), " (must = 22453)")
print()
print("	PRIMARY manifest buckets vs true:")
print("	  quality-only: manifest =", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_quality_review"), " vs true=", q_only_prim, " match?", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_quality_review")==q_only_prim)
print("	  conflict-only: manifest =", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_label_conflict"), " vs true=", c_only_prim, " match?", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_label_conflict")==c_only_prim)
print("	  both: manifest =", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_conflict_and_quality"), " vs true=", both_prim, " match?", sum(1 for r in prim_rows if r["eligibility"]=="quarantined_conflict_and_quality")==both_prim)
print("	  sum =", sum(1 for r in prim_rows if r["eligibility"] in ("quarantined_quality_review","quarantined_label_conflict","quarantined_conflict_and_quality")), " vs true=", union_prim)
print()
print("SECONDARY manifest buckets vs true:")
print("	  quality-only: manifest =", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_quality_review"), " vs true=", q_only_sec, " match?", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_quality_review")==q_only_sec)
print("	  conflict-only: manifest =", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_label_conflict"), " vs true=", c_only_sec, " match?", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_label_conflict")==c_only_sec)
print("	  both: manifest =", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_conflict_and_quality"), " vs true=", both_sec, " match?", sum(1 for r in sec_rows if r["eligibility"]=="quarantined_conflict_and_quality")==both_sec)
print("	  sum =", sum(1 for r in sec_rows if r["eligibility"] in ("quarantined_quality_review","quarantined_label_conflict","quarantined_conflict_and_quality")), " vs true=", union_sec)
print()
print("PRIMARY: true eligible (primary-band no-flag) =", sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)))
print("  manifest 'eligible_provisional' =", sum(1 for r in prim_rows if r["eligibility"]=="eligible_provisional"), " match?",
      sum(1 for r in prim_rows if r["eligibility"]=="eligible_provisional")==sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)))
print()
print("PRIMARY: total reconciliation check (must = 22453):")
print("  manifest eligible(14201) + manifest quarantined total(313) + manifest excluded_non_dr(7939) =", 14201+313+7939, "vs 22453?", 14201+313+7939==22453)
print("  true eligible + true quarantined(326) + true excluded_non_dr =", sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)) + union_prim + sum(1 for r in prim_rows if r["original_label"] not in prim_band), "vs 22453?", sum(1 for r in prim_rows if r["original_label"] in prim_band and not qf(r) and not cf(r)) + union_prim + sum(1 for r in prim_rows if r["original_label"] not in prim_band)==22453)
print()
print("PRIMARY: 326 = 251 Q-only + 26 C-only + 49 both")
print("PRIMARY: manifest buckets sum to 313 -> 13 images MIS-BUCKETED (should be in quarantined buckets but aren't")
print("SECONDARY: manifest buckets sum to 326 -> 0 mis-bucketed (SECONDARY is CORRECT)")
