# Leakage audit report

- Policy: `data-integrity-v1-2026-11-09`
- Generated: 2026-11-09T13:36:41Z
- Images discovered / included / excluded: 23890 / 15894 / 7996

## What was checked

| Check | Method | Result |
|---|---|---|
| Exact duplicates | SHA-256 over every decoded file | 16 group(s), 33 file(s) |
| Perceptual duplicates | dHash distance <= 4 and aHash distance <= 6, full pairwise pass | 969 pair(s), 1392 file(s) |
| Patient / eye grouping | documented filename conventions (folder-scoped IDs) | 7482 key(s) |
| Duplicate/patient groups inside one split | union of both group systems | 8562 split group(s) |
| Cross-label conflicts | duplicate group members with different five-class labels | 18 group(s) quarantined |
| Cross-source copies | identical SHA-256 under two different dataset roots | 0 image(s) |

## Isolation assertions

**FAILED** — the manifest must not be used:
- identical image (sha256) '430ae37a791d8d642c658a1f6e91d8481b3d3f3a5a5a2471c58ec9555905223d' spans partitions ['test', 'train']
- identical image (sha256) '9c8291bde7a25889e1fafd9dc904e7d460b36c9a4347971df8019c9ceb582ef2' spans partitions ['test', 'train']
- identical image (sha256) 'efdf06458c7d7acc9c7fbefba1a08dc8752ced98d75979c3252c8f897d4b5e59' spans partitions ['test', 'validation']
- identical image (sha256) 'e92f2e8351313430aa2caf26bf4f41aa85e8f2ff4f45371cc51fea59f1ce54ae' spans partitions ['test', 'validation']
- identical image (sha256) '718487a5a5ab8dda5f99de5c78ae5c787b3100f290ee7e829001d3d0b4ea55d7' spans partitions ['test', 'validation']
- identical image (sha256) '12c6f28b99bbac190edd3e197c985e2c730a75f70f4c66fe4f462589aa9456e6' spans partitions ['test', 'validation']
- identical image (sha256) 'd7aeb2049d0aad9a607e67241e085f24b8d9c006917ab4398c12db9b8db27313' spans partitions ['test', 'validation']
- identical image (sha256) 'fb4172f8b3fb5907c9046fd12af8edeb92d39983843d73a004581be505374cf2' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1892' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1893' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1514' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1959' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1963' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=102' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1819' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=178' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1817' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=167' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=178' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1488' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=608' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=852' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=855' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1218' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=377' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1397' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1487' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=889' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1209' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1210' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1365' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1847' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1213' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1247' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=1846' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=1877' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=681' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=1879' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1215' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1290' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1344' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=998' spans partitions ['test', 'train']
- patient/eye group 'prefixed:1.Dry AMD|ID=1840' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=572' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=55' spans partitions ['test', 'train']
- patient/eye group 'prefixed:10.Glaucoma|ID=1267' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=615' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=238' spans partitions ['test', 'train', 'validation']
- patient/eye group 'prefixed:1.Dry AMD|ID=212' spans partitions ['train', 'validation']
- patient/eye group 'prefixed:10.Glaucoma|ID=1281' spans partitions ['test', 'train']

## Split distributions

| Split | Total | No apparent DR | Mild NPDR | Moderate NPDR | Severe NPDR | PDR |
|---|---:|---:|---:|---:|---:|---:|
| test | 2168 | 471 | 355 | 708 | 21 | 10 |
| train | 10120 | 2121 | 1705 | 3511 | 66 | 80 |
| validation | 2169 | 466 | 364 | 715 | 21 | 24 |

## Group sizes by split

- split groups: 8562
- largest split group: 5 image(s)
- groups spanning more than one partition: 0

## Patient-level limitation

Patient identifiers are derived only from the documented filename conventions. Two of the three source families in the Mendeley corpus expose an identifier and laterality (`<Class><id>_<left|right>_...` and `<id>[_left|right]`); the number-only `.jpg` files publish none. Those images are grouped at image level and by duplicate cluster only, so patient-independent evaluation is claimed **only** for the subset with a derived identifier.

Cross-folder identifier collisions (the same numeric ID in different class folders) are treated as unrelated patients: identifiers such as `0_right.jpeg` recur in every folder. This is the repository's established policy and it is conservative for leakage (it cannot merge two real patients into one split group by accident, but it can miss a genuine cross-folder patient).
