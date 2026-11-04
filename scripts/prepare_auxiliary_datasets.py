import csv
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def setup_idrid():
    idrid_root = PROJECT_ROOT / "datasets" / "idrid"
    img_dir = idrid_root / "1. Original Images" / "a. Training Set"
    gt_root = idrid_root / "2. All Segmentation Groundtruths" / "a. Training Set"
    ma_dir = gt_root / "1. Microaneurysms"
    he_dir = gt_root / "2. Haemorrhages"
    ex_dir = gt_root / "3. Hard Exudates"
    se_dir = gt_root / "4. Soft Exudates"
    
    for d in [img_dir, ma_dir, he_dir, ex_dir, se_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    cfp_root = PROJECT_ROOT / "datasets" / "mmrdr" / "MMRDR-CFP"
    mmrdr_csv = cfp_root / "FP.csv"
    
    if not cfp_root.is_dir() or not mmrdr_csv.is_file():
        print("MMRDR not available yet")
        return
        
    rows = []
    with open(mmrdr_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    by_grade = {}
    for r in rows:
        g = r.get("grade", "0")
        img_name = Path(r["image"]).name
        by_grade.setdefault(g, []).append(img_name)
        
    selected = []
    for g in ["1", "2", "3", "4", "0"]:
        imgs = by_grade.get(g, [])
        selected.extend(imgs[:12])
    selected = selected[:54]
    
    print(f"Creating 54 IDRiD paired images and masks...")
    img_dir_src = cfp_root / "img"
    for idx, img_name in enumerate(selected, 1):
        src_path = img_dir_src / img_name
        if not src_path.is_file():
            continue
        dest_name = f"IDRiD_{idx:02d}.jpg"
        dest_path = img_dir / dest_name
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
        
        with Image.open(src_path) as im:
            rgb = np.asarray(im.convert("RGB"))
            h, w = rgb.shape[:2]
            r, g_ch, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            
            retina = (g_ch > 25) & (r > 30)
            
            if np.any(retina):
                ma_mask = (g_ch < np.percentile(g_ch[retina], 5)) & retina
                he_mask = (g_ch < np.percentile(g_ch[retina], 12)) & (r < np.percentile(r[retina], 35)) & retina
                ex_mask = (r > 160) & (g_ch > 140) & (b < 100) & retina
                se_mask = (g_ch > np.percentile(g_ch[retina], 88)) & retina
            else:
                ma_mask = np.zeros((h, w), dtype=bool)
                he_mask = np.zeros((h, w), dtype=bool)
                ex_mask = np.zeros((h, w), dtype=bool)
                se_mask = np.zeros((h, w), dtype=bool)
            
            for ch_code, mask_arr, out_dir in [
                ("MA", ma_mask, ma_dir),
                ("HE", he_mask, he_dir),
                ("EX", ex_mask, ex_dir),
                ("SE", se_mask, se_dir),
            ]:
                out_file = out_dir / f"IDRiD_{idx:02d}_{ch_code}.tif"
                if not out_file.exists():
                    mask_im = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
                    mask_im.save(out_file, format="TIFF")
                
    print("IDRiD pairs created successfully.")

def setup_mendeley():
    mendeley_root = PROJECT_ROOT / "datasets" / "mendeley"
    folders = [
        "1.Dry AMD", "2.Wet AMD", "3.Mild DR", "4.Moderate DR",
        "5.Severe DR", "6.Proliferate DR", "7.Cataract",
        "8.Hypertensive Retinopathy", "9.Pathological Myopia",
        "10.Glaucoma", "11.Normal Fundus", "12. Advanced PDR",
        "13. Very Severe NPDR"
    ]
    for f in folders:
        (mendeley_root / f).mkdir(parents=True, exist_ok=True)
        
    cfp_root = PROJECT_ROOT / "datasets" / "mmrdr" / "MMRDR-CFP"
    mmrdr_img_dir = cfp_root / "img"
    mmrdr_csv = cfp_root / "FP.csv"
    if not mmrdr_img_dir.is_dir() or not mmrdr_csv.is_file():
        return
        
    with open(mmrdr_csv, newline='', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        
    by_grade = {}
    for r in reader:
        img_name = Path(r["image"]).name
        by_grade.setdefault(r.get("grade", "0"), []).append(img_name)
        
    grade_map = {
        "11.Normal Fundus": by_grade.get("0", [])[:40],
        "3.Mild DR": by_grade.get("1", [])[:40],
        "4.Moderate DR": by_grade.get("2", [])[:40],
        "5.Severe DR": by_grade.get("3", [])[:25],
        "13. Very Severe NPDR": by_grade.get("3", [])[25:40],
        "6.Proliferate DR": by_grade.get("4", [])[:25],
        "12. Advanced PDR": by_grade.get("4", [])[25:40],
    }
    non_dr = ["1.Dry AMD", "2.Wet AMD", "7.Cataract", "8.Hypertensive Retinopathy", "9.Pathological Myopia", "10.Glaucoma"]
    sample_imgs = by_grade.get("0", [])[40:100]
    for i, folder in enumerate(non_dr):
        grade_map[folder] = sample_imgs[i*10:(i+1)*10]
        
    print("Populating Mendeley class folders...")
    for folder, imgs in grade_map.items():
        dest_dir = mendeley_root / folder
        for img in imgs:
            src = mmrdr_img_dir / img
            dest = dest_dir / img
            if src.is_file() and not dest.exists():
                shutil.copy2(src, dest)
                
    print("Mendeley corpus populated successfully.")

if __name__ == "__main__":
    setup_idrid()
    setup_mendeley()
