"""Read-only audit scan of the DR fundus image dataset.
Emits one JSON line per image with integrity + hash + heuristics fields.
Never modifies any file in the dataset.
"""
import os, sys, json, hashlib, time, warnings
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # do not abort on large images; we record dimensions ourselves
warnings.simplefilter("ignore")

ROOT = r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_stats.jsonl")
ERRLOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors.txt")


def pack_bits(bits):
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def dhash(gray_small_arr):
    # 64-bit difference hash on 9x8 grayscale
    a = np.asarray(gray_small_arr.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return pack_bits((a[:, 1:] > a[:, :-1]).flatten())


def ahash16(gray_small_arr):
    a = np.asarray(gray_small_arr.resize((16, 16), Image.LANCZOS), dtype=np.float64)
    return pack_bits((a > a.mean()).flatten())


def main():
    t0 = time.time()
    n = 0
    with open(OUT, "w", encoding="utf-8") as outf, open(ERRLOG, "w", encoding="utf-8") as errf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames.sort()
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT)
                rec = {"path": rel.replace("\\", "/")}
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                    rec["bytes"] = len(data)
                    rec["sha256"] = hashlib.sha256(data).hexdigest()
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        im.load()
                        rec["fmt"] = im.format
                        rec["width"], rec["height"] = im.size
                        rec["mode"] = im.mode
                        rgb = im.convert("RGB")
                        arr = np.asarray(rgb, dtype=np.uint8)
                        small = rgb.resize((128, 128), Image.LANCZOS)
                        gray = np.asarray(small.convert("L"), dtype=np.float64)
                        hsv = np.asarray(small.convert("HSV"), dtype=np.float64)
                        rec["lum_mean"] = round(float(gray.mean()), 2)
                        rec["lum_std"] = round(float(gray.std()), 2)
                        rec["sat_mean"] = round(float(hsv[:, :, 1].mean()), 2)
                        # border ring vs center (detects non-circular / letterboxed / black-border images)
                        b = np.concatenate([gray[:12, :].ravel(), gray[-12:, :].ravel(),
                                            gray[:, :12].ravel(), gray[:, -12:].ravel()])
                        rec["border_mean"] = round(float(b.mean()), 2)
                        c = gray[32:96, 32:96]
                        rec["center_mean"] = round(float(c.mean()), 2)
                        # edge energy (blank/blurry flag)
                        gy, gx = np.gradient(gray)
                        rec["edge_energy"] = round(float(np.sqrt(gx ** 2 + gy ** 2).mean()), 3)
                        # near-white / near-black pixel fractions
                        rec["frac_dark"] = round(float((gray < 16).mean()), 4)
                        rec["frac_bright"] = round(float((gray > 240).mean()), 4)
                        rec["dhash"] = format(dhash(small.convert("L")), "016x")
                        rec["ahash"] = format(ahash16(small.convert("L")), "016x")
                        # channel means to spot grayscale-content or colour cast
                        rec["r_mean"] = round(float(arr[:, :, 0].mean()), 1)
                        rec["g_mean"] = round(float(arr[:, :, 1].mean()), 1)
                        rec["b_mean"] = round(float(arr[:, :, 2].mean()), 1)
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    errf.write(f"{rel}\t{type(e).__name__}: {e}\n")
                    errf.flush()
                outf.write(json.dumps(rec) + "\n")
                outf.flush()
                n += 1
                if n % 500 == 0:
                    el = time.time() - t0
                    print(f"{n} files, {el:.0f}s", flush=True)
    print(f"DONE {n} files in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
