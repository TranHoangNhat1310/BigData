# ================= SA1B → SA1B-Matte + JSON =================
import os, json, cv2, time
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
import argparse

# ================= CONFIG =================
# Thầy thêm phần argparse để sau này em chạy dòng lệnh cho dễ thay đổi đường dẫn
def get_args():
    parser = argparse.ArgumentParser(description="SA1B Data Processing MapReduce")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input/sa-1b-part-000999/SA-1B-Part-000999")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/SA1B-Matte2")
    return parser.parse_args()

TOP_K = 5
MAX_RES = 512

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ================= LABEL CONVERTER (ZIM-style) =================
def label_converter(mask):
    return cv2.GaussianBlur(mask.astype(np.float32), (11,11), 0)

# ================= MAP FUNCTION =================
def process_one_json(json_file, input_dir, output_dir):
    path = os.path.join(input_dir, json_file)
    try:
        with open(path) as f:
            data = json.load(f)
    except:
        return 0

    H, W = data["image"]["height"], data["image"]["width"]
    masks = []

    # decode segmentation / RLE
    for ann in data["annotations"]:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue
        try:
            if isinstance(seg, dict):
                rle = seg
            else:
                rle = mask_utils.merge(mask_utils.frPyObjects(seg, H, W))
            m = mask_utils.decode(rle).astype(np.float32)
            masks.append((m.sum(), m))
        except:
            continue

    if len(masks) == 0:
        return 0

    masks = sorted(masks, key=lambda x: -x[0])[:TOP_K]
    image_id = data["image"]["image_id"]
    saved = 0

    for idx, (_, m) in enumerate(masks):
        alpha = label_converter(m)

        h, w = alpha.shape
        scale = min(MAX_RES/h, MAX_RES/w, 1.0)
        if scale < 1:
            alpha = cv2.resize(alpha, (int(w*scale), int(h*scale)))

        alpha_uint8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # Save alpha image
        out_img = os.path.join(output_dir, f"{image_id}_{idx}_alpha.png")
        cv2.imwrite(out_img, alpha_uint8)

        # Save JSON metadata tương ứng
        out_json = os.path.join(output_dir, f"{image_id}_{idx}_meta.json")
        meta = {
            "source": json_file,
            "width": alpha_uint8.shape[1],
            "height": alpha_uint8.shape[0],
            "mask_idx": idx,
            "objects": data.get("annotations", [])
        }
        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)

        saved += 1

    return saved

# ================= RUN =================
if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input dir not found: {args.input_dir}")
        exit()

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    print(f"📄 Found {len(json_files)} JSON files")
    print("🧠 Running SAFE MapReduce (sequential map)")
    print("="*60)

    start = time.time()
    total = 0

    for i, jf in enumerate(tqdm(json_files, desc="Generating SA1B-Matte")):
        total += process_one_json(jf, args.input_dir, args.output_dir)

        if (i+1) % 100 == 0:
            print(f"✔ Processed {i+1}/{len(json_files)} | Total mattes: {total}")

    print("\n================ DONE =================")
    print(f"🖼 Total alpha mattes: {total}")
    print(f"📁 Output folder: {args.output_dir}")
    print(f"⏱ Runtime: {time.time()-start:.1f}s")
    print("======================================")
