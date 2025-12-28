
import os
import sys
import glob
import time
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Thêm đường dẫn hiện tại để import zim_anything
sys.path.append(os.getcwd())
try:
    from zim_anything import zim_model_registry, ZimPredictor
except ImportError:
    # Fallback nếu đang ở trong folder con
    sys.path.append(os.path.dirname(os.getcwd()))
    from zim_anything import zim_model_registry, ZimPredictor

def get_args():
    parser = argparse.ArgumentParser(description="ZIM MapReduce Batch Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input dataset folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save masks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ZIM checkpoint")
    parser.add_argument("--backbone", type=str, default="vit_b", choices=["vit_b", "vit_l"], help="Model backbone")
    parser.add_argument("--batch_size", type=int, default=4, help="Chunk Size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()

class MapReduceEngine:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        print(f"⚙️ Initializing ZIM Model ({args.backbone})...")
        self.model = zim_model_registry[args.backbone](checkpoint=args.checkpoint)
        if self.device == "cuda":
            self.model.cuda()
        self.predictor = ZimPredictor(self.model)
        print("✅ Model loaded successfully!")

    def get_all_images(self):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.args.input_dir, "**", ext), recursive=True))
        return sorted(list(set(files)))

    def split_batch(self, file_list, batch_size):
        for i in range(0, len(file_list), batch_size):
            yield file_list[i : i + batch_size]

    def map_process(self, batch_files):
        for img_path in batch_files:
            try:
                image = cv2.imread(img_path)
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(image)
                h, w = image.shape[:2]
                box = np.array([0, 0, w, h]) 
                masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
                best_mask = masks[0]
                mask_img = (best_mask * 255).astype(np.uint8)
                filename = os.path.basename(img_path)
                save_path = os.path.join(self.args.output_dir, filename)
                cv2.imwrite(save_path, mask_img)
            except Exception as e:
                print(f"⚠️ Error: {e}")
            torch.cuda.empty_cache()

    def run(self):
        all_files = self.get_all_images()
        total_files = len(all_files)
        print(f"📂 Found {total_files} images.")
        if total_files == 0: return
        os.makedirs(self.args.output_dir, exist_ok=True)
        print(f"🚀 Starting MapReduce (Batch Size = {self.args.batch_size})...")
        pbar = tqdm(total=total_files, unit="img")
        for batch in self.split_batch(all_files, self.args.batch_size):
            self.map_process(batch)
            pbar.update(len(batch))
        pbar.close()
        print("🎉 DONE!")

if __name__ == "__main__":
    args = get_args()
    engine = MapReduceEngine(args)
    engine.run()
