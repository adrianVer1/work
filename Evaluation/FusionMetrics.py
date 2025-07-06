import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import argparse

def mutual_information(x, y, bins=256):
    """Compute mutual information between two images."""
    hist_2d, x_edges, y_edges = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))
    return mi

def evaluate(fused_dir, ir_dir, vis_dir):
    files = sorted(os.listdir(fused_dir))
    entropies = []
    ssim_ir_list = []
    ssim_vis_list = []
    mi_ir_list = []
    mi_vis_list = []
    cc_ir_list = []
    cc_vis_list = []

    for f in files:
        fused_path = os.path.join(fused_dir, f)
        ir_path = os.path.join(ir_dir, f)
        vis_path = os.path.join(vis_dir, f)
        if not (os.path.exists(ir_path) and os.path.exists(vis_path)):
            continue

        fused = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

        entropies.append(shannon_entropy(fused))
        ssim_ir_list.append(ssim(fused, ir, data_range=ir.max()-ir.min()))
        ssim_vis_list.append(ssim(fused, vis, data_range=vis.max()-vis.min()))
        mi_ir_list.append(mutual_information(fused, ir))
        mi_vis_list.append(mutual_information(fused, vis))
        cc_ir_list.append(np.corrcoef(fused.ravel(), ir.ravel())[0,1])
        cc_vis_list.append(np.corrcoef(fused.ravel(), vis.ravel())[0,1])

    print("Average Entropy:             {:.4f}".format(np.mean(entropies)))
    print("Average SSIM (Fused vs IR):  {:.4f}".format(np.mean(ssim_ir_list)))
    print("Average SSIM (Fused vs VIS): {:.4f}".format(np.mean(ssim_vis_list)))
    print("Average MI   (Fused vs IR):  {:.4f}".format(np.mean(mi_ir_list)))
    print("Average MI   (Fused vs VIS): {:.4f}".format(np.mean(mi_vis_list)))
    print("Average Corr (Fused vs IR):  {:.4f}".format(np.mean(cc_ir_list)))
    print("Average Corr (Fused vs VIS): {:.4f}".format(np.mean(cc_vis_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fusion results")
    parser.add_argument("--fused_dir", required=True, help="Directory of fused images")
    parser.add_argument("--ir_dir", required=True, help="Directory of registered IR images")
    parser.add_argument("--vis_dir", required=True, help="Directory of visible images")
    args = parser.parse_args()
    evaluate(args.fused_dir, args.ir_dir, args.vis_dir)
