# /Users/iujeong/0.local/scripts/save_overlays.py
import os, glob, numpy as np, nibabel as nib, matplotlib.pyplot as plt

IN_DIRS = [
  "/Users/iujeong/0.local/3.normalize/mni_mapped_train",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_val",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_test",
]
OUT_DIR = "/Users/iujeong/0.local/6.report/overlays"
os.makedirs(OUT_DIR, exist_ok=True)

def central_z(mask):
    idx = np.where(mask>0)
    return int(np.median(idx[2])) if len(idx[0])>0 else mask.shape[2]//2

for d in IN_DIRS:
    for img_p in glob.glob(os.path.join(d,"*_norm_mni.nii.gz")):
        sid = os.path.basename(img_p).replace("_norm_mni.nii.gz","")
        gt_p = os.path.join(d, f"{sid}_gtv_mask_mni.nii.gz")
        pr_p = os.path.join(d, f"{sid}_pred_mask_mni.nii.gz")
        if not (os.path.exists(gt_p) and os.path.exists(pr_p)): continue

        img = nib.load(img_p).get_fdata()
        gt  = nib.load(gt_p).get_fdata()>0
        pr  = nib.load(pr_p).get_fdata()>0
        z = central_z(gt)

        a = img[..., z]
        a = (a - np.min(a)) / (np.ptp(a) + 1e-8)
        ov = np.dstack([a,a,a])
        # GT=red, Pred=green
        ov[gt[...,z], :] = [1,0,0]
        ov[np.logical_and(~gt[...,z], pr[...,z]), :] = [0,1,0]

        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(a, cmap="gray"); axs[0].set_title(f"{sid} (z={z})"); axs[0].axis("off")
        axs[1].imshow(a, cmap="gray"); axs[1].imshow(ov, alpha=0.4); axs[1].set_title("overlay"); axs[1].axis("off")
        out = os.path.join(OUT_DIR, f"{sid}_overlay_z{z}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("saved:", out)
