#!/usr/bin/env python3
# scripts/register_to_mni.py
import os
import glob
import argparse
import ants
import nibabel as nib
import numpy as np

# nilearn은 선택적으로 사용(템플릿이 없을 때만)
try:
    from nilearn import datasets
except Exception:
    datasets = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def ensure_mni_template(mni_path: str, resolution: int = 1):
    """
    mni_path 에 MNI 템플릿이 없으면 nilearn으로 내려받아 저장.
    """
    if os.path.exists(mni_path):
        print(f"[MNI] Found template: {mni_path}")
        return mni_path

    if datasets is None:
        raise RuntimeError(
            f"[MNI] Template not found and nilearn is unavailable. "
            f"Please place MNI152_T1_{resolution}mm at: {mni_path}"
        )

    print(f"[MNI] Not found. Downloading MNI152 T1 {resolution}mm via nilearn...")
    ensure_dir(os.path.dirname(mni_path))
    mni_img = datasets.load_mni152_template(resolution=resolution)  # Nifti1Image
    nib.save(mni_img, mni_path)
    print(f"[MNI] Saved: {mni_path}")
    return mni_path


def warp_one_subject(
    norm_path: str,
    mni_path: str,
    out_dir: str,
    use_syn: str = "SyN",
    pred_dir: str | None = None,
):
    """
    norm_path: .../{ID}_norm.nii.gz
    mni_path : .../MNI152_T1_1mm.nii.gz
    pred_dir : .../pred_nii (선택) — {ID}_pred_mask.nii.gz / {ID}_pred_prob.nii.gz
    """
    sid = os.path.basename(norm_path).replace("_norm.nii.gz", "")
    print(f"\n[REG] Subject: {sid}")

    # Read images
    mni = ants.image_read(mni_path)
    mov = ants.image_read(norm_path)

    # Run registration (rigid+affine+SyN 포함 preset)
    reg = ants.registration(fixed=mni, moving=mov, type_of_transform=use_syn)

    # Save warped intensity image (linear interp by default inside registration)
    ants.image_write(reg["warpedmovout"], os.path.join(out_dir, f"{sid}_norm_mni.nii.gz"))

    def warp_mask_from_suffix(suffix: str, out_suffix: str):
        """
        대상: GT/BET 등 norm_path와 동일 prefix를 갖는 마스크
        """
        src = norm_path.replace("_norm.nii.gz", suffix)
        if not os.path.exists(src):
            return False
        mask_img = ants.image_read(src)
        warped = ants.apply_transforms(
            fixed=mni,
            moving=mask_img,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",  # 마스크류는 NN
        )
        ants.image_write(warped, os.path.join(out_dir, f"{sid}{out_suffix}_mni.nii.gz"))
        return True

    # GT / BET mask
    got_gt = warp_mask_from_suffix("_gtv_mask.nii.gz", "_gtv_mask")
    got_bet = warp_mask_from_suffix("_bet_mask.nii.gz", "_bet_mask")
    print(f"[REG]   GT mask: {'ok' if got_gt else 'missing'}")
    print(f"[REG]   BET mask: {'ok' if got_bet else 'missing'}")

    # Pred NIfTI (optional)
    if pred_dir:
        pred_mask_p = os.path.join(pred_dir, f"{sid}_pred_mask.nii.gz")
        pred_prob_p = os.path.join(pred_dir, f"{sid}_pred_prob.nii.gz")

        if os.path.exists(pred_mask_p):
            pm = ants.image_read(pred_mask_p)
            pm_warp = ants.apply_transforms(
                fixed=mni,
                moving=pm,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            ants.image_write(pm_warp, os.path.join(out_dir, f"{sid}_pred_mask_mni.nii.gz"))
            print("[REG]   pred_mask -> ok")
        else:
            print("[REG]   pred_mask -> missing")

        if os.path.exists(pred_prob_p):
            pp = ants.image_read(pred_prob_p)
            pp_warp = ants.apply_transforms(
                fixed=mni,
                moving=pp,
                transformlist=reg["fwdtransforms"],
                interpolator="linear",  # 확률맵은 linear
            )
            ants.image_write(pp_warp, os.path.join(out_dir, f"{sid}_pred_prob_mni.nii.gz"))
            print("[REG]   pred_prob -> ok")
        else:
            print("[REG]   pred_prob -> missing")


def main():
    parser = argparse.ArgumentParser(description="Register subject volumes to MNI space and warp masks.")
    parser.add_argument(
        "--mni-template",
        default="/Users/iujeong/0.local/mni/MNI152_T1_1mm.nii.gz",
        help="Path to MNI152 T1 1mm template.",
    )
    parser.add_argument(
        "--in-dir",
        default="/Users/iujeong/0.local/3.normalize/n_test",
        help="Input directory containing *_norm.nii.gz (and *_gtv_mask.nii.gz, *_bet_mask.nii.gz).",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Optional directory containing {ID}_pred_mask.nii.gz and/or {ID}_pred_prob.nii.gz.",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/iujeong/0.local/3.normalize/mni_mapped",
        help="Output directory for MNI-warped files.",
    )
    parser.add_argument(
        "--transform",
        default="SyN",
        choices=["SyN", "SyNRA", "SyNOnly", "Affine", "Rigid", "ElasticSyN"],
        help="ANTs registration preset (default: SyN).",
    )
    parser.add_argument(
        "--download-mni",
        action="store_true",
        help="If set, will download/save MNI template to --mni-template path if missing.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 템플릿 준비
    if args.download_mni:
        ensure_mni_template(args.mni_template, resolution=1)
    elif not os.path.exists(args.mni_template):
        raise FileNotFoundError(
            f"[MNI] Template not found: {args.mni_template}\n"
            f"      (or run with --download-mni to fetch via nilearn)"
        )

    # 작업 대상 스캔
    norm_list = sorted(glob.glob(os.path.join(args.in_dir, "*_norm.nii.gz")))
    if not norm_list:
        raise RuntimeError(f"[REG] No *_norm.nii.gz found under: {args.in_dir}")

    print(f"[REG] Found {len(norm_list)} subjects.")
    for i, norm_path in enumerate(norm_list, 1):
        print(f"[REG] ({i}/{len(norm_list)}) {os.path.basename(norm_path)}")
        warp_one_subject(
            norm_path=norm_path,
            mni_path=args.mni_template,
            out_dir=args.out_dir,
            use_syn=args.transform,
            pred_dir=args.pred_dir,
        )

    print("\n[REG] Done.")


if __name__ == "__main__":
    main()