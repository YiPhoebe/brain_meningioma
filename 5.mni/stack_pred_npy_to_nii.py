# /stack_pred_from_slices.py
import os, glob, re, shutil
import numpy as np
import nibabel as nib
from collections import defaultdict
from scipy.ndimage import zoom

ROOT_NORM   = "/Users/iujeong/0.local/3.normalize"   # n_train/n_val/n_test
ROOT_SLICES = "/Users/iujeong/0.local/4.slice"       # s_train/s_val/s_test/npy/*.npy
OUT_ROOT    = "/Users/iujeong/0.local/5.mni"         # m_train/m_val/m_test

SPLITS = ["train", "val", "test"]
MAP_IN_N  = {sp: f"n_{sp}" for sp in SPLITS}
MAP_IN_S  = {sp: f"s_{sp}" for sp in SPLITS}
MAP_OUT_M = {sp: f"m_{sp}" for sp in SPLITS}

IS_PROB = True
BIN_THR = 0.5

# ex) BraTS-...-1_slice_092_pred.npy  →  (id, z, kind) = ("BraTS-...-1", 92, "pred")
PAT1 = re.compile(r"^(?P<id>.+?)_slice_(?P<z>\d+)_(?P<kind>img|mask|pred)$")
# fallback: BraTS-...-1_092_pred.npy
PAT2 = re.compile(r"^(?P<id>.+?)_(?P<z>\d+)_(?P<kind>img|mask|pred)$")

def parse_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = PAT1.match(base) or PAT2.match(base)
    if m:
        return m.group("id"), int(m.group("z")), m.group("kind")
    # 최후의 수단: 끝 숫자/끝 토큰 추출
    zs = re.findall(r'\d+', base)
    kind = "pred" if base.endswith("_pred") else "img" if base.endswith("_img") else "mask" if base.endswith("_mask") else ""
    id_guess = base
    if "_slice_" in base:
        id_guess = base.split("_slice_")[0]
    elif kind and base.endswith("_"+kind) and zs:
        id_guess = base[: base.rfind("_"+kind)]
    z = int(zs[-1]) if zs else -1
    return id_guess, z, kind

def safe_copy(src, dst_dir):
    if not os.path.exists(src): return False, None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return True, dst

def build_index_norm(in_dir_n):
    idx = {}
    for norm_p in glob.glob(os.path.join(in_dir_n, "*_norm.nii.gz")):
        sid = os.path.basename(norm_p).replace("_norm.nii.gz","")
        idx[sid] = {
            "norm": norm_p,
            "gtv": norm_p.replace("_norm.nii.gz","_gtv_mask.nii.gz"),
            "bet": norm_p.replace("_norm.nii.gz","_bet_mask.nii.gz"),
        }
    return idx

def stack_and_save(sid, slice_paths, ref_nii, out_dir_m, is_prob=True, bin_thr=0.5):
    H, W, Z = ref_nii.shape

    # 1) 파일명에서 z 인덱스 파싱
    def parse_z(p):
        b = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"_slice_(\d+)_", b) or re.search(r"_(\d+)_(img|mask|pred)$", b)
        return int(m.group(1)) if m else -1

    # 2) 빈 볼륨 만들고, 각 슬라이스를 '그 z 위치'에 배치
    vol = np.zeros((H, W, Z), dtype=np.float32)
    for p in slice_paths:
        z = parse_z(p)
        sl = np.load(p).astype(np.float32)  # (h, w)
        # XY 리샘플 (예: 160x192 → HxW)
        sy, sx = H / sl.shape[0], W / sl.shape[1]
        sl_rs = zoom(sl, (sy, sx), order=1)  # 확률은 선형
        if 0 <= z < Z:
            vol[..., z] = sl_rs  # 정확한 z에 꽂기
        else:
            # z가 범위 밖이면 스킵(필요시 클램프 가능)
            continue

    # 3) 저장
    pred_prob_path = os.path.join(out_dir_m, f"{sid}_pred_prob.nii.gz")
    nib.save(nib.Nifti1Image(vol, ref_nii.affine, header=ref_nii.header), pred_prob_path)

    pred_bin = (vol >= bin_thr).astype(np.uint8) if is_prob else (vol > 0.5).astype(np.uint8)
    pred_mask_path = os.path.join(out_dir_m, f"{sid}_pred_mask.nii.gz")
    nib.save(nib.Nifti1Image(pred_bin, ref_nii.affine, header=ref_nii.header), pred_mask_path)

def main():
    for sp in SPLITS:
        in_dir_n  = os.path.join(ROOT_NORM,  MAP_IN_N[sp])
        in_dir_s  = os.path.join(ROOT_SLICES, MAP_IN_S[sp], "npy")
        out_dir_m = os.path.join(OUT_ROOT,    MAP_OUT_M[sp])
        os.makedirs(out_dir_m, exist_ok=True)

        norm_idx = build_index_norm(in_dir_n)
        if not norm_idx:
            print(f"[warn] no *_norm.nii.gz under {in_dir_n}")
            continue

        flat_files = glob.glob(os.path.join(in_dir_s, "*.npy"))
        if not flat_files:
            print(f"[warn] no slices under {in_dir_s}")
            continue

        # ID별로 pred만 버킷팅
        buckets = defaultdict(list)
        for f in flat_files:
            sid, z, kind = parse_name(f)
            if kind != "pred":  # pred만 쓸 거야
                continue
            buckets[sid].append(f)

        print(f"[info] split={sp}: {len(buckets)} IDs with preds")

        for sid, pred_paths in sorted(buckets.items()):
            if sid not in norm_idx:
                print(f"[skip] {sid}: not found in {in_dir_n}")
                continue
            if not pred_paths:
                print(f"[skip] {sid}: empty pred list")
                continue

            ref_nii = nib.load(norm_idx[sid]["norm"])
            # 스택 & 저장
            stack_and_save(sid, pred_paths, ref_nii, out_dir_m)

            # GT/norm/BET도 같이 복사
            ok_norm, norm_out = safe_copy(norm_idx[sid]["norm"], out_dir_m)
            ok_gt,   gt_out   = safe_copy(norm_idx[sid]["gtv"],  out_dir_m)
            ok_bet,  bet_out  = safe_copy(norm_idx[sid]["bet"],  out_dir_m)

            # 간단 무결성 체크
            try:
                if ok_gt and norm_out:
                    _norm = nib.load(norm_out); _gt = nib.load(gt_out)
                    if _norm.shape != _gt.shape or not np.allclose(_norm.affine, _gt.affine):
                        print(f"[warn] {sid}: norm↔gt mismatch (shape/affine)")
                if ok_bet and norm_out:
                    _norm = nib.load(norm_out); _bet = nib.load(bet_out)
                    if _norm.shape != _bet.shape or not np.allclose(_norm.affine, _bet.affine):
                        print(f"[warn] {sid}: norm↔bet mismatch (shape/affine)")
            except Exception as e:
                print(f"[warn] {sid}: integrity check failed — {e}")

            print(f"[ok] {sp}:{sid} → {out_dir_m}")

if __name__ == "__main__":
    main()