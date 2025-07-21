import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
original_img_root = "/Users/iujeong/0.local/original_data/all_t1c"
gtv_mask_root = "/Users/iujeong/0.local/1.bet_all/b_train"
bet_mask_root = "/Users/iujeong/0.local/1.bet_all/b_train"
save_root = "/Users/iujeong/0.local/visualize/gtv_mask_overlay"
os.makedirs(save_root, exist_ok=True)
original_save_dir = os.path.join(save_root, "original")
gtv_overlay_dir = os.path.join(save_root, "gtv_overlay")
gtv_bet_overlay_dir = os.path.join(save_root, "gtv_bet_overlay")
os.makedirs(original_save_dir, exist_ok=True)
os.makedirs(gtv_overlay_dir, exist_ok=True)
os.makedirs(gtv_bet_overlay_dir, exist_ok=True)

# 시각화 함수들
def save_original_slice(img, slice_idx, save_path):
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {save_path}")
    plt.close()

def save_gtv_overlay(img, gtv, slice_idx, save_path):
    base = img[:, :, slice_idx]
    gtv_mask = gtv[:, :, slice_idx] > 0

    rgb = np.stack([base]*3, axis=-1)
    rgb = rgb / np.max(rgb)
    rgb[gtv_mask, 1] = 1  # Green channel boost

    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor='black')
    print(f"Saved: {save_path}")
    plt.close()

def save_gtv_bet_overlay(img, gtv, bet, slice_idx, save_path):
    base = img[:, :, slice_idx]
    gtv_mask = gtv[:, :, slice_idx] > 0
    bet_mask = bet[:, :, slice_idx] > 0

    rgb = np.stack([base]*3, axis=-1)
    rgb = rgb / np.max(rgb)
    rgb[gtv_mask, 1] = 1  # Green
    rgb[bet_mask, 2] = 1  # Blue

    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor='black')
    print(f"Saved: {save_path}")
    plt.close()

example_id = "BraTS-MEN-RT-0060-1"
t1c_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz"
gtv_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_gtv_mask.nii.gz"
bet_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet_mask.nii.gz"

t1c_nib = nib.load(t1c_path)
t1c = t1c_nib.get_fdata()
mid_slice = t1c.shape[2] // 2
gtv = nib.load(gtv_path).get_fdata()
bet = nib.load(bet_path).get_fdata()

save_original_slice(t1c, mid_slice, f"{original_save_dir}/{example_id}_slice_{mid_slice}.png")
save_gtv_overlay(t1c, gtv, mid_slice, f"{gtv_overlay_dir}/{example_id}_slice_{mid_slice}.png")
save_gtv_bet_overlay(t1c, gtv, bet, mid_slice, f"{gtv_bet_overlay_dir}/{example_id}_slice_{mid_slice}.png")


# Concatenate and save the three images side by side with titles
def save_concatenated_image(original_path, gtv_overlay_path, gtv_bet_overlay_path, save_path):
    import PIL.Image as Image
    from PIL import ImageDraw, ImageFont

    img1 = Image.open(original_path)
    img2 = Image.open(gtv_overlay_path)
    img3 = Image.open(gtv_bet_overlay_path)

    # Define image titles
    titles = ["Original", "GTV", "GTV + BET"]
    images = [img1, img2, img3]
    widths, heights = zip(*(i.size for i in images))

    font_size = 24
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    title_height = font_size + 10
    total_width = sum(widths) + 10 * (len(images) - 1)
    max_height = max(heights) + title_height

    new_img = Image.new("RGB", (total_width, max_height), (0, 0, 0))
    draw = ImageDraw.Draw(new_img)

    x_offset = 0
    for i, im in enumerate(images):
        new_img.paste(im, (x_offset, title_height))
        text_bbox = draw.textbbox((0, 0), titles[i], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((x_offset + (im.size[0] - text_width) // 2, 0), titles[i], fill=(255, 255, 255), font=font)
        x_offset += im.size[0] + 10  # Add gap between images

    new_img.save(save_path)
    print(f"Saved concatenated image with titles: {save_path}")

concat_save_path = os.path.join(save_root, f"{example_id}_concat.png")
original_path = f"{original_save_dir}/{example_id}_slice_{mid_slice}.png"
gtv_overlay_path = f"{gtv_overlay_dir}/{example_id}_slice_{mid_slice}.png"
gtv_bet_overlay_path = f"{gtv_bet_overlay_dir}/{example_id}_slice_{mid_slice}.png"
save_concatenated_image(original_path, gtv_overlay_path, gtv_bet_overlay_path, concat_save_path)