import os
import json
from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

# XML �ㅼ엫�ㅽ럹�댁뒪
ns = {'ns': 'http://www.nih.gov'}

def resample_image(image, out_spacing=(1.0, 1.0, 3.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_size = [
        int(round(original_size[i] * (original_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)

def parse_xml(root, origin, org_spacing, spacing, z_spacing=3.0):

    slice_data = defaultdict(lambda: defaultdict(list))  # slice_idx �� reader �� list of {score, polygon}

    for sess_idx, session in enumerate(root.findall('.//ns:readingSession', ns)):
        reader_id = f"reader_{sess_idx}"

        for nodule in session.findall('ns:unblindedReadNodule', ns):
            nid = nodule.find('ns:noduleID', ns).text
            char = nodule.find('ns:characteristics', ns)
            if char is not None:
                m = char.find('ns:malignancy', ns)
                if m is not None and m.text is not None:
                    mal_score = int(m.text)
                else:
                    continue
            else:
                continue

            for roi in nodule.findall('ns:roi', ns):
                z = roi.find('ns:imageZposition', ns)
                if z is None:
                    continue
                z_pos = float(z.text)
                z_idx = int(round((z_pos - origin[2]) / z_spacing))
                # 由ъ깦�뚮쭅�섎㈃�� z_idx媛� 媛숈� 寃껋쓣 嫄몃윭�댁� �딄퀬, �꾨� 異붽�

                poly = []
                for edge in roi.findall('ns:edgeMap', ns):
                    x = float(edge.find('ns:xCoord', ns).text)
                    y = float(edge.find('ns:yCoord', ns).text)
                    x_pix = round(x * org_spacing[0] / spacing[0])
                    y_pix = round(y * org_spacing[1] / spacing[1])
                    poly.append([x_pix, y_pix])

                if len(poly) >= 3:
                    slice_key = f"slice_{z_idx}"
                    entry = {
                        "noduleID": nid,
                        "score": mal_score if mal_score is not None else -1,
                        "polygon": poly
                    }
                    slice_data[slice_key][reader_id].append(entry)

    return slice_data

def segment_matching(unmatched_polygons, xy_thr):
    matched_segment = []
    centroids = []
    matched_flag = [False] * len(unmatched_polygons)

    # 1. centroid 誘몃━ 怨꾩궛
    for nodule in unmatched_polygons:
        coords = np.array(nodule['polygon'])
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)

    # 2. 留ㅼ묶 �섑뻾
    for i, c1 in enumerate(centroids):
        if matched_flag[i]:
            continue

        base = unmatched_polygons[i].copy()
        base['score'] = [base['score']]  # score瑜� list濡� 珥덇린��
        matched_flag[i] = True

        for j in range(i + 1, len(unmatched_polygons)):
            if matched_flag[j]:
                continue

            c2 = centroids[j]
            if euclidean(c1, c2) <= xy_thr:
                matched_flag[j] = True
                score_j = unmatched_polygons[j]['score']
                if score_j not in base['score']:
                    base['score'].append(score_j) # score �꾩쟻

        matched_segment.append(base)

    return matched_segment

def save_png_with_overlay(image, dict_segment_score, save_dir, save_png_dir):

    img_array = sitk.GetArrayFromImage(image)  # shape: (Z, Y, X)

    for slice_num in dict_segment_score:
        z_idx = int(slice_num.split('_')[-1])
        img = img_array[z_idx]

        fig, axes = plt.subplots(1,2, figsize=(10, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')

        polygons = []
        for reader in dict_segment_score[slice_num]:
            polygons.extend(dict_segment_score[slice_num][reader])

        # segment 留ㅼ묶 �뚭퀬由ъ쬁 (以묐났�섎뒗 �멸렇癒쇳듃 �쒓굅)
        polygons_matched = segment_matching(polygons, xy_thr=5)

        for i, poly in enumerate(polygons_matched):

            # x_pix = (x - origin[0]) / spacing[0]  # �� X異� index
            # y_pix = (y - origin[1]) / spacing[1]

            patch = patches.Polygon(poly['polygon'], closed=True, edgecolor='red', fill=False, linewidth=.5)
            axes[1].add_patch(patch)

            mal_score = max(poly['score'])

        axes[1].set_title('Overlay')
        axes[1].axis('off')

        fig.tight_layout(pad=0)
        os.makedirs(f'{save_png_dir}/{subj_id}', exist_ok=True)
        fig.savefig(f'{save_png_dir}/{subj_id}/slice_{z_idx:03d}_{mal_score}.png', bbox_inches=None, pad_inches=0)
        plt.close(fig)

        os.makedirs(f'{save_dir}/{subj_id}', exist_ok=True)
        np.save(f'{save_dir}/{subj_id}/slice_{z_idx:03d}_{mal_score}.npy', img)


def process(subj_id, dicom_dir, root, nii_save_dir, save_dir, save_png_dir):

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir))
    image = reader.Execute()
    resampled = resample_image(image, out_spacing=(1.0, 1.0, 3.0))

    sitk.WriteImage(resampled, f'{nii_save_dir}/{subj_id}.nii.gz')

    origin = resampled.GetOrigin()
    spacing = resampled.GetSpacing()
    org_spacing = image.GetSpacing()

    dict_seg_scores = parse_xml(root, origin, org_spacing, spacing, z_spacing=3.0)
    save_png_with_overlay(resampled, dict_seg_scores, save_dir, save_png_dir)
    # print(resampled.GetSize())
    return dict_seg_scores


df = pd.read_csv('/home/jinsu/PycharmProjects/lidc-idri/metadata.csv')
df = df[df["Number of Images"] > 10].copy()
subj_list = sorted(df['Subject ID'].unique())

xml_dir = "./LIDC-XML-only/tcia-lidc-xml"
xml_paths = glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)

data_root = "/home/jinsu/PycharmProjects/lidc-idri/manifest-1600709154662/"
save_root = "/home/jinsu/PycharmProjects/lidc-idri/resampled_nii/"
slice_save_root = "/home/jinsu/PycharmProjects/lidc-idri/slices/"
slice_png_save_root = "/home/jinsu/PycharmProjects/lidc-idri/slices_png/"
os.makedirs(save_root, exist_ok=True)
os.makedirs(slice_save_root, exist_ok=True)
os.makedirs(slice_png_save_root, exist_ok=True)

dict_annotation = {}
for xml in tqdm(xml_paths):

    tree = ET.parse(xml)
    root = tree.getroot()
    element = root.find('.//ns:SeriesInstanceUid', ns)
    if element is None:
        continue

    uid = element.text
    if uid not in df['Series UID'].values:
        continue

    subj_id = df.loc[df['Series UID'] == uid, 'Subject ID'].values[0]
    if subj_id in dict_annotation:
        subj_id = f'{subj_id}_1'
    dicom_dir = data_root + df.loc[df['Series UID'] == uid, 'File Location'].values[0].replace('\\', '/')

    dict_polygons_scores = process(
        subj_id=subj_id,
        dicom_dir=dicom_dir,
        root=root,
        nii_save_dir= f'{save_root}',
        save_dir=f'{slice_save_root}',
        save_png_dir=f'{slice_png_save_root}')
    if not bool(dict_polygons_scores.keys()):
        continue

    dict_annotation[subj_id] = dict_polygons_scores

with open("/home/jinsu/PycharmProjects/lidc-idri/nodule_malignancy_scores.json", 'w') as f:
    json.dump(dict_annotation, f, indent=4)
