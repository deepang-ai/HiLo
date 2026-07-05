# -*- coding: utf-8 -*-

'''
Program :   The evaluation functions for the ATM 22 Challenge, including the TD / BD / DSC / Precision
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   evaluation_atm22.py
Date    :   2022/02/02 16:19
Version :   V1.0 (Modified: remove Branch/Length, add ASD & HD95)
'''
import os
import numpy as np
import nibabel
import glob
from scipy import ndimage
import csv
import argparse

# ========================
# 新增：ASD 和 HD95 计算
# ========================

def compute_surface(mask, connectivity=1):
    """
    提取二值掩码的表面（边界体素）。
    表面 = 前景 - 内部（通过腐蚀）
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, connectivity)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    surface = mask.astype(bool) & (~eroded)
    return surface

def crop_to_foreground(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, dim) for dim in mask.shape)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(min_v), int(max_v)) for min_v, max_v in zip(mins, maxs))


def _surface_distances(source_surface, target_surface, spacing):
    if not source_surface.any() or not target_surface.any():
        return np.array([])
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance_map[source_surface]


def surface_metrics_calculation(pred, label, spacing=(1.0, 1.0, 1.0)):
    pred = pred.astype(bool)
    label = label.astype(bool)

    if not pred.any() or not label.any():
        return np.nan, np.nan

    crop = crop_to_foreground(pred | label)
    surf_pred = compute_surface(pred[crop])
    surf_label = compute_surface(label[crop])

    dist_pred_to_label = _surface_distances(surf_pred, surf_label, spacing)
    dist_label_to_pred = _surface_distances(surf_label, surf_pred, spacing)
    all_dists = np.concatenate([dist_pred_to_label, dist_label_to_pred])
    if all_dists.size == 0:
        return np.nan, np.nan

    asd = round((dist_pred_to_label.mean() + dist_label_to_pred.mean()) / 2.0, 3)
    hd95 = round(np.percentile(all_dists, 95), 3)
    return asd, hd95


def asd_calculation(pred, label, spacing=(1.0, 1.0, 1.0)):
    """
    Average Surface Distance (ASD)
    """
    return surface_metrics_calculation(pred, label, spacing=spacing)[0]


def hd95_calculation(pred, label, spacing=(1.0, 1.0, 1.0)):
    """
    95th percentile Hausdorff Distance (HD95)
    """
    return surface_metrics_calculation(pred, label, spacing=spacing)[1]

# ========================
# 保留原有其他指标函数
# ========================

def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score

def false_positive_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr

def false_negative_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr

def sensitivity_calculation(pred, label):
    sensitivity = round(100 - false_negative_rate_calculation(pred, label), 3)
    return sensitivity

def specificity_calculation(pred, label):
    specificity = round(100 - false_positive_rate_calculation(pred, label), 3)
    return specificity

def precision_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precision

def alr_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp_volume = np.sum(pred - pred * label) + smooth
    gt_volume = np.sum(label) + smooth
    alr = round((fp_volume / gt_volume) * 100, 3)
    return alr

def amr_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn_volume = np.sum(label - pred * label) + smooth
    gt_volume = np.sum(label) + smooth
    amr = round((fn_volume / gt_volume) * 100, 3)
    return amr

def iou_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    fp = np.sum(pred - pred * label) + smooth
    fn = np.sum(label - pred * label) + smooth
    iou = round((tp / (tp + fp + fn)) * 100, 3)
    return iou



# ========================
# 主评估函数（修改版）
# ========================

def evaluation(label_path, pred_path, csv_path):
    filelist_pred = sorted(glob.glob(os.path.join(pred_path, "*.nii.gz")))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
    fold_name = os.path.basename(os.path.dirname(csv_path))

    dices = []
    ious = []
    tprs = []
    fprs = []
    spes = []
    amrs = []
    precisions = []
    asds = []
    hd95s = []

    results_per_case = []

    for pred_file in filelist_pred:
        print("Processing file:", pred_file)
        name = os.path.basename(pred_file)[0:-7]

        label_name = name + ".nii.gz"
        # parsing_name = name + "_parse.nii.gz"  # 不再使用
        # skeleton_name = name + "_skel.nii.gz"  # 不再使用

        label = nibabel.load(os.path.join(label_path, label_name))
        pred = nibabel.load(pred_file)

        spacing_xyz = pred.header.get_zooms()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # 注意顺序：Z,Y,X

        label = (np.asanyarray(label.dataobj) > 0).astype(np.uint8)
        pred = (np.asanyarray(pred.dataobj) > 0).astype(np.uint8)

        if pred.sum() == 0:
            print("All black prediction!")
            cur_dice = 0.0
            cur_iou = 0.0
            cur_tpr = 0.0
            cur_fpr = 0.0
            cur_pre = 0.0
            cur_spe = 0.0
            cur_amr = 100.0
            cur_asd = np.nan
            cur_hd95 = np.nan
        else:


            cur_dice = dice_coefficient_score_calculation(pred, label)
            cur_iou = iou_calculation(pred, label)
            cur_tpr = sensitivity_calculation(pred, label)
            cur_fpr = false_positive_rate_calculation(pred, label)
            cur_pre = precision_calculation(pred, label)
            cur_spe = specificity_calculation(pred, label)
            cur_amr = amr_calculation(pred, label)
            cur_asd, cur_hd95 = surface_metrics_calculation(pred, label, spacing=spacing_zyx)

        # 收集结果
        dices.append(cur_dice)
        ious.append(cur_iou)
        tprs.append(cur_tpr)
        fprs.append(cur_fpr)
        spes.append(cur_spe)
        amrs.append(cur_amr)
        precisions.append(cur_pre)
        asds.append(cur_asd)
        hd95s.append(cur_hd95)

        results_per_case.append({
            "case": name,
            "dataset": dataset_name,
            "fold": fold_name,
            "class": "vessel",
            "Dice (%)": cur_dice,
            "IoU (%)": cur_iou,
            "Sensitivity_TPR (%)": cur_tpr,
            "Specificity (%)": cur_spe,
            "FPR (%)": cur_fpr,
            "AMR (%)": cur_amr,
            "Precision (%)": cur_pre,
            "ASD (mm)": cur_asd,
            "HD95 (mm)": cur_hd95
        })

        print(
            name,
            "Dice: %0.4f" % (cur_dice),
            "IoU: %0.4f" % (cur_iou),
            "TPR: %0.4f" % (cur_tpr),
            "Spe: %0.4f" % (cur_spe),
            "FPR: %0.4f" % (cur_fpr),
            "AMR: %0.4f" % (cur_amr),
            "Pre: %0.4f" % (cur_pre),
            "ASD: %s" % ("nan" if np.isnan(cur_asd) else f"{cur_asd:.4f}"),
            "HD95: %s" % ("nan" if np.isnan(cur_hd95) else f"{cur_hd95:.4f}")
        )

    # 统计（忽略 NaN）
    def safe_mean(arr):
        arr = np.array(arr)
        return np.nanmean(arr) if np.any(~np.isnan(arr)) else np.nan

    def safe_std(arr):
        arr = np.array(arr)
        return np.nanstd(arr) if np.any(~np.isnan(arr)) else np.nan

    dice_mean, dice_std = safe_mean(dices), safe_std(dices)
    iou_mean, iou_std = safe_mean(ious), safe_std(ious)
    tpr_mean, tpr_std = safe_mean(tprs), safe_std(tprs)
    fpr_mean, fpr_std = safe_mean(fprs), safe_std(fprs)
    spe_mean, spe_std = safe_mean(spes), safe_std(spes)
    amr_mean, amr_std = safe_mean(amrs), safe_std(amrs)
    pre_mean, pre_std = safe_mean(precisions), safe_std(precisions)
    asd_mean, asd_std = safe_mean(asds), safe_std(asds)
    hd95_mean, hd95_std = safe_mean(hd95s), safe_std(hd95s)

    print(
        "dice: %0.4f (%0.4f), iou: %0.4f (%0.4f), tpr: %0.4f (%0.4f), "
        "fpr: %0.4f (%0.4f), spe: %0.4f (%0.4f), amr: %0.4f (%0.4f), "
        "pre: %0.4f (%0.4f), asd: %0.4f (%0.4f), hd95: %0.4f (%0.4f)" % (
            dice_mean, dice_std,
            iou_mean, iou_std,
            tpr_mean, tpr_std,
            fpr_mean, fpr_std,
            spe_mean, spe_std,
            amr_mean, amr_std,
            pre_mean, pre_std,
            asd_mean, asd_std,
            hd95_mean, hd95_std
        )
    )

    # 保存 CSV
    fieldnames = [
        "case",
        "dataset",
        "fold",
        "class",
        "Dice (%)",
        "IoU (%)",
        "Sensitivity_TPR (%)",
        "Specificity (%)",
        "FPR (%)",
        "AMR (%)",
        "Precision (%)",
        "ASD (mm)",
        "HD95 (mm)"
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_per_case:
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    formatted_row[k] = "nan"
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)

    print(f"\n✅ Per-case metrics saved to: {csv_path}")

# ========================
# 主程序入口
# ========================

def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such directory: {path}")


def fold_sort_key(path):
    fold_name = os.path.basename(path)
    fold_id = fold_name.replace("fold_", "", 1)
    return int(fold_id) if fold_id.isdigit() else fold_name


def discover_validation_jobs(result_root, plans_name, configuration):
    trainer_pattern = os.path.join(result_root, f"*__{plans_name}__{configuration}")
    jobs = []

    for trainer_dir in sorted(glob.glob(trainer_pattern)):
        if not os.path.isdir(trainer_dir):
            continue

        trainer_name = os.path.basename(trainer_dir).split("__", 1)[0]
        fold_dirs = [
            fold_dir for fold_dir in glob.glob(os.path.join(trainer_dir, "fold_*"))
            if os.path.isdir(fold_dir)
        ]

        for fold_dir in sorted(fold_dirs, key=fold_sort_key):
            validation_dir = os.path.join(fold_dir, "validation")
            checkpoint_final = os.path.join(fold_dir, "checkpoint_final.pth")
            fold_name = os.path.basename(fold_dir)
            if not os.path.exists(checkpoint_final):
                print(f"[skip] {trainer_name} {fold_name}: checkpoint_final.pth not found")
                continue
            if not os.path.isdir(validation_dir):
                print(f"[skip] {trainer_name} {fold_name}: no validation dir")
                continue
            if not glob.glob(os.path.join(validation_dir, "*.nii.gz")):
                print(f"[skip] {trainer_name} {fold_name}: no prediction files")
                continue

            jobs.append(
                {
                    "trainer_name": trainer_name,
                    "fold_name": fold_name,
                    "pred_path": validation_dir,
                }
            )

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageCAS validation predictions")
    parser.add_argument("--tr", type=str, default=None, help="Trainer name. If omitted, discover all validation jobs")
    parser.add_argument("--id", type=int, default=140, help="dataset ID, default: 140")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate when --tr is specified")
    parser.add_argument("--plans", type=str, default="nnUNetPlans", help="nnUNet plans name")
    parser.add_argument("--config", type=str, default="3d_fullres", help="nnUNet configuration name")
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Ground-truth label directory. Default: /datasets/public/GZHU/ImageCAS/nnUNet/<dataset>/labelsTr",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=None,
        help="Dataset result directory. Default: result/<dataset>",
    )
    args = parser.parse_args()

    dataset_id = args.id
    dataset = "ImageCAS"
    dataset_name = f"Dataset{dataset_id:03}_{dataset}"
    label_path = args.label_path or f"/datasets/public/GZHU/{dataset}/nnUNet/{dataset_name}/labelsTr"
    result_root = args.result_root or os.path.join("result", dataset_name)
    metrics_root = os.path.join("metrics", dataset_name)

    check_path_exists(label_path)
    check_path_exists(result_root)

    if args.tr is None:
        jobs = discover_validation_jobs(result_root, args.plans, args.config)
        if len(jobs) == 0:
            raise FileNotFoundError(f"No validation predictions found under: {result_root}")
    else:
        fold_name = f"fold_{args.fold}"
        jobs = [
            {
                "trainer_name": args.tr,
                "fold_name": fold_name,
                "pred_path": os.path.join(
                    result_root,
                    f"{args.tr}__{args.plans}__{args.config}",
                    fold_name,
                    "validation",
                ),
            }
        ]

    print(f"Dataset: {dataset_name}")
    print(f"Label path: {label_path}")
    print(f"Result root: {result_root}")
    print(f"Found validation jobs: {len(jobs)}")

    for job in jobs:
        trainer_name = job["trainer_name"]
        fold_name = job["fold_name"]
        pred_path = job["pred_path"]
        csv_dir = os.path.join(metrics_root, fold_name)
        csv_path = os.path.join(csv_dir, f"{trainer_name}.csv")

        if os.path.exists(csv_path):
            print(f"[skip] {trainer_name} {fold_name}: metrics already exist at {csv_path}")
            continue

        check_path_exists(pred_path)
        os.makedirs(csv_dir, exist_ok=True)

        print("=" * 80)
        print("Trainer name:", trainer_name)
        print("Fold:", fold_name)
        print("Prediction path:", pred_path)
        print("CSV path:", csv_path)

        evaluation(label_path, pred_path, csv_path)
