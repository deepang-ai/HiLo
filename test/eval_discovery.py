import glob
import os


def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such directory: {path}")


def fold_sort_key(path):
    fold_name = os.path.basename(path)
    fold_id = fold_name.replace("fold_", "", 1)
    return int(fold_id) if fold_id.isdigit() else fold_name


def normalize_result_root(result_root, dataset_name):
    result_root = os.path.normpath(result_root)
    if os.path.basename(result_root) == dataset_name:
        return result_root

    dataset_result_root = os.path.join(result_root, dataset_name)
    if os.path.isdir(dataset_result_root):
        return dataset_result_root

    return result_root


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


def make_single_validation_job(result_root, trainer_name, plans_name, configuration, fold):
    fold_name = f"fold_{fold}"
    return {
        "trainer_name": trainer_name,
        "fold_name": fold_name,
        "pred_path": os.path.join(
            result_root,
            f"{trainer_name}__{plans_name}__{configuration}",
            fold_name,
            "validation",
        ),
    }
