import os
import json
import shutil
import random
import subprocess

base_dir = "path/to/CAS2023/"
dataset_id = 40
dataset_name = f"Dataset{dataset_id:03}_CAS2023"

out_dir = os.path.join(base_dir, f"nnUNet/{dataset_name}")

imagesTr_dir = os.path.join(out_dir, "imagesTr") # 定义训练图像目录（nnUNet要求的标准目录，存放训练集图像）
labelsTr_dir = os.path.join(out_dir, "labelsTr") # 定义训练标签目录（nnUNet要求的标准目录，存放训练集标签）

imagesTs_dir = os.path.join(out_dir, "imagesTs")
labelsTs_dir = os.path.join(out_dir, "labelsTs")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)
os.makedirs(labelsTs_dir, exist_ok=True)

os.environ["nnUNet_raw"] = os.path.join(base_dir, "nnUNet")
os.environ["nnUNet_preprocessed"] = os.path.join(
    base_dir, "nnUNet", "trans_data"
)



image_path = "data"
label_path = "mask"
image_suffix = ".nii.gz"


all_files = []  # 存储所有文件的字典，保持顺序

abs_image_path = os.path.join(base_dir, image_path)
abs_label_path = os.path.join(base_dir, label_path)

# 筛选出所有图像文件，并按前缀数字排序（保证顺序）
image_file_list = [f for f in os.listdir(abs_image_path) if f.endswith(image_suffix)]
# 按文件名前缀的数字排序（处理多位数，比如10.img.nii.gz 排在 2.img.nii.gz 后面）
image_file_list.sort(key=lambda x: int(x.split(".")[0]))

for image_file in image_file_list:
    # 提取前缀（去掉后缀）
    image_prefix = image_file[:-len(image_suffix)]
    # 拼接图像和标签的完整路径
    image_file_path = os.path.join(abs_image_path, image_file)
    label_file_path = os.path.join(abs_label_path, image_file)

    # 按字典格式添加到总列表（核心：保留你要求的字典结构）
    all_files.append({
        "original_name": image_prefix,
        "image": image_file_path,
        "label": label_file_path
    })

total_num = len(all_files)
train_size = int(total_num * 0.8)
val_size = int(total_num * 0.2)
test_size = total_num - train_size - val_size  # 处理余数

# 划分（所有元素都是字典格式）
train_img_list = all_files[:train_size]
val_img_list = all_files[train_size:train_size+val_size]
test_img_list = all_files[train_size+val_size:]

print("The number all samples:", len(train_img_list) + len(val_img_list) + len(test_img_list))
print("The number train samples:", len(train_img_list))
print("The number valid samples:", len(val_img_list))
print("The number test samples:", len(test_img_list))

train_images = []
val_images = []
test_images = []

for item in train_img_list:
    print("Processing train sample:", item)
    original_name = item["original_name"] # 获取样本原始名称
    image_src_path = item["image"] # 获取图像源文件路径
    # 定义图像目标路径（nnUNet要求单通道图像命名为"原始名称_0000.nii.gz"）
    image_dist_path = os.path.join(imagesTr_dir, f"{original_name}_0000.nii.gz")
    shutil.copyfile(image_src_path, image_dist_path)  # 复制图像到目标目录

    label_src_path = item["label"] # 获取标签源文件路径
    # 定义标签目标路径（nnUNet要求标签命名为"原始名称.nii.gz"）
    label_dist_path = os.path.join(labelsTr_dir, f"{original_name}.nii.gz")
    shutil.copyfile(label_src_path, label_dist_path) # 复制标签到目标目录

    train_images.append(original_name)

for item in val_img_list:
    print("Processing validation sample:", item)
    original_name = item["original_name"]
    image_src_path = item["image"]
    image_dist_path = os.path.join(imagesTr_dir, f"{original_name}_0000.nii.gz")
    shutil.copyfile(image_src_path, image_dist_path)

    label_src_path = item["label"]
    label_dist_path = os.path.join(labelsTr_dir, f"{original_name}.nii.gz")
    shutil.copyfile(label_src_path, label_dist_path)

    val_images.append(original_name)

for item in test_img_list:
    print("Processing test sample:", item)
    original_name = item["original_name"]
    image_src_path = item["image"]
    image_dist_path = os.path.join(imagesTs_dir, f"{original_name}_0000.nii.gz")
    shutil.copyfile(image_src_path, image_dist_path)

    label_src_path = item["label"]
    label_dist_path = os.path.join(labelsTs_dir, f"{original_name}.nii.gz")
    shutil.copyfile(label_src_path, label_dist_path)

    test_images.append(original_name)

assert len(train_images) == 80
assert len(val_images) == 20
assert len(test_images) == 0

# 定义dataset.json内容（nnUNet必需的数据集元信息配置文件）
dataset_json = {
    "channel_names": {   # 定义图像通道名称（此处为单通道CT）
        "0": "ImageCAS",
    },
    "labels": {  # 定义标签类别（背景为0，气道为1）
        "background": 0,
        "vessel": 1,
    },
    "numTraining": len(train_images) + len(val_images), # 总训练样本数（训练集+验证集）
    "file_ending": ".nii.gz", # 数据文件的后缀名
}

# 将dataset.json写入输出目录
with open(os.path.join(out_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

# 定义splits_final.json内容（记录训练集和验证集的划分）
splits_final_json = [{"train": train_images, "val": val_images, "test": test_images}]
print("The number of train samples:", len(train_images))
print("The number of valid samples:", len(val_images))
print("The number of test samples:", len(test_images))
print(splits_final_json)
# 将划分信息写入输出目录
with open(os.path.join(out_dir, "splits_final.json"), "w") as f:
    json.dump(splits_final_json, f, indent=4)

# 调用nnUNet命令：提取数据集特征指纹（用于后续计划生成）
# -d 指定数据集ID（3位数字格式）
subprocess.call(["nnUNetv2_extract_fingerprint", "-d", f"{dataset_id:03}"])
# 调用nnUNet命令：生成实验计划（包括网络配置、数据预处理参数等）
subprocess.call(["nnUNetv2_plan_experiment", "-d", f"{dataset_id:03}"])


plan_path = os.path.join(
    os.getenv("nnUNet_preprocessed"),
    dataset_name,
    "nnUNetPlans.json",
)
with open(plan_path, "r") as f:
    plans_json = json.load(f)

plans_json["configurations"]["3d_fullres"]["patch_size"] = [96, 96, 96]

with open(plan_path, "w") as f:
    json.dump(plans_json, f, indent=4)


subprocess.call(
    ["nnUNetv2_preprocess", "-d", f"{dataset_id:03}", "-c", "3d_fullres", "-np", "2"]
)

print("Done!")
