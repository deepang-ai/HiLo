export nnUNet_raw=path/to/CAS2023/nnUNet
export nnUNet_preprocessed=path/to/CAS2023/nnUNet/trans_data
export nnUNet_results=path/to/result
export nnUNet_compile=False

cd ../../

export CUDA_VISIBLE_DEVICES=1
python train.py 40 3d_fullres 0 --c -num_gpus 1 -tr HiLoTrainer
