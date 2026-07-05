export nnUNet_raw=path/to/ImageCAS/nnUNet
export nnUNet_preprocessed=path/to/ImageCAS/nnUNet/trans_data
export nnUNet_results=path/to/result
export nnUNet_compile=False

cd ../../

export CUDA_VISIBLE_DEVICES=0
python train.py 30 3d_fullres 0 --c -num_gpus 1 -tr HiLoTrainer
