conda activate rdd_env
tensorboard --logdir=runs/results

htop

watch -n 1 nvidia-smi

python train.py --config config/rdd.yaml