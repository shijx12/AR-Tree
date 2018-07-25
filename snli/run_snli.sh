
# 300d
CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type snli --glove-path /data/sjx/glove.840B.300d.py36.pkl --data-path /data/sjx/SA-Tree-Exp/snli.pt --model-type STG-SA --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --dropout 0.1 --batch-size 128 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --optimizer adam --sample-num 2 --use-batchnorm --fix-word-embedding --save-dir /data/sjx/SA-Tree-Exp/snli/300d_stgsa_default

# 100d
# CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type snli --glove-path /data/sjx/glove.6B.100d.py36.pkl --data-path /data/sjx/SA-Tree-Exp/snli.pt --model-type STG-SA --word-dim 100 --hidden-dim 100 --clf-hidden-dim 200 --clf-num-layers 1 --dropout 0.1 --batch-size 128 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --optimizer adam --sample-num 2 --use-batchnorm --save-dir /data/sjx/SA-Tree-Exp/snli/100d_stgsa_default
