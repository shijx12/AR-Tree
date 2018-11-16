
# 300d
CUDA_VISIBLE_DEVICES=1 python3.6 train.py --data-type snli --glove-path /data/sjx/glove.840B.300d.py36.pt --data-path /data/sjx/AR-Tree-Exp/snli.pt --model-type STG --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --dropout 0.1 --batch-size 128 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --optimizer adam --patience 10 --clip 5 --sample-num 2 --use-batchnorm --rank-input w --fix-word-embedding --leaf-rnn-type lstm --cuda --save-dir /data/sjx/AR-Tree-Exp/debug2

# 100d
# CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type snli --glove-path /data/sjx/glove.6B.100d.py36.pkl --data-path /data/sjx/SA-Tree-Exp/snli.pt --model-type STG --word-dim 100 --hidden-dim 100 --clf-hidden-dim 200 --clf-num-layers 1 --dropout 0.1 --batch-size 128 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --optimizer adam --patience 10 --sample-num 2 --use-batchnorm --cuda --save-dir /data/sjx/SA-Tree-Exp/snli/100d_stgsa_default
