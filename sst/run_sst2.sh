CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type sst2 --data-path /data/share/stanfordSentimentTreebank --model-type STG-SA --hidden-dim 300 --clf-hidden-dim 300 --clf-num-layers 1 --dropout 0.5 --batch-size 32 --max-epoch 20 --lr 1 --l2reg 1e-5 --optimizer adadelta --save-dir /data/sjx/SA-Tree-Exp/sst2/stgsa_default

