CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type age --data-path /data/sjx/SentEmbData/age2.pickle --model-type STG-SA --hidden-dim 600 --clf-hidden-dim 2000 --clf-num-layers 2 --dropout 0.3 --batch-size 50 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --optimizer adam --save-dir /data/sjx/SA-Tree-Exp/age2/stgsa_default

