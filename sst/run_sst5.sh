CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type sst5 --model-type STG --leaf-rnn-type lstm --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --dropout 0.5 --batch-size 64 --max-epoch 20 --lr 1 --l2reg 1e-5 --clip 5 --optimizer adadelta --patience 20 --cuda --data-path /data/share/stanfordSentimentTreebank --save-dir /data/sjx/AR-Tree-Exp/debug

