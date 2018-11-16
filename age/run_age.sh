CUDA_VISIBLE_DEVICES=0 python3.6 train.py --data-type age --model-type STG --leaf-rnn-type bilstm --hidden-dim 600 --clf-hidden-dim 2000 --clf-num-layers 2 --dropout 0.3 --batch-size 50 --max-epoch 10 --lr 0.001 --l2reg 1e-5 --clip 5 --optimizer adam --patience 20 --cuda --data-path /data/sjx/SentEmbData/age2.pickle --save-dir /data/sjx/AR-Tree-Exp/debug

