CUDA_VISIBLE_DEVICES=0 python main.py --dataset PharmKG --num_iterations 500 --batch_size 128 --lr 0.003 --dr 1.0 --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 --hidden_dropout2 0.3 --label_smoothing 0.1