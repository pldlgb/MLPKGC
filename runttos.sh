# CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18RR --num_iterations 500 --batch_size 128 --lr 0.003 --dr 1.0 --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 --hidden_dropout2 0.3 --label_smoothing 0.1 --reverse True

export CUDA_VISIBLE_DEVICES=5

nohup python -u main_ttos.py \
    --dataset FB15k-237 \
    --num_iterations 700 \
    --batch_size 128 \
    --lr 0.001 \
    --dr 1.0 \
    --edim 200 \
    --rdim 100 \
    --input_dropout 0.3 \
    --hidden_dropout1 0.4 \
    --hidden_dropout2 0.5 \
    --label_smoothing 0.1 \
    --reverse True > log/RunTTOSTucker4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset PharmKG --num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1