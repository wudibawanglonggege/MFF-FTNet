#mul
python -u train.py WTH forecast_multivar --dim 96 --alpha 5e-4 --kernels  1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed 1 --hidden_dim 64 --depth 8 --momentum 0.9 --weight_decay 1e-4 --lr 0.001 --eval
#uni
python -u train.py WTH forecast_univar --dim 96 --alpha 5e-5 --kernels  1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 1 --hidden_dim 64 --depth 8 --momentum 0.9 --weight_decay 1e-4 --lr 0.001 --eval
