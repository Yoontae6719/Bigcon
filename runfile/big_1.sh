for cat_dim in 64
do
for cont_emb in 32
do
for cont_hidden in 16
do
for cat_depth in 4
do
for cat_heads in 8
do

    python -u run.py \
      --is_training 1 \
      --root_path ./DL_dataset/ \
      --data Bigcon \
      --train_csv train.csv \
      --test_csv test.csv \
      --fold 0 \
      --model_id bigcon \
      --model my_model \
      --num_cont 8 \
      --num_cat 23 \
      --cat_unique_list 63 179 52 15 4 25 20 7 10 5 5 20 9 11 20 3 3 3 3 3 4 25 8 \
      --cat_dim $cat_dim \
      --cont_emb $cont_emb \
      --cont_hidden $cont_hidden \
      --cat_depth $cat_depth \
      --cat_heads $cat_heads \
      --batch_size 5120 &
      
      
    python -u run.py \
      --is_training 1 \
      --root_path ./DL_dataset/ \
      --data Bigcon \
      --train_csv train.csv \
      --test_csv test.csv \
      --fold 1 \
      --model_id bigcon \
      --model my_model \
      --num_cont 8 \
      --num_cat 23 \
      --cat_unique_list 63 179 52 15 4 25 20 7 10 5 5 20 9 11 20 3 3 3 3 3 4 25 8 \
      --cat_dim $cat_dim \
      --cont_emb $cont_emb \
      --cont_hidden $cont_hidden \
      --cat_depth $cat_depth \
      --cat_heads $cat_heads \
      --batch_size 5120 &
      
    python -u run.py \
      --is_training 1 \
      --root_path ./DL_dataset/ \
      --data Bigcon \
      --train_csv train.csv \
      --test_csv test.csv \
      --fold 2 \
      --model_id bigcon \
      --model my_model \
      --num_cont 8 \
      --num_cat 23 \
      --cat_unique_list 63 179 52 15 4 25 20 7 10 5 5 20 9 11 20 3 3 3 3 3 4 25 8 \
      --cat_dim $cat_dim \
      --cont_emb $cont_emb \
      --cont_hidden $cont_hidden \
      --cat_depth $cat_depth \
      --cat_heads $cat_heads \
      --batch_size 5120 &
      
    python -u run.py \
      --is_training 1 \
      --root_path ./DL_dataset/ \
      --data Bigcon \
      --train_csv train.csv \
      --test_csv test.csv \
      --fold 3 \
      --model_id bigcon \
      --model my_model \
      --num_cont 8 \
      --num_cat 23 \
      --cat_unique_list 63 179 52 15 4 25 20 7 10 5 5 20 9 11 20 3 3 3 3 3 4 25 8 \
      --cat_dim $cat_dim \
      --cont_emb $cont_emb \
      --cont_hidden $cont_hidden \
      --cat_depth $cat_depth \
      --cat_heads $cat_heads \
      --batch_size 5120 &
      
    python -u run.py \
      --is_training 1 \
      --root_path ./DL_dataset/ \
      --data Bigcon \
      --train_csv train.csv \
      --test_csv test.csv \
      --fold 4 \
      --model_id bigcon \
      --model my_model \
      --num_cont 8 \
      --num_cat 23 \
      --cat_unique_list 63 179 52 15 4 25 20 7 10 5 5 20 9 11 20 3 3 3 3 3 4 25 8 \
      --cat_dim $cat_dim \
      --cont_emb $cont_emb \
      --cont_hidden $cont_hidden \
      --cat_depth $cat_depth \
      --cat_heads $cat_heads \
      --batch_size 5120
done
done
done
done
done