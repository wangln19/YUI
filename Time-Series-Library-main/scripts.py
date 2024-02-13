import os
import pandas as pd

# for _ in pd.date_range('2023-04-01', '2023-10-31', freq='D'):
#     print(_.strftime('%Y-%m-%d'))
#     os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model iTransformer --data shanxi --features MS --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))
# os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model TimesNet --data shanxi_elec --features MS --seq_len 96*7 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 16 --d_ff 32 --des 'Exp' --itr 1000 --top_k 5')

for _ in pd.date_range('2023-01-01', '2023-9-30', freq='D'):
    print(_.strftime('%Y-%m-%d'))
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path iso_new_england.csv --model_id iso_{} --model VMD_LSTM --data iso_new_england_per_point --features S --target Price --seq_len 168 --label_len 1 --pred_len 1 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))# --d_model 128 --d_ff 128
    os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path iso_new_england.csv --model_id iso_{} --model FEDformer --data iso_new_england --features MS --seq_len 168 --label_len 24 --pred_len 24 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 9 --dec_in 9 --c_out 1 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d'))) #  --d_model 128 --d_ff 128


# for _ in pd.date_range('2023-04-01', '2023-12-31', freq='D'):
#     print(_.strftime('%Y-%m-%d'))
#     os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_capacity_only.csv  --seasonal_patterns Yearly --model_id shanxi_capacity_{} --model TimesNet --data shanxi_capacity --features S --seq_len 7 --label_len 5 --pred_len 5 --target day_ahead_thermal_power_capacity --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --d_model 16 --d_ff 32 --des \'Exp\' --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))

