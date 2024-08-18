import os
import pandas as pd


for _ in pd.date_range('2023-09-30', '2024-03-31', freq='D'):
    print(_.strftime('%Y-%m-%d'))
    os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model TSMixer --data shanxi --features MS --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))

    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model VMD_LSTM --data shanxi_per_point --features S --target dayahead_price1 --seq_len 672 --label_len 1 --pred_len 1 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))# --d_model 128 --d_ff 128
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model PatchTST --data shanxi --features MS --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_power.csv --model_id shanxi_{} --model DLinear --data shanxi_power --features MS --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model TimesNet --data shanxi --features S --target dayahead_price1 --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))

# # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model TimesNet --data shanxi --features MS --seq_len 96*7 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 16 --d_ff 32 --des 'Exp' --itr 1000 --top_k 5')

# for _ in pd.date_range('2023-01-01', '2023-9-30', freq='D'): #2023-01-01
#     print(_.strftime('%Y-%m-%d'))
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path iso_new_england.csv --model_id iso_{} --model VMD_LSTM --data iso_new_england_per_point --features S --target Price --seq_len 168 --label_len 1 --pred_len 1 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))# --d_model 128 --d_ff 128
    # os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path iso_new_england.csv --model_id iso_{} --model Crossformer --data iso_new_england --features MS --seq_len 168 --label_len 24 --pred_len 24 --target Price --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 9 --dec_in 9 --c_out 1 --des Exp --itr 1 --top_k 5 --learning_rate 0.00001 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d'))) #  --d_model 128 --d_ff 128
    

# for _ in pd.date_range('2023-12-29', '2024-02-21', freq='D'):
#     print(_.strftime('%Y-%m-%d'))
#     os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_capacity_only.csv  --seasonal_patterns Yearly --model_id shanxi_capacity_{} --model TimesNet --data shanxi_capacity --features S --seq_len 7 --label_len 5 --pred_len 5 --target day_ahead_thermal_power_capacity --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --d_model 16 --d_ff 32 --des \'Exp\' --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))

# for _ in pd.date_range('2022-10-01', '2022-12-31', freq='D'):
#     print(_.strftime('%Y-%m-%d'))
#     os.system('python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path iso_capacity_only.csv  --seasonal_patterns Yearly --model_id iso_capacity_{} --model TimesNet --data iso_capacity --features S --seq_len 14 --label_len 2 --pred_len 2 --target Total_Available_Capacity --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --d_model 16 --d_ff 32 --des \'Exp\' --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))


# for _ in pd.date_range('2024-01-01', '2024-01-31', freq='D'):
#     os.system('python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model Crossformer --data shanxi --features MS --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d')))



# import threading
# import subprocess

# class TimeoutException(Exception):
#     pass

# def timeout(seconds):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             result = [TimeoutException("运行时间超过 {} 秒，自动终止".format(seconds))]
#             def new_func():
#                 try:
#                     result[0] = func(*args, **kwargs)
#                 except Exception as e:
#                     result[0] = e
#             thread = threading.Thread(target=new_func)
#             thread.daemon = True
#             thread.start()
#             thread.join(seconds)
#             if isinstance(result[0], BaseException):
#                 raise result[0]
#             return result[0]
#         return wrapper
#     return decorator

# @timeout(4000)  # 设置超时时间为400秒
# def run_command(command):
#     subprocess.run(command, shell=True)

# # 主程序
# for _ in pd.date_range('2023-12-08', '2024-03-31', freq='D'):
#     print(_.strftime('%Y-%m-%d'))
#     command = 'python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path shanxi_elec.csv --model_id shanxi_{} --model TimesNet --data shanxi --features S --target dayahead_price1 --seq_len 672 --label_len 96 --pred_len 96 --inverse  --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --d_ff 128 --des Exp --itr 1 --top_k 5 --tgt_dt {}'.format(_.strftime('%Y-%m-%d'), _.strftime('%Y-%m-%d'))
#     try:
#         run_command(command)
#     except TimeoutException as e:
#         print(e)
