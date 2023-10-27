#rename
import pandas as pd

ori_path = '/root/notebooks/dataset/'
new_path = '/home/a3ilab01/Desktop/jason/'

# # 讀取 CSV 檔案
# df = pd.read_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_EXP1.csv')

# # 將 'file_path' 欄位的字串進行替換
# df['file_path'] = df['file_path'].str.replace(ori_path, new_path)

# # 儲存修改後的 CSV 檔案
# df.to_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_EXP1.csv', index=False)

# # 讀取 CSV 檔案
# df = pd.read_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_EXP2.csv')

# # 將 'file_path' 欄位的字串進行替換
# df['file_path'] = df['file_path'].str.replace(ori_path, new_path)

# # 儲存修改後的 CSV 檔案
# df.to_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_EXP2.csv', index=False)

# # 讀取 CSV 檔案
# df = pd.read_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_test.csv')

# # 將 'file_path' 欄位的字串進行替換
# df['file_path'] = df['file_path'].str.replace(ori_path, new_path)

# # 儲存修改後的 CSV 檔案
# df.to_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_test.csv', index=False)

# # 讀取 CSV 檔案
# df = pd.read_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_val_com.csv')

# # 將 'file_path' 欄位的字串進行替換
# df['file_path'] = df['file_path'].str.replace(ori_path, new_path)

# # 儲存修改後的 CSV 檔案
# df.to_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_val_com.csv', index=False)

# 讀取 CSV 檔案
df = pd.read_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_val.csv')

# 將 'file_path' 欄位的字串進行替換
df['file_path'] = df['file_path'].str.replace(ori_path, new_path)

# 儲存修改後的 CSV 檔案
df.to_csv('/home/a3ilab01/Desktop/jason/fruit_8_csv/fruit_dataset_val.csv', index=False)
