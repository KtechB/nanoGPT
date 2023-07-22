import numpy as np
import os

# 分割されたファイルのリスト
input_files = [f'train{i}.bin' for i in range(5)  ]
# 各ファイルの形状とデータ型を保存するリスト
shapes = []
dtypes = []

# 各ファイルから形状とデータ型を取得
for file in input_files:
    mmapped_data = np.memmap(file, mode='r')
    shapes.append(mmapped_data.shape)
    dtypes.append(mmapped_data.dtype)

# すべてのファイルが同じデータ型を持っていることを確認
assert len(set(dtypes)) == 1, "All files must have the same dtype"
final_dtype = dtypes[0]

# すべての形状が同じ次元数を持っていることを確認
assert len(set([len(shape) for shape in shapes])) == 1, "All files must have the same number of dimensions"
dims = len(shapes[0])

# 新しい形状を計算
new_shape = [sum([shape[i] for shape in shapes]) if i == 0 else shapes[0][i] for i in range(dims)]

# 集約されたファイルを作成
output_file = 'train.bin'
aggregated_mmapped_data = np.memmap(output_file, dtype=final_dtype, mode='w+', shape=tuple(new_shape))

# 入力ファイルのデータを集約ファイルにコピー
start_index = 0
for file, shape in zip(input_files, shapes):
    mmapped_data = np.memmap(file, dtype=final_dtype, mode='r', shape=shape)
    end_index = start_index + shape[0]
    aggregated_mmapped_data[start_index:end_index] = mmapped_data
    start_index = end_index

# ファイルをディスクにフラッシュ
aggregated_mmapped_data.flush()

print(f"Aggregated {len(input_files)} files into {output_file}")
