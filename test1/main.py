import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import Levenshtein  # 引入编辑距离库

# 读取sheet1，只读取前三列
df1 = pd.read_excel('test1-data.xlsx', sheet_name='sheet1', header=None, usecols=[0, 1, 2])
df1.columns = ['编号', '期刊全名', '影响因子']

# 读取sheet2，读取所有列，假设简称可能在后续所有列中
df2 = pd.read_excel('test1-data.xlsx', sheet_name='sheet2', header=None)

# 将第一列设为期刊名，其余列为简称
df2.columns = ['期刊名'] + [f'简称{i}' for i in range(1, df2.shape[1])]


# 定义期刊名规范化函数
def normalize_name(name):
    name = str(name).lower()
    name = name.replace('&', 'and')
    name = ''.join(char for char in name if char.isalnum() or char.isspace())
    name = ' '.join(name.split())
    return name


# 规范化期刊名
df1['规范化期刊名'] = df1['期刊全名'].apply(normalize_name)
df2['规范化期刊名'] = df2['期刊名'].apply(normalize_name)

# 创建期刊名到简称的映射，允许多个简称
name_to_abbrs = defaultdict(set)
for idx, row in df2.iterrows():
    norm_name = row['规范化期刊名']
    # 获取该行所有简称，去除空值
    abbrs = [abbr for abbr in row[1:-1] if pd.notnull(abbr)]
    for abbr in abbrs:
        name_to_abbrs[norm_name].add(abbr)

# 收集结果
results = []
for idx, row in tqdm(df1.iterrows(), total=len(df1)):
    full_name = row['期刊全名']
    impact_factor = row['影响因子']
    norm_name = row['规范化期刊名']
    abbreviations = list(name_to_abbrs.get(norm_name, set()))

    # 如果没有直接匹配，尝试使用编辑距离匹配
    if not abbreviations:
        min_distance = float('inf')
        closest_match = None
        for name in name_to_abbrs.keys():
            # 计算编辑距离
            distance = Levenshtein.distance(norm_name, name)
            # 计算相对距离（归一化距离）
            relative_distance = distance / max(len(norm_name), len(name))
            # 设置阈值，例如0.2（可根据需求调整）
            if relative_distance < 0.2 and distance < min_distance:
                min_distance = distance
                closest_match = name
        if closest_match:
            abbreviations = list(name_to_abbrs.get(closest_match, set()))

    result_row = [full_name, impact_factor] + abbreviations
    results.append(result_row)

# 确定最大简称数量
max_abbrs = max(len(r) - 2 for r in results)
columns = ['期刊全名', '影响因子'] + [f'简称{i + 1}' for i in range(max_abbrs)]

# 创建结果DataFrame
df_results = pd.DataFrame(results, columns=columns)

# 按期刊全名字母序排列
df_results.sort_values('期刊全名', inplace=True)

# 将结果写入sheet3
with pd.ExcelWriter('test1-data.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_results.to_excel(writer, sheet_name='sheet3', index=False, header=False)
