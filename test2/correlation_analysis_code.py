import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 数据
data = {
    "年份": [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014],
    "平均房价": [10437.00, 9991.00, 10322.67, 10030.17, 9469.42, 8859.18, 8008.11, 7565.00, 6854.72, 6369.76],
    "GDP": [1260582.1, 1204724.0, 1149237.0, 1013567.0, 986515.2, 919281.1, 832035.9, 746395.1, 688858.2, 643563.1],
    "人均GDP": [89358, 85310, 81370, 71828, 70078, 65534, 59592, 53783, 49922, 46912],
    "失业率": [0.0510, 0.0560, 0.0510, 0.0424, 0.0360, 0.0380, 0.0390, 0.0400, 0.0410, 0.0410],
    "人均可支配收入": [39218, 36883, 35128, 32189, 30733, 28228, 25974, 23821, 21966, 20167],
    "M2货币供应量": [2922713.33, 2664320.84, 2382899.56, 2186795.89, 1986488.82, 1826744.20, 1690235.31, 1550066.67, 1392278.11, 1228374.81]
}

# 加载数据
df = pd.DataFrame(data)

# 计算相关性矩阵
correlation_matrix = df[["平均房价", "GDP", "人均GDP", "失业率", "人均可支配收入", "M2货币供应量"]].corr()

# 设置支持中文显示的字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei 黑体
rcParams['axes.unicode_minus'] = False   # 解决坐标轴负号显示问题

# 可视化相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("经济指标与平均房价的相关性矩阵")
plt.show()

# 滞后分析：将房价列向后移动一行以进行滞后相关性计算
df["滞后房价"] = df["平均房价"].shift(1)

# 计算滞后相关性
lagged_corr = df.corr().loc["滞后房价", ["平均房价", "GDP", "人均GDP", "失业率", "人均可支配收入", "M2货币供应量"]].dropna()

# 可视化滞后相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(lagged_corr.to_frame(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("滞后房价与经济指标的相关性矩阵")
plt.show()

# 预测分析：绘制 GDP、人均可支配收入与房价的散点图
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
axes[0].scatter(df["GDP"], df["平均房价"], alpha=0.7)
axes[0].set_title("GDP 与平均房价的关系")
axes[0].set_xlabel("GDP（亿元）")
axes[0].set_ylabel("平均房价（元）")

axes[1].scatter(df["人均可支配收入"], df["平均房价"], alpha=0.7, color="green")
axes[1].set_title("人均可支配收入与平均房价的关系")
axes[1].set_xlabel("人均可支配收入（元）")
axes[1].set_ylabel("平均房价（元）")

plt.tight_layout()
plt.show()