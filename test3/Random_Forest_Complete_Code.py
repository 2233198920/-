import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # SimHei为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = pd.read_csv("test3-data.csv")

# 数据预处理
TARGET_COLUMN = 'group'  # 目标列
X = data.drop(columns=[TARGET_COLUMN])  # 特征
y = data[TARGET_COLUMN]  # 目标值

# 处理缺失值（用均值填充）
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 初始随机森林模型以评估特征重要性
rf_initial = RandomForestClassifier(random_state=42)
rf_initial.fit(X_scaled, y)

# 获取特征重要性
feature_importances = rf_initial.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]  # 按重要性降序排序

# 可视化所有特征的重要性
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
plt.xticks(range(len(feature_importances)), feature_names[sorted_idx], rotation=90)
plt.xlabel('特征')  # x轴标签
plt.ylabel('重要性')  # y轴标签
plt.title('所有特征的特征重要性')  # 图标题
plt.tight_layout()
plt.show()

# 选择前6个重要特征
top_features = feature_names[sorted_idx][:6]
X_top_features = X_scaled[:, sorted_idx[:6]]  # 仅保留前6个特征

# 超参数训练随机森林模型
rf = RandomForestClassifier(
    n_estimators=210,  # 树的数量
    max_depth=None,  # 树的最大深度
    min_samples_split=2,  # 内部节点拆分所需的最小样本数
    min_samples_leaf=1,  # 叶节点最小样本数
    max_features='sqrt',  # 每棵树分裂时考虑的最大特征数
    bootstrap=True,  # 是否使用自助法
    random_state=42  # 随机种子
)

# 进行5折交叉验证
cv_scores = cross_val_score(rf, X_top_features, y, cv=5)

# 输出5折交叉验证结果
print("5折交叉验证的得分：", cv_scores)
print("5折交叉验证的平均得分：", cv_scores.mean())

