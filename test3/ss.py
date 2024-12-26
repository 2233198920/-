import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # 仅保留准确率评估
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 数据加载
file_path = 'test3-data.csv'
data = pd.read_csv(file_path)

# 数据预处理
imputer_knn = KNNImputer(n_neighbors=5)
data['incrassation'] = imputer_knn.fit_transform(data[['incrassation']])

def outlier_treatment_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
    return df

continuous_cols = ['wbc', 'neut', 'ast', 'tbil', 'alp', 'afp', 'cea', 'ca199', 'size', 'age1']
for col in continuous_cols:
    data = outlier_treatment_iqr(data, col)

imputer = SimpleImputer(strategy='mean')
data[continuous_cols] = imputer.fit_transform(data[continuous_cols])

scaler = StandardScaler()
data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

categorical_cols = ['gender', 'ft', 'djs', 'stone', 'incrassation']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical = pd.DataFrame(ohe.fit_transform(data[categorical_cols]))
encoded_categorical.columns = ohe.get_feature_names_out(categorical_cols)
data = data.drop(categorical_cols, axis=1)
data = pd.concat([data, encoded_categorical], axis=1)

# 特征选择：计算皮尔逊相关系数
corr_matrix = data.corr(method='pearson')
corr_with_target = corr_matrix['group'].drop('group')
corr_with_target_abs = corr_with_target.abs()

# 选择相关性最高的特征（例如，选择前10个特征）
top_k = 10
top_features = corr_with_target_abs.sort_values(ascending=False).head(top_k).index.tolist()
print("选择的特征：", top_features)

# 数据分割
X = data[top_features]
y = data['group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 模型训练和调优（随机森林）
param_grid = {
    'n_estimators': [100, 200, 300],     # 森林中树的数量
    'max_depth': [None, 10, 20, 30],     # 树的最大深度
    'min_samples_split': [2, 5, 10],     # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4],       # 叶子节点最少样本数
    'max_features': ['sqrt', 'log2', None]  # 修改此处，去掉 'auto'
}

rf_model = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=50,             # 设定尝试的参数组合数量
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=0              # 输出调优过程的详细信息
)

print("开始进行随机森林的超参数调优...")
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_
print("最佳参数组合：", random_search.best_params_)

# 模型评估
y_pred = best_rf_model.predict(X_test)

# 仅保留准确率的评估
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率：{accuracy:.4f}")

# 交叉验证
cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='accuracy')
print(f"交叉验证得分：{cv_scores}")
print(f"平均交叉验证得分：{cv_scores.mean():.4f}")
print(f"交叉验证得分标准差：{cv_scores.std():.4f}")

# 可视化特征重要性
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('特征重要性')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.show()
