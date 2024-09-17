import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone


# 创建一个自定义的MultiOutputRegressor
class CustomMultiOutputRegressor(MultiOutputRegressor):
    def __init__(self, estimators):
        self.estimators = estimators
        super().__init__()

    def fit(self, X, y):
        self.estimators_ = [clone(e) for e in self.estimators]
        for estimator, yi in zip(self.estimators_, y.T):
            estimator.fit(X, yi)
        return self

    def predict(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])


# 读取数据样本
df = pd.read_csv('AFM.csv')  # 根据实际文件名修改

# 假设你的数据存储在pandas DataFrame中，名为df
# 首先，我们需要将数据分割为特征和目标
features = df[['酸蚀时间(s)', '抛光程度(mesh)']]
targets = df[['Ra(nm)', 'length(μm)', 'Angle(°)']]

# 之后，我们将数据分为训练集和测试集
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2,
                                                                              random_state=42)


# 创建一个对特征进行对数变换的函数
def log_transform(x):
    return np.log1p(x)


# 因为“抛光程度(mesh)”是类别变量，我们需要对它进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', FunctionTransformer(log_transform), ['酸蚀时间(s)']),
        ('cat', OneHotEncoder(), ['抛光程度(mesh)'])])

# 为每个目标变量指定一个不同的模型
estimators = [
    Ridge(),  # 用于预测 'Ra(nm)'
    RandomForestRegressor(),  # 用于预测 'length(μm)'
    LinearRegression()  # 用于预测 'Angle(°)'
]


# 创建一个管道来进行预处理和模型拟合
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', CustomMultiOutputRegressor(estimators))])

# 使用训练数据拟合模型
model.fit(features_train, targets_train)

# 使用测试集进行预测
predictions = model.predict(features_test)

# 计算并打印MSE
mse = mean_squared_error(targets_test, predictions, multioutput='raw_values')
print('Mean squared error: ', mse)

# 计算并打印R² score
r2 = r2_score(targets_test, predictions, multioutput='raw_values')
print('R2 score: ', r2)
