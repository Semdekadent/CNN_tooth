import joblib
import numpy as np
import matplotlib.pyplot as plt

models = joblib.load('model_1.pkl')

# new_data_point中前三位是抛光度的，100代表120，010代表240，001代表500，最后一位是时间
new_data_point = [[1, 0, 0, 30]]
l_b = models[0].predict(new_data_point)
u_b = models[1].predict(new_data_point)
print(l_b)
print(u_b)
mesh = {'120': [1, 0, 0], '240': [0, 1, 0], '500': [0, 0, 1]}


def draw(mesh_level: str):
    # 预测0到60秒的数据并绘制连续光滑曲线
    X_time_continuous = np.arange(0, 61, 0.5)  # 从0到60秒，以1秒为间隔生成时间点
    predicted_y_lower = []  # 存储预测的Ra值
    predicted_y_upper = []
    predicted_y=[]
    for time_point in X_time_continuous:
        model_input_list = mesh[mesh_level] + [time_point]
        time_point_X = np.array([model_input_list])  # 创建包含当前时间点的输入数据
        lower_bound = models[0].predict(time_point_X)  # 预测下界
        upper_bound = models[1].predict(time_point_X)  # 预测上界
        predicted_y_lower.append(lower_bound)  # 将预测结果添加到列表中
        predicted_y_upper.append(upper_bound)
        predicted_y.append((lower_bound+upper_bound)/2)

    # 多项式拟合
    degree = 6  # 多项式的阶数
    coeff_upper = np.polyfit(X_time_continuous, predicted_y_upper, degree)
    smooth_curve_upper = np.polyval(coeff_upper, X_time_continuous)

    coeff_lower = np.polyfit(X_time_continuous, predicted_y_lower, degree)
    smooth_curve_lower = np.polyval(coeff_lower, X_time_continuous)

    coeff = np.polyfit(X_time_continuous, predicted_y, degree)
    smooth_curve = np.polyval(coeff, X_time_continuous)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    # plt.plot(X_time_continuous, predicted_y_upper, label='upper', marker='o', color='blue')
    plt.plot(X_time_continuous, smooth_curve_upper, label='fitted_upper', linestyle='--', color='blue')
    # plt.plot(X_time_continuous, predicted_y_lower, label='lower', marker='o', color='red')
    plt.plot(X_time_continuous, smooth_curve_lower, label='fitted_lower', linestyle='--', color='red')
    plt.plot(X_time_continuous, smooth_curve, label='fitted_curve', linestyle='--', color='green')
    # 获取y轴的数据范围
    y_min, y_max = plt.ylim()

    # 将y轴的显示范围扩大成两倍
    plt.ylim(0, y_max * 2.0)
    plt.xlabel('s')
    plt.ylabel('Ra')
    plt.title('mesh' + mesh_level)
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__=='__main__':
    draw('120')
    draw('240')
    draw('500')
