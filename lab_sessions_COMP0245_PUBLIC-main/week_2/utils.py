import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from typing import Literal
import json 

def fit_and_pred(
        task:Literal["sin*cos","california_housing"]="sin*cos" ,
        method:Literal["Decision_Tree","Polynomial","Bagging","Random_Forest","AdaBoost"] = "Decision_Tree",
        hyperparams:list = ...):
    """An integrated regressor

    This function provides a convinent interface for all the excercise in this LAB session

    Parameters
    ----------
    task : 
    {"sin*cos", "california_housing"}, default="sin*cos"
    method : 
    {"Decision_Tree","Polynomial","Bagging","Random_Forest","AdaBoost"}
    hyperparams :{Decision_Tree [Max_Depth,Splitter], Polynomial[Degree], Bagging[Max_Depth,N_Estimators,Random_State],Random_Forest[N_Estimators,Random_State],AdaBoost[Max_Depth,N_Estimators,Random_State,loss]}
    
    Returns
    ----------
    x1, x2, y, y_pred_all, X_test, y_test, y_pred

    """
    if task == "sin*cos":
        # Generate synthetic data
        x1 = np.arange(0, 10, 0.1)
        x2 = np.arange(0, 10, 0.1)
        x1, x2 = np.meshgrid(x1, x2)
        y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

        # Flatten the arrays
        x1 = x1.flatten()
        x2 = x2.flatten()
        y = y.flatten()
        X = np.vstack((x1, x2)).T

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if method == "Decision_Tree":
 
            # Decision Tree Regression
            rgs = DecisionTreeRegressor(max_depth=hyperparams[0],splitter=hyperparams[1])
            rgs.fit(X_train,y_train)
            y_pred = rgs.predict(X_test)
            y_pred_all = rgs.predict(X)


        elif method == "Polynomial":
            # polynomial regression
            poly = PolynomialFeatures(degree=hyperparams[0])
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            X_all = poly.transform(X)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_all = model.predict(X_all)

        elif method == "Bagging":
            # bagging regression
            rgs = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=hyperparams[0]),n_estimators=hyperparams[1],random_state=hyperparams[2])
            rgs.fit(X_train,y_train)
            y_pred = rgs.predict(X_test)
            y_pred_all = rgs.predict(X)

        elif method == "Random_Forest":
            # randonforest regression
            rgs = RandomForestRegressor(n_estimators=hyperparams[0],random_state=hyperparams[1])
            rgs.fit(X_train,y_train)
            y_pred = rgs.predict(X_test)
            y_pred_all = rgs.predict(X)

        elif method == "AdaBoost":
            # AdaBoost regression
            rgs = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=hyperparams[0]),n_estimators=hyperparams[1],random_state=hyperparams[2],loss = hyperparams[3])
            rgs.fit(X_train,y_train)
            y_pred = rgs.predict(X_test)
            y_pred_all = rgs.predict(X)

        x1 = x1.reshape(100,100)
        x2 = x2.reshape(100,100)
        y = y.reshape(100,100)
        y_pred_all = y_pred_all.reshape(100,100)
        MSE_test = mean_squared_error(y_test,y_pred)
        with open("./245/2/record.pkl","r") as file:
            MSE_Table = json.load(file)
            MSE_Table[f"{task}_{method}_{hyperparams}"]=MSE_test
        with open("./245/2/record.pkl","w") as file:
            json.dump(MSE_Table,file)     
            
        return x1,x2,y,y_pred_all,X_test,y_test,y_pred,MSE_test
    elif task == "california_housing":
        data = fetch_california_housing()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if method == "AdaBoost":
            rgs = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=hyperparams[0]),n_estimators=hyperparams[1],random_state=hyperparams[2],loss = hyperparams[3])
            rgs.fit(X_train,y_train)
            y_pred = rgs.predict(X_test)
            y_pred_all = rgs.predict(X)
            MSE_test = mean_squared_error(y_test,y_pred)
    return X,feature_names,y,y_pred_all,X_test,y_test,y_pred,MSE_test
def visualize(x1,x2,y,y_pred_all,
              task:Literal["sin*cos","california_housing"]="sin*cos" ,
              method:Literal["Decision_Tree","Polynomial","Bagging","Random_Forest","AdaBoost"] = "Decision_Tree",
              hyperparams:list = ...,save = True,show = True,path = ""):
    if task == "sin*cos":
        fig = plt.figure(figsize=(12, 12))

        # 实际值
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(x1, x2, y, cmap='viridis')
        ax.set_title('Actual Data')

        # 预测值
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(x1, x2, y_pred_all, cmap='plasma')
        ax.set_title(f'{method} Predicted Data \nHyperparameters = {hyperparams}')
    elif task == "california_housing":
        X = x1
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # 绘制二维散点图，并使用颜色代表预测值
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(122)
        scatter1 = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred_all, cmap='viridis', s=10)
        fig.colorbar(scatter1, ax=ax, label='Prediction Value') 
        ax.set_title('California Housing Prediction Color Map')
        ax.set_xlabel('inputs')
        ax.set_ylabel('housing')
        ax = fig.add_subplot(121)
        scatter2 = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=10)
        fig.colorbar(scatter2, ax=ax, label='Actual Value')
        ax.set_title('California Housing Actual Value Color Map')
        ax.set_xlabel('inputs')
        ax.set_ylabel('housing')


    if save:
        fig.savefig("./245/2/"+path+"/"+f"{task}_{method}_{hyperparams}.png")
    if show:
        plt.show()
    fig.clf()
    plt.close()

# 示例字典
def get_min_MSE():
    with open("./245/2/record.pkl","r") as file:
        MSE_Table = json.load(file)
    # 找到字典中的最小值
    filtered_dict = {k: v for k, v in MSE_Table.items() if "Polynomial" not in k}
    min_value = min(filtered_dict.values())

    # 找到所有对应最小值的键
    min_keys = [k for k, v in MSE_Table.items() if v == min_value]

    return min_keys,min_value


def init():
    with open("./245/2/record.pkl","w") as file:
        MSE_Table = dict()
        json.dump(MSE_Table,file)

# # 重塑 x1, x2 和 y
# x1 = x1.reshape(100,100)
# x2 = x2.reshape(100,100)
# y = y.reshape(100,100)
# y_pred_poly_all = y_pred_poly_all.reshape(100,100)
# y_pred_tree_all = y_pred_tree_all.reshape(100,100)
# # 绘制3D实际值和预测值



# plt.show()
# print(get_min_MSE())
