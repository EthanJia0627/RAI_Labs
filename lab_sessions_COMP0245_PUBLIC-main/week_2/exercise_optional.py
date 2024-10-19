from utils import *
path = "ex5"
task = "california_housing"
method = "AdaBoost"
depth = 2
n_estimators = 50
random_state = 42
loss = ['linear', 'square', 'exponential']
MSE_depth = []
for depth in range(1,21):
    hyperparams = [depth,n_estimators,random_state,loss[0]]
    X,feature_names,y,y_pred_all,X_test,y_test,y_pred,MSE_test = fit_and_pred(task,method,hyperparams)
    visualize(X,...,y,y_pred_all,task,method,hyperparams,save=True,show=False,path=path)
    MSE_depth.append(MSE_test)
depths = range(1,21)
plt.plot(depths, MSE_depth, marker='o')
plt.title(f'Depth on fitting {task} with {method}')
plt.xlabel('Tree Depth')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(f"./245/2/"+path+f"/{task}_{method}_Depth_MSE.png")