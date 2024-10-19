from utils import *
path = "ex1"
MSE_depth = []
task = "sin*cos"
method = "Decision_Tree"
for depth in range(1,21):
    splitter = "best" 
    hyperparams = [depth,splitter]
    x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE_best=fit_and_pred(task,method,hyperparams)
    visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path = path)    
    splitter = "random" 
    hyperparams = [depth,splitter]
    x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE_random=fit_and_pred(task,method,hyperparams)
    visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path = path)
    MSE_depth.append([Cur_MSE_best,Cur_MSE_random])
MSE_splitter = list(zip(*MSE_depth))


depths = range(1, 21)
plt.plot(depths, MSE_splitter[0], label='Best Splitter', marker='o')

# 绘制 random splitter 的 MSE 变化
plt.plot(depths, MSE_splitter[1], label='Random Splitter', marker='s')

# 设置图例、标题和标签
plt.title(f'Hyperparameters on fitting {task} with {method}')
plt.xlabel('Tree Depth')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(f"./245/2/"+path+f"/{task}_{method}_MSE.png")
plt.show()

MSE_degree = []
method = "Polynomial"
for degree in range(1,21):
    hyperparams = [degree]
    x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE=fit_and_pred(task,method,hyperparams)
    visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path = path)
    MSE_degree.append(Cur_MSE)    
depths = range(1, 21)
plt.plot(depths, MSE_splitter[0], label='Decision Tree Best Splitter', marker='o')

# 绘制 random splitter 的 MSE 变化
plt.plot(depths, MSE_degree, label='Polynomial', marker='s')

# 设置图例、标题和标签
plt.title(f'Decision Tree and Polynomial on fitting {task}')
plt.xlabel('Tree Depth/Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(f"./245/2/"+path+"/Decision_Tree_Polynomial_MSE.png")
plt.show()

