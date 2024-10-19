from utils import *
path = "ex2"
task = "sin*cos"
method = "Bagging"
depth = 16
N_Estimators = 50
Random_State = 42
# MSE_depth = []
# for depth in range(1,21):
#     hyperparams = [depth,N_Estimators,Random_State]
#     x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE=fit_and_pred(task,method,hyperparams)
#     visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path=path)    
#     MSE_depth.append([Cur_MSE])
# depths = range(1, 21)

# plt.plot(depths, MSE_depth, marker='o')
# plt.title(f'Depth on fitting {task} with {method}')
# plt.xlabel('Tree Depth')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.legend()
# plt.savefig(f"./245/2/"+path+f"/{task}_{method}_Depth_MSE.png")
# plt.show()

MSE_N_Estimators = []
N_Estimators_range = []
for N_Estimators in range(1,21):
    N_Estimators_range.append(N_Estimators*5)
    hyperparams = [depth,5*N_Estimators,Random_State]
    x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE=fit_and_pred(task,method,hyperparams)
    visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path=path)    
    MSE_N_Estimators.append([Cur_MSE])

plt.plot(N_Estimators_range, MSE_N_Estimators, marker='o')
plt.title(f'N_Estimators on fitting {task} with {method}')
plt.xlabel('N Estimators')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(f"./245/2/"+path+f"/{task}_{method}_N_Estimators_MSE.png")
plt.show()

MSE_Random_State = []
Random_State_range = []
for Random_State in range(1,21):
    Random_State_range.append(Random_State*5)
    hyperparams = [depth,N_Estimators,5*Random_State]
    x1, x2, y, y_pred_all, X_test, y_test, y_pred,Cur_MSE=fit_and_pred(task,method,hyperparams)
    visualize(x1,x2,y,y_pred_all,task,method,hyperparams,show=False,path=path)    
    MSE_Random_State.append([Cur_MSE])

plt.plot(Random_State_range, MSE_Random_State, marker='o')
plt.title(f'Random_State on fitting {task} with {method}')
plt.xlabel('Random_State')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(f"./245/2/"+path+f"/{task}_{method}_Random_State_MSE.png")
plt.show()