import pickle

from matplotlib import pyplot as plt
import numpy as np
savepath = "./242/final"
def save_data(data,path,name):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(path,name):
    with open(path+"/"+name+".pkl","rb") as file:
        data = pickle.load(file)
    return data
def plot_error(save_path,Qcoeff_init,Rcoeff_init,N_mpc_init):
    ## delete space in the path
    path = save_path + f"/error_{Qcoeff_init}_{Rcoeff_init}_{N_mpc_init}.pkl"
    path = path.replace(" ","")
    with open(path, 'rb') as f:
        error = pickle.load(f)
    path = save_path + f"/error_{Qcoeff_init}_{Rcoeff_init}_{N_mpc_init}_noEKF.pkl"
    path = path.replace(" ","")
    with open(path, 'rb') as f:
        EKF_error = pickle.load(f)
    error = np.array(error)
    EKF_error = np.array(EKF_error)
    plt.figure()
    plt.plot(error[:,0], label='x error',color='red')
    plt.plot(error[:,1], label='y error',color='blue')
    plt.plot(error[:,2], label='theta error',color='green')
    plt.plot(EKF_error[:,0], label='x error no EKF',color='red',linestyle='dashed')
    plt.plot(EKF_error[:,1], label='y error no EKF',color='blue',linestyle='dashed')
    plt.plot(EKF_error[:,2], label='theta error no EKF',color='green',linestyle='dashed')
    plt.legend()
    plt.grid()
    plt.show()

Qcoeff_init = np.array([380,380,2500])
Rcoeff_init = 3.5
N_mpc_init = 20
plot_error(savepath,Qcoeff_init,Rcoeff_init,N_mpc_init)