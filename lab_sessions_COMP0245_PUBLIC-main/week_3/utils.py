import pickle
import matplotlib.pyplot as plt



savepath = "./245/3"


def save_data(data,name,path =savepath):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(name,path = savepath):
    with open(path+"/"+name+".pkl","rb") as file:
        data = pickle.load(file)
    return data

def get_error(acq_func,length_scale):
    [best_kd,best_kp,best_result,tracking_errors]=load_data(acq_func+f"_{length_scale}")
    return best_result,tracking_errors

def get_PD(acq_func,length_scale):
    [best_kd,best_kp,best_result,tracking_errors]=load_data(acq_func+f"_{length_scale}")
    return best_kp,best_kd


# 误差可视化函数
def plot_error_vs_acq_func(acq_funcs, length_scale):
    plt.figure(figsize=(10, 6))

    for acq_func in acq_funcs:
        best_result, tracking_errors = get_error(acq_func, length_scale)
        plt.plot(tracking_errors, label=f'{acq_func} (best={best_result:.4f})')

    plt.title(f'Tracking Errors for Different Acquisition Functions (Length Scale={length_scale})')
    plt.xlabel('Iterations')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath+f'/Acquisition Functions on Error with Length Scale {length_scale}.png')

def plot_error_vs_length_scale(acq_func, length_scales):
    plt.figure(figsize=(10, 6))

    for length_scale in length_scales:
        best_result, tracking_errors = get_error(acq_func, length_scale)
        plt.plot(tracking_errors, label=f'Length Scale={length_scale} (best error={best_result:.4f})')

    plt.title(f'Tracking Errors for Different Length Scales (Acquisition Function={acq_func})')
    plt.xlabel('Iterations')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath+f'/Length Scales on Error with Acquisition Function {acq_func}.png')



def compare(method = "func"):
    if method == "func":
        for length in [0.1,1,10]:
            plot_error_vs_acq_func(["LCB","EI","PI"],length)
    elif method == "length":
        for func in ["LCB","EI","PI"]:
            plot_error_vs_length_scale(func,[0.1,1,10])
    ...

# compare("func")
# compare("length")