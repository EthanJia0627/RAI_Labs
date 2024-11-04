import pickle
import matplotlib.pyplot as plt

savepath = "./245/3"
damping = True

def save_data(data, name, path=savepath):
    with open(path + "/" + name + ".pkl", "wb") as file:
        pickle.dump(data, file)

def load_data(name, path=savepath):
    with open(path + "/" + name + ".pkl", "rb") as file:
        data = pickle.load(file)
    return data

def get_error(acq_func, length_scale, damping=False):
    damping_str = " with Damping" if damping else ""
    [best_kd, best_kp, best_result, tracking_errors] = load_data(acq_func + f"_{length_scale}" + damping_str)
    return best_result, tracking_errors

def get_PD(acq_func, length_scale, damping=False):
    damping_str = " with Damping" if damping else ""
    [best_kd, best_kp, best_result, tracking_errors] = load_data(acq_func + f"_{length_scale}" + damping_str)
    return best_kp, best_kd

# 误差可视化函数
def plot_error_vs_acq_func(acq_funcs, length_scale, damping):
    damping_str = " with Damping" if damping else ""
    plt.figure(figsize=(10, 6))
    for acq_func in acq_funcs:
        best_result, tracking_errors = get_error(acq_func, length_scale, damping)
        plt.plot(tracking_errors, label=f'{acq_func} (best={best_result:.4f})')

    plt.title(f'Tracking Errors for Different Acquisition Functions {damping_str}\n(Length Scale={length_scale})')
    plt.xlabel('Iterations')
    plt.ylabel('Tracking Error ' + damping_str)
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath + f'/Compare/Function/Acquisition Functions on Error with Length Scale {length_scale}' + damping_str + '.png')

def plot_error_vs_length_scale(acq_func, length_scales, damping):
    damping_str = " with Damping" if damping else ""
    plt.figure(figsize=(10, 6))
    for length_scale in length_scales:
        best_result, tracking_errors = get_error(acq_func, length_scale, damping)
        plt.plot(tracking_errors, label=f'Length Scale={length_scale} (best error={best_result:.4f})')

    plt.title(f'Tracking Errors for Different Length Scales {damping_str}\n(Acquisition Function={acq_func})')
    plt.xlabel('Iterations')
    plt.ylabel('Tracking Error ' + damping_str)
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath + f'/Compare/Length/Length Scales on Error with Acquisition Function {acq_func}' + damping_str + '.png')


def plot_error_vs_damping(acq_funcs, length_scales):
    for acq_func in acq_funcs:
        for length_scale in length_scales:
            plt.figure(figsize=(10, 6))
            for damping in [True, False]:
                damping_str = " with Damping" if damping else " without Damping"
                best_result, tracking_errors = get_error(acq_func, length_scale, damping)
                plt.plot(tracking_errors, label=damping_str+f"best={best_result:.4f}")

            plt.title(f'Tracking Error Comparison with and without Damping\n(Acquisition Function={acq_func}, Length Scale={length_scale})')
            plt.xlabel('Iterations')
            plt.ylabel('Tracking Error')
            plt.legend()
            plt.grid(True)
            plt.savefig(savepath + f'/Compare/Damping/Damping Effect on Error for {acq_func} with Length Scale {length_scale}.png')

def compare(method, damping=False):
    if method == "func":
        for length in [0.1, 1, 10]:
            plot_error_vs_acq_func(["LCB", "EI", "PI"], length, damping)
    elif method == "length":
        for func in ["LCB", "EI", "PI"]:
            plot_error_vs_length_scale(func, [0.1, 1, 10], damping)
    elif method == "damping":
        plot_error_vs_damping(["LCB", "EI", "PI"], [0.1, 1, 10])

# 调用比较函数
compare("func", damping)
compare("length", damping)
compare("damping")
