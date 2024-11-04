import pickle
savepath = "./242/2"
def save_data(data,path,name):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(path,name):
    with open(path+"/"+name+".pkl","rb") as file:
        data = pickle.load(file)
    return data

def get_data_with(Q_val,R_val,damping = False):
    damping = "_Damping" if damping else ""
    dataname = f'data with Q:{Q_val},R:{R_val}'+damping
    return load_data(savepath,dataname)