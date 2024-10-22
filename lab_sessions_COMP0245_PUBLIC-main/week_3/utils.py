import pickle
import matplotlib.pyplot as plt



savepath = "./245/4"


def save_data(data,name,path =savepath):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(name,path = savepath):
    with open(path+"/"+name+".pkl","rb") as file:
        data = pickle.load(file)
    return data

