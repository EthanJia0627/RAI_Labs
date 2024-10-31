import pickle
savepath = "./242/final"
def save_data(data,path,name):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(path,name):
    with open(path+"/"+name+".pkl","rb") as file:
        data = pickle.load(file)
    return data

