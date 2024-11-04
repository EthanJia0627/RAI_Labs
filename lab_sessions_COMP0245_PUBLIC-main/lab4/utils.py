import pickle
import matplotlib.pyplot as plt



savepath = "./245/4"

structures = [256,128,64,'2layer']
loss_funcs = ['NLLLoss','MSELoss','CrossEntropyLoss']
activations = ['Relu','Sigmoid']
optimizers = ['SGD','Adam','RMSprop']
learning_rates = [0.0001,0.01,1]
batch_sizes = [64,128,256]

def save_data(data,name,path =savepath):
    with open(path+"/"+name+".pkl","wb") as file:
        pickle.dump(data,file)

def load_data(name,path = savepath):

    try:
        with open(path+"/"+name+".pkl","rb") as file:
            data = pickle.load(file)
    except FileNotFoundError:
        return None
    return data

