import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
data = load_diabetes()
x = data.data
y = data.target
print(x.shape,y.shape)
x = np.hstack([np.ones((x.shape[0],1)),x])
plt.scatter(x,y)
plt.show()