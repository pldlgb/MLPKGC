import numpy as np 
import os
from matplotlib import pyplot as plt 
os.system('export DISPLAY=:0.0')

x = np.arange(1,11) 
y =  2 * x +  5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.savefig('./pic/1.png')