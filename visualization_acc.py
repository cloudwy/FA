import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_GR = [0.994785,0.813635,0.649825,0.507655,0.42498]
print(mean_GR)
var_GR = [5.1595E-05,0.020253227,0.006657944,0.009077331,0.001999509]
print(var_GR)
mean_noGR = [0.994585,0.640325,0.48431,0.387605,0.35388]
print(mean_noGR)
var_noGR = [3.41E-05,0.016615,0.011115,0.007188,0.003537]
print(var_noGR)

# plot
xdata = np.array(['1','2','3','4','5'])
ydata_mean_GR = mean_noGR

plt.errorbar(xdata,mean_noGR,yerr=var_noGR,fmt='x',color='green', ecolor='lightgreen',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_GR,yerr=var_GR,fmt='o',color='red', ecolor='pink',elinewidth=3,capsize=0)
#mean_all = 0.7848
#plt.hlines(mean_all,xdata[0],xdata[4],colors='lightgray')

plt.title('AAE1_MNIST')
plt.xlabel('Task')
plt.ylabel('Acc')
#plt.legend(['all_Data','without_GR', 'with_GR'])
plt.legend(['no_GR', 'with_GR'])
plt.show()



