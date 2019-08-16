import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_GR = [0.9974,0.97006,0.91214,0.8945,0.76127]
print(mean_GR)
var_GR = [2.45E-07,5.19E-06,6.23E-05,9.52E-05,5.93E-05]
print(var_GR)
mean_noGR = [0.99806,0.95358,0.8191,0.76568,0.61898]
print(mean_noGR)
var_noGR = [6.13E-07,6.17E-06,0.000171,0.000145,0.000419]
print(var_noGR)

# plot
xdata = np.array(['1','2','3','4','5'])
ydata_mean_GR = mean_noGR

plt.errorbar(xdata,mean_noGR,yerr=var_noGR,fmt='x',color='green', ecolor='lightgreen',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_GR,yerr=var_GR,fmt='o',color='red', ecolor='pink',elinewidth=3,capsize=0)
#mean_all = 0.7848
#plt.hlines(mean_all,xdata[0],xdata[4],colors='lightgray')

plt.title('AE_MNIST')
plt.xlabel('Task')
plt.ylabel('Acc')
#plt.legend(['all_Data','without_GR', 'with_GR'])
plt.legend(['no_GR', 'with_GR'])
plt.show()



