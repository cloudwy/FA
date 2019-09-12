import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_noGR = [0.99794,0.95434,0.81535,0.774405,0.615463]
print(mean_noGR)
var_noGR = [1.64632E-07,9.06779E-06,0.00027201,0.000202887,0.000130327]
print(var_noGR)

mean_GR = [0.997715,0.973285,0.917495,0.891235,0.759785]
print(mean_GR)
var_GR = [3.61342E-07,4.42239E-06,8.15816E-05,8.72834E-05,0.000202923]
print(var_GR)

mean_upp = [0.99739,0.971695,0.92973,0.91317,0.80011]
print(mean_upp)
var_upp = [4.17789E-07,7.65839E-06,5.42275E-05,7.18348E-05,0.000147218]
print(var_upp)

# plot
xdata = np.array(['1','2','3','4','5'])
ydata_mean_GR = mean_noGR

plt.errorbar(xdata,mean_noGR,yerr=var_noGR,fmt='x',color='green', ecolor='lightgreen',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_GR,yerr=var_GR,fmt='o',color='red', ecolor='pink',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_upp,yerr=var_upp,fmt='+',color='blue', ecolor='lightblue',elinewidth=3,capsize=0)
#mean_all = 0.7848
#plt.hlines(mean_all,xdata[0],xdata[4],colors='lightgray')

plt.title('AE_MNIST')
plt.xlabel('Task')
plt.ylabel('Acc')
#plt.legend(['without_GR', 'with_GR'])
plt.legend(['lower bound', 'with_GR','upper bound'])
plt.show()



