import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_noGR = [0.99632, 0.94954, 0.82297, 0.76231, 0.61284]
print(mean_noGR)
var_noGR = [9.77333E-07, 1.88382E-05, 0.000205285, 0.00025789, 0.000160592]
print(var_noGR)

mean_GR = [0.99678, 0.96591, 0.91191, 0.87322, 0.76602]
print(mean_GR)
var_GR = [1.20178E-06, 9.23878E-06, 0.000299454, 0.000118022, 0.000275128]
print(var_GR)

mean_upp = [0.99687, 0.96233, 0.91422, 0.87402, 0.761602]
print(mean_upp)
var_upp = [4.40111E-07,1.20757E-05,7.9464E-05,3.94462E-05,7.55023E-05]
print(var_upp)

# plot
xdata = np.array(['1','2','3','4','5'])
ydata_mean_GR = mean_noGR

plt.errorbar(xdata,mean_noGR,yerr=var_noGR,fmt='x',color='black', ecolor='lightgray',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_GR,yerr=var_GR,fmt='o',color='green', ecolor='lightgreen',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_upp,yerr=var_upp,fmt='+',color='red', ecolor='mistyrose',elinewidth=3,capsize=0)
#mean_all = 0.7848
#plt.hlines(mean_all,xdata[0],xdata[4],colors='lightgray')

plt.title('MNIST')
plt.xlabel('Task')
plt.ylabel('Acc')
#plt.legend(['without_GR', 'with_GR'])
plt.legend(['Lower bound', 'Rehearsal','Upper bound'])
plt.show()



