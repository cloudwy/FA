import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_noGR = [0.970405,0.85067,0.73883,0.6441985,0.620735]
print(mean_noGR)
var_noGR = [0.00047148, 0.000118408, 0.000111942,0.000189812,0.000121793]
print(var_noGR)

mean_GR = [0.977855,0.8972025,0.785475,0.6768825,0.68452]
print(mean_GR)
var_GR = [7.19192E-06,1.65662E-05,0.000142592,5.49242E-05,0.0003304]
print(var_GR)

mean_upp = [0.978255,0.88583,0.78728,0.6817475,0.67697]
print(mean_upp)
var_upp = [7.38414E-06,4.55097E-05,3.68884E-05,0.000214301,0.000319543]
print(var_upp)

# plot
xdata = np.array(['1','2','3','4','5'])
ydata_mean_GR = mean_noGR

plt.errorbar(xdata,mean_noGR,yerr=var_noGR,fmt='x',color='black', ecolor='lightgray',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_GR,yerr=var_GR,fmt='o',color='green', ecolor='lightgreen',elinewidth=3,capsize=0)
plt.errorbar(xdata,mean_upp,yerr=var_upp,fmt='+',color='red', ecolor='mistyrose',elinewidth=3,capsize=0)
#mean_all = 0.7848
#plt.hlines(mean_all,xdata[0],xdata[4],colors='lightgray')

plt.title('Fashion-MNIST')
plt.xlabel('Task')
plt.ylabel('Acc')
#plt.legend(['without_GR', 'with_GR'])
plt.legend(['Lower bound', 'Rehearsal','Upper bound'])
plt.show()



