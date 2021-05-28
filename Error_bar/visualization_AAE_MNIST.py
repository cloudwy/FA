import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_noGR = [0.99326,0.69641,0.52882,0.40937,0.36982]
print(mean_noGR)
var_noGR = [0.000226038,0.001924352,0.00122144,0.0008961,0.000420835]
print(var_noGR)

mean_GR = [0.99191,0.774,0.56736,0.47994,0.376175]
print(mean_GR)
var_GR = [0.000219443,0.009254498,0.006815134,0.004567369,0.006826251]
print(var_GR)

mean_upp = [0.9971,0.87025,0.77185,0.71199,0.65027]
print(mean_upp)
var_upp = [0.00001316,0.001623665,0.009134969,0.009492092,0.005447565]
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
plt.legend(['Lower bound', 'DGR','Upper bound'])
plt.show()



