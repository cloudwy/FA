import numpy as np
import matplotlib.pyplot as plt

# calculate mean and variance
mean_noGR = [0.968435,0.67632,0.495095,0.36792,0.34703]
print(mean_noGR)
var_noGR = [0.000273652,0.002686306,0.006395452,0.01837042,0.014657438]
print(var_noGR)

mean_GR = [0.97329,0.81128,0.609115,0.48695,0.34217]
print(mean_GR)
var_GR = [0.00018161,0.004664653,0.024786899,0.016273707,0.040615151]
print(var_GR)

mean_upp = [0.969685,0.841515,0.7415,0.59938,0.573135]
print(mean_upp)
var_upp = [0.000118798,0.001985345,0.002485191,0.001351144,0.001158503]
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
plt.legend(['Lower bound', 'DGR','Upper bound'])
plt.show()



