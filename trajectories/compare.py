from matplotlib import pyplot as plt 
import numpy as np
from operator import itemgetter

data10 = np.loadtxt('results10wo_omega.csv', delimiter=',')
data20 = np.loadtxt('results20wo_omega.csv', delimiter=',')
data50 = np.loadtxt('results50wo_omega.csv', delimiter=',')
m10 = np.mean(data10, axis=0)
m20 = np.mean(data20, axis=0)
m50 = np.mean(data50, axis=0)
wo_omega = np.vstack([m50,np.vstack([m20, m10])])

data10 = np.loadtxt('results10_omega.csv', delimiter=',')
data20 = np.loadtxt('results20_omega.csv', delimiter=',')
data50 = np.loadtxt('results50_omega.csv', delimiter=',')
m10 = np.mean(data10, axis=0)
m20 = np.mean(data20, axis=0)
m50 = np.mean(data50, axis=0)
w_omega = np.vstack([m50,np.vstack([m20, m10])])

data10 = np.loadtxt('GT10.csv', delimiter=',')
data20 = np.loadtxt('GT20.csv', delimiter=',')
data50 = np.loadtxt('GT50.csv', delimiter=',')
m10 = np.mean(data10, axis=0)
m20 = np.mean(data20, axis=0)
m50 = np.mean(data50, axis=0)
GT = np.vstack([m50,np.vstack([m20, m10])])

data10 = np.loadtxt('GT-S10.csv', delimiter=',')
data20 = np.loadtxt('GT-S20.csv', delimiter=',')
data50 = np.loadtxt('GT-S50.csv', delimiter=',')
m10 = np.mean(data10, axis=0)
m20 = np.mean(data20, axis=0)
m50 = np.mean(data50, axis=0)
GT_S = np.vstack([m50,np.vstack([m20, m10])])

data10 = np.loadtxt('NH10.csv', delimiter=',')
data20 = np.loadtxt('NH20.csv', delimiter=',')
data50 = np.loadtxt('NH50.csv', delimiter=',')
m10 = np.mean(data10, axis=0)
m20 = np.mean(data20, axis=0)
m50 = np.mean(data50, axis=0)
NH = np.vstack([m50,np.vstack([m20, m10])])

plt.plot(wo_omega[:,1],wo_omega[:,0], 'bo')
plt.plot(wo_omega[:,1],wo_omega[:,0], 'b', label='Siplified GT-CTRL (Python)')
plt.plot(w_omega[:,1],w_omega[:,0], 'go')
plt.plot(w_omega[:,1],w_omega[:,0], 'g', label='Full GT-CTRL (Python)')
plt.plot(GT_S[:,1],GT_S[:,0], 'ro')
plt.plot(GT_S[:,1],GT_S[:,0], 'r', label='Siplified GT-CTRL (Simulink)')
plt.plot(GT[:,1],GT[:,0], 'mo')
plt.plot(GT[:,1],GT[:,0], 'm', label='Full GT-CTRL (Simulink)')
plt.plot(NH[:,1],NH[:,0], 'ko')
plt.plot(NH[:,1],NH[:,0], 'k', label='NH-CTRL (Simulink)')
plt.title('Dependency of mean error in position from median velocity')
plt.xlabel('Median velocity [m/s]')
plt.ylabel('Mean std_deviation [m]')
plt.legend(loc='upper left')
plt.show()