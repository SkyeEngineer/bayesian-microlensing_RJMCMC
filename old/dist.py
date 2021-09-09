import MulensModel as mm
import Functions as mc
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from scipy.stats import truncnorm, loguniform, uniform


#plt.style.use('ggplot')
print(plt.style.available)
#print(plt.rcParams["font.family"].available)


#print(matplotlib.get_cachedir())


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times New Roman']})
#rc('text', usetex=True)


#plt.rcParams["font.family"] = "serif"
#print(plt.rcParams.keys())
#plt.rcParams['font.size'] = 12




s_pi = mc.logUniDist(0.2, 5)
q_pi = mc.logUniDist(10e-6, 1)
alpha_pi = mc.uniDist(0, 360)
u0_pi = mc.uniDist(0, 2)
t0_pi = mc.uniDist(0, 72)
tE_pi = mc.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  mc.logUniDist(10**-4, 10**-2)

distr = tE_pi

y=[]
x=np.linspace(1, 100, 1000)
mu=0
for i in x:
    mu+=np.exp(distr.log_PDF(i))*i
    y.append(np.exp(distr.log_PDF(i)))
print(mu/len(x))
#print(y)

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12

plt.style.use('seaborn-bright')

plt.rcParams["legend.edgecolor"] = '0'
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.title_fontsize"] = 10
plt.rcParams["legend.fontsize"] = 9

plt.rcParams["grid.linestyle"] = 'dashed' 
plt.rcParams["grid.alpha"] = 0.25




plt.plot(x, y, label='Probability\nDensity')
plt.xlabel(r'Parameter [$\chi$]')
plt.ylabel(r'Probability Density [$\rho$]')
plt.title('Probability Density Function')
plt.legend(title='Entries')#, framealpha=1.0, edgecolor='0.0')  #

#plt.axis('scaled')
plt.tight_layout()
plt.grid()
plt.savefig('Plots/pdf-test.png')