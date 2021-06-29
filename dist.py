import MulensModel as mm
import Functions as mc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform

s_pi = mc.logUniDist(0.2, 5)
q_pi = mc.logUniDist(10e-6, 1)
alpha_pi = mc.uniDist(0, 360)
u0_pi = mc.uniDist(0, 2)
t0_pi = mc.uniDist(0, 72)
tE_pi = mc.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  mc.logUniDist(10**-4, 10**-2)

distr = q_pi

y=[]
x=np.linspace(10e-6, 1, 500)
for i in x:
    y.append(np.exp(distr.logPDF(i)))
print(distr.dist.cdf(1))
plt.plot(x, y)
plt.xlabel('Parameter Value')
plt.ylabel('Probability Density')
plt.title(" probability density function")
plt.savefig('Plots/pdf-test.png')