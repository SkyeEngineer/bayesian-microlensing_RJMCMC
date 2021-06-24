import MulensModel as mm
import Functions as mc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform

s_pi = mc.loguni(0.2, 5)
q_pi = mc.loguni(10e-6, 1)
alpha_pi = mc.uni(0, 360)
u0_pi = mc.uni(0, 2)
t0_pi = mc.uni(0, 72)
tE_pi = mc.trunclognorm(1, 100, 10**1.15, 10**0.45)
rho_pi =  mc.loguni(10**-4, 10**-2)

dist = q_pi

y=[]
x=np.linspace(0, 1, 250)
for i in x:
    y.append(np.exp(dist.logPDF(i)))
plt.plot(x, y)
plt.xlabel('Parameter Value')
plt.ylabel('Probability Density')
plt.title(str(dist.__init__())+" probability density function")
plt.savefig('Plots/pdf-test.png')