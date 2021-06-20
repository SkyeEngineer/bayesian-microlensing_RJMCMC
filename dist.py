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

y=[]
x=np.linspace(0.5, 150, 200)
for i in x:
    y.append(np.exp(tE_pi.pdf(i)))
plt.plot(x, y)
plt.xlabel('value') # Set the y axis label of the current axis.
plt.ylabel('density') # Set a title.
plt.title('pdfs')
plt.savefig('pdf test.png')
#print(tE_pi.a)
#print(tE_pi.b)