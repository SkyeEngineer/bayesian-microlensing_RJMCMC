# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Lightcurve plotting]

import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import Functions as f
import matplotlib.patches as mpatches

# Get chi2 by creating a new model and fitting to previously generated data
def chi2(t_0, u_0, t_E, rho, q, s, alpha, Data):
    Model = mm.Model({'t_0': t_0, 'u_0': u_0,'t_E': t_E, 'rho': rho, 'q': q, 's': s,'alpha': alpha})
    Model.set_magnification_methods([0., 'VBBL', 72.])
    Event = mm.Event(datasets=Data, model=Model)
    return Event.get_chi2()



# Synthetic Events
#BinaryModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
BinaryModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0056, 'q': 0.0009, 's': 1.3, 'alpha': 210.8})
BinaryModel.set_magnification_methods([0., 'VBBL', 72.])

SingleModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5})
SingleModel.set_magnification_methods([0., 'point_source', 72.])

# initialise
t = BinaryModel.set_times(n_epochs = 500)
i = np.where(np.logical_and(0 <= t, t <= 72))
binaryError = BinaryModel.magnification(t[i])/20 + 0.25



## PLOT LIGHTCURVES ##

# plot binary
BinaryModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
plt.title('Weak binary lightcurve')
plt.xlabel('Time [days]')
plt.ylabel('Magnification [?]')
err = mpatches.Patch(label='binaryError', alpha=0.5)
plt.legend(handles=[err])
lower = BinaryModel.magnification(t[i]) - binaryError / 2
upper = BinaryModel.magnification(t[i]) + binaryError / 2
plt.fill_between(t[i], lower, upper, alpha = 0.25)
#plt.axis('square')!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.savefig('Plots/curve-weak-binary.png')
plt.clf()

# plot single
SingleModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
plt.title('Single lightcurve')
plt.xlabel('Time [days]')
plt.ylabel('Magnification [?]')
plt.savefig('Plots/curve-single.png')
plt.clf()

r=r

## PLOT POSTERIOR SURFACES ##

Data = mm.MulensData(data_list=[t, BinaryModel.magnification(t), BinaryModel.magnification(t)/20 + 0.25], phot_fmt='flux', chi2_fmt='flux')

density = 5
print(chi2(36, 0.133, 61.5, 0.00096, 0.0039, 1.120, 223.8, Data))

# extents
yLower = 1e-6
yUpper = 0.1
xLower = 0.75
xUpper = 1.25

yaxis = np.linspace(yLower, yUpper, density)
xaxis = np.linspace(xLower, xUpper, density)
result = np.zeros((density, density))
x = -1
y = -1

for i in yaxis:
    x += 1
    y = -1
    for j in xaxis:
        y += 1
        result[x][y] = chi2(36, 0.133, 61.5, 0.00096, i, j, 223.8, Data)

result = np.flip(result, 0) # So lower bounds meet

plt.imshow(np.sqrt(result), interpolation='none', extent=[xLower, xUpper, yLower, yUpper,], aspect=(xUpper-xLower) / (yUpper-yLower)) #cmap = plt.cm.BuPu_r
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q [Unitless]')
plt.title('Mass Fraction / Separation Chi^2 Statistic')
cbar = plt.colorbar(fraction = 0.046, pad = 0.04) # empirical nice auto sizing
cbar.set_label('sqrt(Chi^2)', rotation = 90)
plt.scatter(1.120, 0.0039, c = [(1,0,0)], marker = 7)
plt.savefig('Plots/temp.png')
plt.clf()