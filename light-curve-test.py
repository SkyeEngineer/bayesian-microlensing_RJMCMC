import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import Functions as mc
import matplotlib.patches as mpatches

# Synthetic Event Parameters/Initialisation
#SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.000002, 's': 4.2, 'alpha': 223.8})
SBModel.set_magnification_methods([0., 'VBBL', 144.])

SSModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5})
SSModel.set_magnification_methods([0., 'point_source', 72.])

#SBModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('Plots/curve-strong-binary.png')

t=SBModel.set_times(n_epochs = 500)
SBModel.plot_magnification(t_range=[0, 144], subtract_2450000=False, color='black')
plt.title('Weakly binary lightcurve')
red_patch = mpatches.Patch(label='Uncertainty', alpha=0.5)
plt.legend(handles=[red_patch])
plt.fill_between(t[np.where(np.logical_and(t>=0, t<=144))], SBModel.magnification(t[np.where(np.logical_and(t>=0, t<=144))])-SBModel.magnification(t[np.where(np.logical_and(t>=0, t<=144))])/10, SBModel.magnification(t[np.where(np.logical_and(t>=0, t<144))])+SBModel.magnification(t[np.where(np.logical_and(t>=0, t<=144))])/10, alpha=0.25)
#plt.axis('square')
plt.savefig('Plots/curve-weak-binary.png')
plt.clf()

SSModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
plt.savefig('Plots/curve-weak-single.png')
plt.clf()

d=d

t=SBModel.set_times(n_epochs = 100)
# Generate Synthetic Lightcurve
Data = mm.MulensData(data_list=[t, SSModel.magnification(t), SSModel.magnification(t)/20], phot_fmt='flux', chi2_fmt='flux')

th1 = np.array([36., 0.133, 61.5])
th2 = np.array([36., 0.133, 61.5, 0.00096, 0.00001, 2.0, 223.8])
print(np.exp(mc.Likelihood(1, Data, th1, 5)))
print(np.exp(mc.Likelihood(2, Data, th2, 5)))
c=d


# Get chi2 by creating a new model and fitting to previously generated data
def func(t_0, u_0, t_E, rho, q, s, alpha, Data):
    my_f_model = mm.Model({'t_0': t_0, 'u_0': u_0,'t_E': t_E, 'rho': rho, 'q': q, 's': s,'alpha': alpha})
    my_f_model.set_magnification_methods([0., 'VBBL', 72.])

    my_event = mm.Event(datasets=Data, model=my_f_model)
    #print(my_event.get_chi2())
    return my_event.get_chi2()
    #return np.exp(mc.Likelihood(2, Data, [t_0, u_0, t_E, rho, q, s, alpha,], 5))



'''

toaxis = np.linspace(2452843.06, 2452853.06, de)
teaxis = np.linspace(56.5, 66.5, de)
result=np.zeros((de,de))
x=-1
y=-1
for i in toaxis:
    x=x+1
    y=-1
    for j in teaxis:
        y=y+1
        result[x][y] = func(i, 0.133, j, 0.00096, 0.0039, 1.120, 223.8, my_data)

plt.imshow(result, cmap='hot', interpolation='none', extent=[56.5, 66.5, 2853.06, 2843.06,])
plt.xlabel('te [days]') # Set the y axis label of the current axis.
plt.ylabel('to-2450000 [days]') # Set a title.
plt.title('Closest Approach/Crossing Time Chi Squared over parameter space')
plt.savefig('Plots/teto.png')
'''
de=5
print(func(36, 0.133, 61.5, 0.00096, 0.0039, 1.120, 223.8, Data))

#a=1e-6
#b=1.0
#c=0.2
#d=5.0

a=1e-6
b=0.1
c=0.75
d=1.25

qaxis = np.linspace(a, b, de)
saxis = np.linspace(c, d, de)
result=np.zeros((de,de))
x=-1
y=-1
for i in qaxis:
    x=x+1
    y=-1
    for j in saxis:
        y=y+1
        result[x][y] = func(36, 0.133, 61.5, 0.00096, i, j, 223.8, Data)

result=np.flip(result, 0)

plt.imshow(np.sqrt(result), interpolation='none', extent=[c, d, a, b,], aspect=(d-c)/(b-a))#plt.cm.BuPu_r
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q [Unitless]')
plt.title('Mass Fraction / Separation Chi^2 Statistic')
cbar=plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('sqrt(Chi^2)', rotation=90)
plt.scatter(1.120, 0.0039, c=[(1,0,0)], marker=7)#2 is chevron triangle
plt.savefig('Plots/temp.png')

#plt.figure()
#SBModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('Plots/curve-strong-binary.png')