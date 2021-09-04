# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Lightcurve plotting]

#from interfaceing import get_model_centers, get_posteriors
from os import error
from types import LambdaType
from numpy.core.defchararray import title
from numpy.core.fromnumeric import size
from numpy.core.function_base import linspace
import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import main_functions as f
import matplotlib.patches as mpatches
import plot_functions as pltf
import copy


pltf.Style

#theta = [36, 0.1, 36, 0.8, 0.25, 123]
#theta = [36, 0.1, 36, 0.01, 0.01, 0.6, 123]
#theta = [36, 0.1, 36, 0.001, 0.005, 0.8, 89] symmetric casutic grazing
#n_epochs = 720
#epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
#for a in [73, 123, 304]:
    #print(a)
#    theta[-1] = a
#    data = f.Synthetic_Light_Curve(theta, 1, n_epochs, 2300)
#    plt.plot(data.time, data.flux, linewidth = 0.99, label = a)

#theta = get_model_centers(get_posteriors(1), data.flux)
#data = f.Synthetic_Light_Curve(theta, 1, n_epochs, 230.0)
#plt.plot(data.time, data.flux, linestyle = "dashed", label = theta[-1])
#plt.legend()
#print(theta)
#plt.savefig('temp.png')
#plt.clf()

#theta_r = [36, 0.133, 61.5]
#ts = [0, 72]


#pltf.PlotLightcurve(0, theta_r, r"$\uparrow t_0$", "blue", 1, False, ts)

#theta_r = [36, 0.133, 61.5, 0.009, 1.10, 180]



#theta_q = copy.deepcopy(theta_r)
#theta_q[3] = theta_q[3] + 0.0015
#pltf.PlotLightcurve(1, theta_r, r"$\uparrow q$", "red", 1, False, ts)
#plt.savefig('temp.png')
#plt.clf()
#throw=throw

if False:
    # REPORT QUALITY PLOT

    ts = [0, 72]
    n_epochs = 7200
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    v = 3450
    w = 3750
    t_del = 100

    data_3 = f.Synthetic_Light_Curve([36, 0.1, 10, 0.01, 0.6, 60], 1, n_epochs, 23)
    data_2 = f.Synthetic_Light_Curve([36, 0.1, 10, 0.01, 0.2, 60], 1, n_epochs, 23)
    data_1 = f.Synthetic_Light_Curve([36, 0.1, 10], 0, n_epochs, 23)

    lower = data_1.flux - 3*data_1.err_flux
    upper = data_1.flux + 3*data_1.err_flux

    #main
    plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 0.375, label = r'$\pm3\sigma$', color = 'red')
    plt.plot(epochs[0:w+t_del], data_1.flux[0:w+t_del], c='black', linewidth=0.75, label=r'$F$')
    plt.legend(frameon=False, loc='lower right', handlelength=0.7)
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    #inset
    inset = plt.axes([0.17, 0.455, 0.41, 0.41])
    inset.scatter(epochs[v:w], data_3.flux[v:w], c='cyan', s=1, label = 's = 0.6')
    inset.scatter(epochs[v:w], data_1.flux[v:w], c='black', s=1, label = 's = 0')
    inset.axes.get_yaxis().set_ticklabels([])
    inset.tick_params(axis="y", direction="in", pad=-0)
    inset_leg = plt.legend(frameon=False, handletextpad=0.1, columnspacing=0.01, loc='upper right') #, borderpad=0.1, 
    for handle in inset_leg.legendHandles:
        handle.set_sizes([6.0])

    #residual
    frame_resid = plt.axes([0.125, -0.1, 0.775, 0.1])
    plt.plot(epochs[0:w+t_del], data_1.flux[0:w+t_del]-data_3.flux[0:w+t_del], c="black")
    frame_resid.set_xticklabels([])
    frame_resid.xaxis.tick_top()
    frame_resid.set_ylabel('Residual')

    plt.savefig('Plots/evans-curves.png', bbox_inches="tight", dpi=500, transparent=True)
    plt.clf()

if True:
    # REPORT QUALITY PLOT

    ts = [0, 72]
    n_epochs = 7200
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    v = 3450
    w = 3750
    t_del = 100

    data_3 = f.Synthetic_Light_Curve([36, 0.1, 10, 0.01, 0.6, 60], 1, n_epochs, 23)
    data_2 = f.Synthetic_Light_Curve([36, 0.1, 10, 0.01, 0.2, 60], 1, n_epochs, 23)
    data_1 = f.Synthetic_Light_Curve([36, 0.1, 10], 0, n_epochs, 23)

    lower = data_1.flux - 3*data_1.err_flux
    upper = data_1.flux + 3*data_1.err_flux

    #main
    plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 0.375, label = r'$\pm3\sigma$', color = 'red')
    plt.plot(epochs[0:w+t_del], data_1.flux[0:w+t_del], c='black', linewidth=0.75, label=r'$F$')
    plt.legend(frameon=False, loc='lower right', handlelength=0.7)
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    #inset
    inset = plt.axes([0.17, 0.455, 0.41, 0.41])
    inset.scatter(epochs[v:w], data_3.flux[v:w], c='cyan', s=1, label = 's = 0.6')
    inset.scatter(epochs[v:w], data_1.flux[v:w], c='black', s=1, label = 's = 0')
    inset.axes.get_yaxis().set_ticklabels([])
    inset.tick_params(axis="y", direction="in", pad=-0)
    inset_leg = plt.legend(frameon=False, handletextpad=0.1, columnspacing=0.01, loc='upper right') #, borderpad=0.1, 
    for handle in inset_leg.legendHandles:
        handle.set_sizes([6.0])

    #residual
    frame_resid = plt.axes([0.125, -0.1, 0.775, 0.1])
    plt.plot(epochs[0:w+t_del], data_1.flux[0:w+t_del]-data_3.flux[0:w+t_del], c="black")
    frame_resid.set_xticklabels([])
    frame_resid.xaxis.tick_top()
    frame_resid.set_ylabel('Residual')

    plt.savefig('Plots/evans-curves.png', bbox_inches="tight", transparent=True)
    plt.clf()

close=close

if False:

    ts = [0, 72]

    pltf.PlotLightcurve(0, [45, 0.2, 20], r"$q=0$", "blue", 1, False, ts)

    pltf.PlotLightcurve(1, [45, 0.2, 20, 0.00001, 1.0, 300], r"$q=0.0025$", "green", 1, False, ts)

    pltf.PlotLightcurve(1, [45, 0.2, 20, 0.0001, 1.0, 300], r"$q=0.0035$", "yellow", 1, False, ts)

    pltf.PlotLightcurve(1, [45, 0.2, 20, 0.001, 1.0, 300], r"$q=0.1$", "red", 1, False, ts)




    plt.legend()

    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.tight_layout()
    plt.savefig('Plots/qCurves.png')
    plt.clf()


if True:
    #plt.grid()

    #theta_r = [36, 0.133, 61.5,  0.001, 0.008, 1.2, 300] # crash
    theta_r = [36, 1.0, 1, 0.07, 1.49, 225]
    # alpha = 2pi/3
    #mean of q (1-10e-6)/(ln(1/10e-6)) = 0.07
    #mean of s (5-0.2)/(ln(5/0.2)) = 1.49

    ts = [0, 72]

    for te in linspace(1, 2, 9):
        theta_te = copy.deepcopy(theta_r)
        theta_te[2] = te
        pltf.PlotLightcurve(1, theta_te, "te="+str(te), 'yellow', 1, False, ts)

    plt.legend()


    #plt.title('Binary lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.tight_layout()

    plt.axes([0.125, 0.55, 0.3, 0.3])
    
    #theta_s = copy.deepcopy(theta_r)
    #theta_s[4] = s
    pltf.PlotLightcurve(1, theta_r, " ", 'yellow', 1, True, [34, 36])
    
#    ax = plt.gca()
#    ax.axes.xaxis.set_visible(False)
#    ax.axes.yaxis.set_visible(False)
    #plt.legend(title = 'caustic', fontsize = 9)

    plt.savefig('Plots/WidthParamCurve.png')
    plt.clf()

#throw=throw

if True:
    #plt.grid()

    #theta_r = [36, 0.133, 61.5,  0.001, 0.008, 1.2, 300] # crash
    theta_r = [36, 1.0, 6, 0.07, 0.2, 120]
    # alpha = 2pi/3
    #mean of q (1-10e-6)/(ln(1/10e-6)) = 0.07 max too low 4.69 aligns with 6 is moa data. Accurate since single lens. No binary data 
    #mean of tE approximately exp(10^1.15 - (10^0.45)^2/2) max approximately 5, E = 0.2

    ts = [0, 72]

    for s in [0.2, 1, 2, 3]:#linspace(0.2, 3.0, 10):
        theta_s = copy.deepcopy(theta_r)
        theta_s[4] = s
        pltf.PlotLightcurve(1, theta_s, "s="+str(s), 'blue', 0.5, False, ts)




    #plt.title('Binary lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.tight_layout()

    #plt.axes([0.125, 0.55, 0.3, 0.3])
    
    #for s in linspace(0.2, 5, 2):
    #    theta_s = copy.deepcopy(theta_r)
    #    theta_s[4] = s
    #    pltf.PlotLightcurve(1, theta_s, " ", 'blue', 0.1, True, [26, 46])
    
    #ax = plt.gca()
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    #plt.legend(title = 'caustic', fontsize = 9)
    plt.legend()
    plt.savefig('Plots/MeanParamCurve.png')
    plt.clf()

if True:
    theta_r = [36, -0.1, 61.5, 0.01, 0.2, 25]

    ts = [0, 72]

    for s in linspace(0.2, 3.0, 10):
        theta_s = copy.deepcopy(theta_r)
        theta_s[4] = s
        pltf.PlotLightcurve(1, theta_s, "s="+str(s), 'blue', 0.5, False, ts)


    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.tight_layout()

    plt.legend()

    plt.axes([0.125, 0.55, 0.3, 0.3])
    theta_s = copy.deepcopy(theta_r)
    pltf.PlotLightcurve(1, theta_s, " ", 'blue', 0.1, True, [26, 46])

    theta_s = copy.deepcopy(theta_r)
    theta_s[1] = 0.1
    pltf.PlotLightcurve(1, theta_s, " ", 'blue', 0.1, True, [26, 46])

    plt.savefig('Plots/MeanParamCurve.png')
    plt.clf()

throw=throw



if True:
    #plt.grid()

    #theta_r = [36, 0.133, 61.5,  0.001, 0.008, 1.2, 300] # crash
    theta_r = [36, 0.133, 61.5, 0.009, 1.10, 180]

    ts = [0, 72]



    theta_q = copy.deepcopy(theta_r)
    theta_q[3] = theta_q[3] + 0.0015
    pltf.PlotLightcurve(1, theta_q, r"$\uparrow q$", "red", 1, False, ts)
    
    theta_s = copy.deepcopy(theta_r)
    theta_s[4] = theta_s[4] + 0.04
    pltf.PlotLightcurve(1, theta_s, r"$\uparrow s$", "orange", 1, False, ts)

    theta_a = copy.deepcopy(theta_r)
    theta_a[5] = theta_a[5] + 90
    #theta_a[6] = theta_a[6] - 60
    pltf.PlotLightcurve(1, theta_a, r"$\updownarrow \alpha$", "blue", 1, False, [28, 72])

    plt.legend()

    pltf.PlotLightcurve(1, theta_r, "base", "black", 1, False, ts)

    #plt.title('Binary lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    plt.tight_layout()

    plt.axes([0.125, 0.55, 0.3, 0.3])
    pltf.PlotLightcurve(1, theta_q, " ", "red", 1, True, [5, 45])
    pltf.PlotLightcurve(1, theta_s, " ", "orange", 1, True, [5, 45])
    pltf.PlotLightcurve(1, theta_a, " ", "blue", 1, True, [20, 60])
    pltf.PlotLightcurve(1, theta_r, " ", "black", 1, True, [5, 45])
    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    #plt.legend(title = 'caustic', fontsize = 9)





    plt.savefig('Plots/BinaryParamCurve.png')
    plt.clf()

if True:
    #plt.grid()


    theta_r = [36, 0.133, 61.5]
    ts = [0, 72]

    #theta_fs = copy.deepcopy(theta_r)
    #theta_fs[0] = theta_fs[0] - 0.1
    #pltf.PlotLightcurve(0, theta_fs, r"$\downarrow f_s$", "purple", 1, False, ts)

    theta_t0 = copy.deepcopy(theta_r)
    theta_t0[0] = theta_t0[0] + 15
    pltf.PlotLightcurve(0, theta_t0, r"$\uparrow t_0$", "blue", 1, False, ts)
    
    theta_u0 = copy.deepcopy(theta_r)
    theta_u0[1] = theta_u0[1] - 0.02
    pltf.PlotLightcurve(0, theta_u0, r"$\downarrow u_0$", "orange", 1, False, ts)

    theta_tE = copy.deepcopy(theta_r)
    theta_tE[2] = theta_tE[2] + 25
    pltf.PlotLightcurve(0, theta_tE, r"$\uparrow t_E$", "red", 1, False, ts)

    #theta_p = copy.deepcopy(theta_r)
    #theta_p[3] = theta_tE[3] + 0.1
    #pltf.PlotLightcurve(0, theta_p, r"$\uparrow \rho$", "green", 1, False, ts)

    plt.legend()

    pltf.PlotLightcurve(0, theta_r, "base", "black", 1, False, ts)

    #plt.title('Single lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    plt.tight_layout()
    plt.savefig('Plots/SingleParamCurve.png')
    plt.clf()


    theta_r = [36, 0.133, 61.5, 0.0001]
    theta_fs = 1 - 0.1
    model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho'], theta_r)))
    model.set_magnification_methods([0., 'finite_source_uniform_Gould94', 72.])
    A = (model.magnification(epochs) - 1.0)*theta_fs + 1.0
    plt.plot(epochs, A, color = 'blue', label = r"$\downarrow f_s$", alpha = 1)


    theta_p = copy.deepcopy(theta_r)
    theta_p[3] = theta_p[3] + 0.0999
    model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho'], theta_p)))
    model.set_magnification_methods([0., 'finite_source_uniform_Gould94', 72.])
    A = model.magnification(epochs)
    plt.plot(epochs, A, color = 'red', label = r"$\uparrow \rho$", alpha = 1)


    plt.legend()
    #theta_p = copy.deepcopy(theta_r)
    #theta_p[3] = theta_p[3] - 0.0001
    model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho'], theta_r)))
    model.set_magnification_methods([0., 'finite_source_uniform_Gould94', 72.])
    A = model.magnification(epochs)
    plt.plot(epochs, A, color = 'black', label = "base", alpha = 1)

    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    plt.tight_layout()
    plt.savefig('Plots/FiniteParamCurve.png')
    plt.clf()


close=close

# Get chi2 by creating a new model and fitting to previously generated data
def chi2(t_0, u_0, t_E, rho, q, s, alpha, Data):
    Model = mm.Model({'t_0': t_0, 'u_0': u_0,'t_E': t_E, 'rho': rho, 'q': q, 's': s,'alpha': alpha})
    Model.set_magnification_methods([0., 'VBBL', 72.])
    Event = mm.Event(datasets = Data, model=Model)
    return Event.get_chi2()



# Synthetic Events
#BinaryModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.001, 'q': 0.00005, 's': 1.12, 'alpha': 200.8})
theta = [36, 0.833, 31.5, 0.001, 0.02, 1.10, 180]
BinaryModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta)))
BinaryModel.set_magnification_methods([0., 'VBBL', 72.])

SingleModel = mm.Model({'t_0': 36, 'u_0': 0.833, 't_E': 31.5})
SingleModel.set_magnification_methods([0., 'point_source', 72.])

##theta = [36, 0.833, 31.5, 0.001, 0.002, 1.10, 180]
#SingleModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta)))
#SingleModel.set_magnification_methods([0., 'VBBL', 72.])

# initialise
t = BinaryModel.set_times(n_epochs = 720)
i = np.where(np.logical_and(0 <= t, t <= 72))
binaryError = BinaryModel.magnification(t[i])/30 + 0.2



## NOISE

true_data = BinaryModel.magnification(t[i])

    # Noise is due to poisson process
    # These values are calculated from comparing the zero-point magnitude (1 count/sec)
    # to the magnitude range (20-25) and the exposure time, then calculating the SNR ratio, 
    # which is the sqrt(count)

signal_to_noise_baseline = np.random.uniform(23.0, 230.0)

noise = np.random.normal(0.0, np.sqrt(true_data) / signal_to_noise_baseline, len(t[i])) 

model_data = true_data + 0#noise

plt.plot(t[i], model_data)
plt.plot(t[i], SingleModel.magnification(t[i]), color = 'red')
#plt.title('Weak binary lightcurve')
plt.xlabel('Time [days]')
plt.ylabel('Magnification')
#err = mpatches.Patch(label='binaryError', alpha=0.5)
#plt.legend(handles=[err])
#lower = BinaryModel.magnification(t[i]) - binaryError / 2
#upper = BinaryModel.magnification(t[i]) + binaryError / 2
#plt.fill_between(t[i], lower, upper, alpha = 0.25)
#plt.axis('square')!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.tight_layout()
plt.savefig('Plots/Temp.png')
plt.clf()

throw=throw

## PLOT LIGHTCURVES ##


# plot binary
plt.grid()
BinaryModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red')
#plt.title('Weak binary lightcurve')
plt.xlabel('Time [days]')
plt.ylabel('Magnification')
#err = mpatches.Patch(label='binaryError', alpha=0.5)
#plt.legend(handles=[err])
#lower = BinaryModel.magnification(t[i]) - binaryError / 2
#upper = BinaryModel.magnification(t[i]) + binaryError / 2
#plt.fill_between(t[i], lower, upper, alpha = 0.25)
#plt.axis('square')!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.tight_layout()
plt.savefig('Plots/confusing-curve.png')
plt.clf()



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


## PLOT POSTERIOR SURFACES ##
error = BinaryModel.magnification(t[i])/100+0.5/BinaryModel.magnification(t[i])
Data = mm.MulensData(data_list=[t[i], BinaryModel.magnification(t[i]), error],  phot_fmt = 'flux', chi2_fmt = 'flux')

Data.plot(show_errorbars=True)
plt.savefig('temp.png')
plt.clf()

throw=throw

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
