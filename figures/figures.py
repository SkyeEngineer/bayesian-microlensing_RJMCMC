"""Figures for project report."""

#from interfaceing import get_model_centres, get_posteriors
from os import error
from types import LambdaType
from numpy.core.defchararray import title
from numpy.core.fromnumeric import size
from numpy.core.function_base import linspace
import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import sampling
import light_curve_simulation
import matplotlib.patches as mpatches
import plotting as pltf
import copy


pltf.style()

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

#theta = get_model_centres(get_posteriors(1), data.flux)
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
    #plt.grid()

    #theta_r = [36, 0.133, 61.5,  0.001, 0.008, 1.2, 300] # crash
    theta_r = [36, 0.133, 61.5, 0.009, 1.10, 180]

    ts = [10, 72-10]



    theta_q = copy.deepcopy(theta_r)
    theta_q[3] = theta_q[3] + 0.0015
    pltf.PlotLightcurve(1, theta_q, r"$\uparrow q$", "red", 1, False, ts)
    
    theta_s = copy.deepcopy(theta_r)
    theta_s[4] = theta_s[4] + 0.04
    pltf.PlotLightcurve(1, theta_s, r"$\uparrow s$", "orange", 1, False, ts)

    theta_a = copy.deepcopy(theta_r)
    theta_a[5] = theta_a[5] + 135
    #theta_a[6] = theta_a[6] - 60
    pltf.PlotLightcurve(1, theta_a, r"$\updownarrow \alpha$", "cyan", 1, False, [33, 72-10])

    pltf.PlotLightcurve(1, theta_r, "base", "black", 1, False, ts)
    plt.legend(frameon=False, handlelength=0.7)

    #plt.title('Binary lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    plt.tight_layout()

    plt.axes([0.125, 0.5, 0.4, 0.4])
    pltf.PlotLightcurve(1, theta_q, " ", "red", 1, True, [5+10, 45-10])
    pltf.PlotLightcurve(1, theta_s, " ", "orange", 1, True, [5+10, 45-10])
    pltf.PlotLightcurve(1, theta_a, " ", "cyan", 1, True, [25+10, 65-10])
    pltf.PlotLightcurve(1, theta_r, " ", "black", 1, True, [5+10, 45-10])
    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    #plt.legend(title = 'caustic', fontsize = 9)





    plt.savefig('Plots/binary-params.png')
    plt.clf()



if False:
    #plt.grid()


    theta_r = [36, 0.133, 61.5]
    ts = [10, 72-10]

    #theta_fs = copy.deepcopy(theta_r)
    #theta_fs[0] = theta_fs[0] - 0.1
    #pltf.PlotLightcurve(0, theta_fs, r"$\downarrow f_s$", "purple", 1, False, ts)

    theta_t0 = copy.deepcopy(theta_r)
    theta_t0[0] = theta_t0[0] + 15
    pltf.PlotLightcurve(0, theta_t0, r"$\uparrow t_0$", "cyan", 1, False, ts)
    
    theta_u0 = copy.deepcopy(theta_r)
    theta_u0[1] = theta_u0[1] - 0.02
    pltf.PlotLightcurve(0, theta_u0, r"$\downarrow u_0$", "orange", 1, False, ts)

    theta_tE = copy.deepcopy(theta_r)
    theta_tE[2] = theta_tE[2] + 25
    pltf.PlotLightcurve(0, theta_tE, r"$\uparrow t_E$", "red", 1, False, ts)

    #theta_p = copy.deepcopy(theta_r)
    #theta_p[3] = theta_tE[3] + 0.1
    #pltf.PlotLightcurve(0, theta_p, r"$\uparrow \rho$", "green", 1, False, ts)


    pltf.PlotLightcurve(0, theta_r, "base", "black", 1, False, ts)
    plt.legend(frameon=False, handlelength=0.7)

    #plt.title('Single lens parameterisation')
    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    plt.tight_layout()
    plt.savefig('Plots/single-params.png')
    plt.clf()

if False:
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

if True:
    # REPORT QUALITY PLOT

    ts = [0, 72]
    n_epochs = 720
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

    noisy_data = light_curve_simulation.synthetic_single(sampling.State(truth=[36, 1.0, 5.5]), n_epochs, 23)

    noisy_lower = noisy_data.flux - noisy_data.err_flux
    noisy_upper = noisy_data.flux + noisy_data.err_flux
    plt.fill_between(epochs, noisy_lower, noisy_upper, alpha = 1.0, label = r'$\pm3\sigma$', color = 'pink', linewidth=0.0)

    clean_data = light_curve_simulation.synthetic_single(sampling.State([36, 1.0, 5.5]), n_epochs, 230, )
    clean_lower = clean_data.flux - clean_data.err_flux
    clean_upper = clean_data.flux + clean_data.err_flux


    plt.fill_between(epochs, clean_lower, clean_upper, alpha = 1.0, label = r'$\pm3\sigma$', color = 'green', linewidth=0.0)
    plt.scatter(epochs, clean_data.flux, c='lime', label=r'$s=0.2$', s=1)
    plt.scatter(epochs, noisy_data.flux, c='red', label=r'$s=0.7$', s=1)
    
    #legend
    #main_leg = plt.legend(frameon=False, loc='lower right', handlelength=0.7)
    #shapes = iter(main_leg.legendHandles)
    #next(shapes)
    #for handle in shapes:
    #    handle.set_sizes([15.0])
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.savefig('figures/dirty-curves.png', bbox_inches="tight")#, dpi=500, transparent=True)
    plt.clf()


if False:
    # REPORT QUALITY PLOT

    ts = [0, 72]
    n_epochs = 720
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    v = 130
    w = 170
    t_del = 720-175#10

    data_3 = light_curve_simulation.synthetic_binary(sampling.State([15, 0.1, 10, 0.01, 0.7, 60]), n_epochs, 23, )
    #data_2 = f.Synthetic_Light_Curve([36, 0.1, 10, 0.01, 0.2, 60], 1, n_epochs, 23)
    data_1 = light_curve_simulation.synthetic_single(sampling.State(truth=[15, 0.1, 10, 0.01, 0.2, 60]), n_epochs, 23)

    lower = data_1.flux - 3*data_1.err_flux
    upper = data_1.flux + 3*data_1.err_flux

    #main
    plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 1.0, label = r'$\pm3\sigma$', color = 'black', linewidth=0.0)
    plt.scatter(epochs[0:w+t_del], data_1.flux[0:w+t_del], c='red', label=r'$s=0.2$', s=1)
    plt.scatter(epochs[0:w+t_del], data_3.flux[0:w+t_del], c='lime', label=r'$s=0.7$', s=1)
    
    #legend
    #main_leg = plt.legend(frameon=False, loc='lower right', handlelength=0.7)
    #shapes = iter(main_leg.legendHandles)
    #next(shapes)
    #for handle in shapes:
    #    handle.set_sizes([15.0])
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    #inset
    inset = plt.axes([0.17+0.3, 0.45, 0.41, 0.41])
    #inset.fill_between(epochs[v:w], lower[v:w], upper[v:w], alpha = 1.0, label = r'$\pm3\sigma$', color = 'black', linewidth=0.0)
    inset.scatter(epochs[v:w], data_3.flux[v:w], c='lime', s=1, label = r'$s=0.7$')
    inset.scatter(epochs[v:w], data_1.flux[v:w], c='red', s=1, label = r'$s=0.2$')

    #inset.axes.get_yaxis().set_ticklabels([])
    #inset.tick_params(axis="y", direction="in", pad=-0)

    #residual
    frame_resid = plt.axes([0.125, -0.125, 0.775, 0.1])
    residuals = data_1.flux[0:w+t_del]-data_3.flux[0:w+t_del]
    lower = -3*data_1.err_flux[0:w+t_del]
    upper = 3*data_1.err_flux[0:w+t_del]
    #plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 0.5, label = r'$F\pm3\sigma$', color = 'black', linewidth=0.0)
    plt.plot(epochs[0:w+t_del], residuals, c="black")
    frame_resid.set_xticklabels([])
    frame_resid.xaxis.tick_top()
    frame_resid.set_ylabel('Residual')

    plt.savefig('figures/evans-curves.png', bbox_inches="tight")#, dpi=500, transparent=True)
    plt.clf()

throw=throw


if False:
    # REPORT QUALITY PLOT

    ts = [0, 72]
    n_epochs = 7200
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

    data_1 = f.Synthetic_Light_Curve([36, -0.1, 61.5, 0.01, 0.2, 25], 1, n_epochs, 10000)
    data_2 = f.Synthetic_Light_Curve([36, -0.1, 61.5, 0.01, 0.8, 25], 1, n_epochs, 10000)
    data_3 = f.Synthetic_Light_Curve([36, -0.1, 61.5, 0.01, 1.3, 25], 1, n_epochs, 10000)
    data_4 = f.Synthetic_Light_Curve([36, -0.1, 61.5, 0.01, 5.0, 25], 1, n_epochs, 10000)

    #main
    plt.plot(epochs, data_1.flux, c='black', linewidth=0.75, label='s=0.2')
    plt.plot(epochs, data_2.flux, c='purple', linewidth=0.75, label='s=0.8')
    plt.plot(epochs, data_3.flux, c='magenta', linewidth=0.75, label='s=1.2')
    plt.plot(epochs, data_4.flux, c='red', linewidth=0.75, label='s=5.0')
    plt.legend(frameon=False, handlelength=0.7)
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.savefig('Plots/kennedy-curves.png', bbox_inches="tight", transparent=False)
    plt.clf()



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


if False:
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

if False:
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

if False:
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


















