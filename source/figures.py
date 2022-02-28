"""Figures for project report."""


#from turtle import color
import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import sampling
import light_curve_simulation
import matplotlib.patches as mpatches
import plotting
import copy

if __name__ == "__main__":
        
    plotting.style()

    if True:
        ts = [10, 20]
        s=2.5
        n=7200

        theta_r = sampling.State(truth=[0.5, 15, 0.1, 10, 0.01, 0.3, 60])
        data=light_curve_simulation.synthetic_binary(theta_r, n, 23, seed = 42)
        plt.scatter(y=data.flux, x=data.time, s=s, label=r's = 0.3', c='tab:orange')

        #theta_r.truth[5] = 0.4
        #data=light_curve_simulation.synthetic_binary(theta_r, n, 23, seed = 43)
        #plt.scatter(y=data.flux, x=data.time, s=s, label='0.4', c='tab:purple')

        #theta_r.truth[5] = 0.5
        #data=light_curve_simulation.synthetic_binary(theta_r, n, 23, seed = 44)
        #plt.scatter(y=data.flux, x=data.time, s=s, label='0.4', c='tab:orange')

        theta_r.truth[5] = 0.6
        data=light_curve_simulation.synthetic_binary(theta_r, n, 23, seed = 45)
        plt.scatter(y=data.flux, x=data.time, s=s, label=r's = 0.6', c='tab:blue')



        plt.xlim(ts)
        plt.ylim([0.5, 6.75])

        plt.xlabel('time [days]', fontsize=14)
        plt.ylabel('normalised flux', fontsize=14)
        plt.tick_params(axis="both", which="major", labelsize=10)



        plt.gca().tick_params(which='both', top=True, right=True, direction='in')
        plt.minorticks_on()
        plt.gca().tick_params(which='major', length=10)
        plt.gca().tick_params(which='minor', length=5)

        import matplotlib.ticker as ticker
        plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(5))


        plt.gca().set_aspect('equal')

        plt.legend(fontsize = 14, frameon = False, markerscale=3, handletextpad=0.1)

        plt.tight_layout()

        #plt.gcf().set_size_inches(18.5, 10.5)
        #plt.savefig('figures/binary_family.pdf', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()


        theta_r.truth[5] = 0.825
        plotting.flux(1, theta_r, [5, 25], caustics=0.005, label=r's < 1.0', color='tab:orange')


        theta_r.truth[5] = 1.0
        plotting.flux(1, theta_r, [5, 25], caustics=0, label=r's = 1.0', color='tab:blue')



        plt.legend(fontsize = 14, frameon = False, markerscale=7, handletextpad=0.1)

        plt.xlim([-0.5, 0.35])
        plt.ylim([-0.21, 0.21])

        plt.gca().tick_params(which='both', top=True, right=True, direction='in')
        plt.minorticks_on()
        plt.gca().tick_params(which='major', length=10, labelsize=10)
        plt.gca().tick_params(which='minor', length=5)

        plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(7))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(7))

        plt.gca().set_box_aspect(1)

        #plt.axis('square')

        #plt.gca().set_aspect('equal')

        plt.tight_layout()


        #plt.gcf().set_size_inches(18.5, 18.5)
        plt.savefig('figures/caustic_family.pdf', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()







    if False:

        theta_r = sampling.State(truth=[50, 0.5, 25, 0.5, 1.0, 295])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [30, 62], caustics=0.02, color="purple")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #plt.savefig('figures/binary.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.5, 25, 0.5, 1.0, 295])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 55], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary1.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.45, 25, 0.5, 1.0, 287])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 55], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary2.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.4, 25, 0.5, 1.0, 282])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 60], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary3.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.35, 25, 0.5, 1.0, 277])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 65], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary4.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.3, 25, 0.5, 1.0, 270])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 65], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary5.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()

        theta_r = sampling.State(truth=[50, 0.25, 25, 0.5, 1.0, 263])
        ts = [0, 72]

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        plt.axes([0.225, 0.505, 0.4, 0.4])
        plotting.magnification(1, theta_r, [35, 66], caustics=0.02, color="orange")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig('figures/binary6.png', bbox_inches="tight", dpi=100, transparent=True)
        plt.clf()


        theta_r = sampling.State(truth=[50, 0.5, 10])
        ts = [0, 72]

        plotting.magnification(0, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        #plt.savefig('figures/single.png', bbox_inches="tight", dpi=50, transparent=True)
        plt.clf()


    if False:


        theta_r = sampling.State(truth=[36, 0.133, 61.5])
        ts = [10, 72-10]

        theta_t0 = copy.deepcopy(theta_r)
        theta_t0.truth[0] = theta_t0.truth[0] - 15
        plotting.magnification(0, theta_t0, ts, label=r"$\downarrow t_0$", color="blue")
        
        theta_u0 = copy.deepcopy(theta_r)
        theta_u0.truth[1] = theta_u0.truth[1] - 0.02
        plotting.magnification(0, theta_u0, ts, label=r"$\downarrow u_0$", color="orangered")

        theta_tE = copy.deepcopy(theta_r)
        theta_tE.truth[2] = theta_tE.truth[2] - 25
        plotting.magnification(0, theta_tE, ts, label=r"$\downarrow t_E$", color="purple")

        plotting.magnification(0, theta_r, ts, color="black")

        size = plt.gcf().get_size_inches()

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(fontsize = 17, frameon = False, handlelength=0.7, labelspacing=0.25)

        plt.savefig('figures/single-params.png', bbox_inches="tight", dpi=250, transparent=True)
        plt.clf()



        theta_r = sampling.State(truth=[36, 0.133, 61.5, 0.009, 1.10, 180])
        ts = [10, 72-10]

        theta_q = copy.deepcopy(theta_r)
        theta_q.truth[3] = theta_q.truth[3] + 0.0015
        plotting.magnification(1, theta_q, ts, label=r"$\uparrow q$", color="purple")
        
        theta_s = copy.deepcopy(theta_r)
        theta_s.truth[4] = theta_s.truth[4] + 0.04
        plotting.magnification(1, theta_s, ts, label=r"$\uparrow s$", color="orangered")

        theta_a = copy.deepcopy(theta_r)
        theta_a.truth[5] = theta_a.truth[5] + 135
        plotting.magnification(1, theta_a, [33, 72-10], label=r"$\uparrow \alpha$", color="blue")

        plotting.magnification(1, theta_r, ts, color="black")

        plt.xlabel('Time [days]')
        plt.ylabel('Magnification')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(fontsize = 17, frameon = False, handlelength=0.7, labelspacing=0.25)
        plt.tight_layout()

        plt.axes([0.155, 0.515, 0.4, 0.4])
        plotting.magnification(1, theta_q, [5+10, 45-10], caustics=0.001, color="purple")
        plotting.magnification(1, theta_s, [5+10, 45-10], caustics=0.001, color="orangered")
        plotting.magnification(1, theta_a, [25+10, 65-10], caustics=0.006, color="blue")
        plotting.magnification(1, theta_r, [5+10, 45-10], caustics=0.006, color="black")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.gcf().set_size_inches(size)

        plt.savefig('figures/binary-params.png', bbox_inches="tight", dpi=250, transparent=True)
        plt.clf()

    if False:

        ts = [0, 72]
        n_epochs = 720
        epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

        noisy_data = light_curve_simulation.synthetic_single(sampling.State(truth=[36, 1.0, 5.0]), n_epochs, 23)
        noisy_lower = noisy_data.flux - noisy_data.err_flux
        noisy_upper = noisy_data.flux + noisy_data.err_flux

        plt.fill_between(epochs, noisy_lower, noisy_upper, alpha = 1.0, label = r'$\pm\sigma$', color = 'plum', linewidth=0.0)
        plt.scatter(epochs, noisy_data.flux, c='black', label=r'Signal', s=1)

        #clean_data = light_curve_simulation.synthetic_single(sampling.State([36, 1.0, 5.0]), n_epochs, 230, )
        #clean_lower = clean_data.flux - clean_data.err_flux
        #clean_upper = clean_data.flux + clean_data.err_flux

        #plt.fill_between(epochs, clean_lower, clean_upper, alpha = 1.0, label = r'$\pm3\sigma$', color = 'green', linewidth=0.0)
        #plt.scatter(epochs, clean_data.flux, c='lime', label=r'$s=0.2$', s=1)
        
        #legend
        main_leg = plt.legend(frameon=False, loc='upper right', handlelength=0.7)
        shapes = iter(main_leg.legendHandles)
        next(shapes)
        for handle in shapes:
            handle.set_sizes([15.0])

        plt.xlabel('Time [days]')
        plt.ylabel('Observed Flux')
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(fontsize = 15, frameon = False, handlelength=0.7, labelspacing=0.25)

        plt.savefig('figures/dirty-curves.png', bbox_inches="tight", dpi=250, transparent=True)
        plt.clf()


    if False:

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
        #plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 1.0, color = 'plum', linewidth=0.0)
        plt.scatter(epochs[0:w+t_del], data_1.flux[0:w+t_del], c='black', label=r'$s=0.2$', s=1)
        plt.scatter(epochs[0:w+t_del], data_3.flux[0:w+t_del], c='blue', label=r'$s=0.7$', s=1)
        
        #legend
        #main_leg = plt.legend(frameon=False, loc='lower right', handlelength=0.7)
        #shapes = iter(main_leg.legendHandles)
        #next(shapes)
        #for handle in shapes:
        #    handle.set_sizes([15.0])

        plt.xlabel('Time [days]')
        plt.ylabel('Observed Flux')
        plt.tick_params(axis="both", which="major", labelsize=12)


        #inset
        inset = plt.axes([0.395, 0.37, 0.49, 0.49])
        #inset.fill_between(epochs[v:w], lower[v:w], upper[v:w], alpha = 1.0, label = r'$\pm3\sigma$', color = 'black', linewidth=0.0)
        inset.scatter(epochs[v:w], data_1.flux[v:w], c='black', label = r'$s=0.2$', s=1)
        inset.scatter(epochs[v:w], data_3.flux[v:w], c='blue', label = r'$s=0.7$', s=1)
        plt.legend(fontsize = 13.5, frameon = False, handlelength=0.7, labelspacing=0.25, loc='upper left')
        plt.tick_params(axis="both", which="major", labelsize=12)

        #inset.axes.get_yaxis().set_ticklabels([])
        #inset.tick_params(axis="y", direction="in", pad=-0)

        #residual
        frame_resid = plt.axes([0.125, -0.2, 0.775, 0.175])
        residuals = data_1.flux[0:w+t_del]-data_3.flux[0:w+t_del]
        lower = -3*data_1.err_flux[0:w+t_del]
        upper = 3*data_1.err_flux[0:w+t_del]
        plt.fill_between(epochs[0:w+t_del], lower[0:w+t_del], upper[0:w+t_del], alpha = 1, label = r'$\pm3\sigma$', color = 'plum', linewidth=0.0)
        plt.plot(epochs[0:w+t_del], residuals, c="black")
        frame_resid.set_xticklabels([])
        frame_resid.xaxis.tick_top()
        frame_resid.set_ylabel('Residual')
        plt.legend(fontsize = 13.5, frameon = False, handlelength=0.7, labelspacing=0.25, loc='upper right')
        plt.tick_params(axis="both", which="major", labelsize=12)

        plt.savefig('figures/evans-curves.png', bbox_inches="tight", dpi=250, transparent=True)
        plt.clf()




















