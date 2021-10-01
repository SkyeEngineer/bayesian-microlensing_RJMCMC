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





def centre_offsets_pointilism(supset_model, subset_model, symbols, name = '', dpi = 100):

    supset_offsets = (supset_model.sampled.states_array(scaled = True) - supset_model.centre.scaled[:, np.newaxis])
    subset_offsets = (subset_model.sampled.states_array(scaled = True) - subset_model.centre.scaled[:, np.newaxis])
    n_dim = subset_model.D

    style()
    # construct shape with corner
    figure = corner.corner(subset_offsets.T)

    # font/visibility
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

    # extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))


    # Loop over the diagonal to remove from plot
    for i in range(n_dim):
        ax = axes[i, i]
        ax.cla()
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        

    # loop over lower triangle
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            
            # overlay points
            ax.scatter(subset_offsets[xi, :], subset_offsets[yi, :], c = np.linspace(0.0, 1.0, subset_model.sampled.n), cmap = 'winter', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)
            ax.scatter(supset_offsets[xi, :], supset_offsets[yi, :], c = np.linspace(0.0, 1.0, supset_model.sampled.n), cmap = 'spring', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)

            if yi == n_dim - 1: # last row
                ax.set_xlabel(symbols[xi])
                ax.tick_params(axis = 'x', labelrotation = 45)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0: # first column
                ax.set_ylabel(symbols[yi])
                ax.tick_params(axis = 'y', labelrotation = 45)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

    figure.savefig('results/' + name + '-centreed-pointilism.png', bbox_inches = "tight", dpi = dpi, transparent=True)
    figure.clf()

    return