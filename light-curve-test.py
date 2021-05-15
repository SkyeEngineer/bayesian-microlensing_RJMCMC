import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np

def chi2_for_model(theta, event, parameters_to_fit):
    """for given event set attributes from parameters_to_fit
    (list of str) to values from the theta list"""
    for (key, parameter) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, parameter, theta[key])
    return event.get_chi2()

my_1S2L_model = mm.Model({'t_0': 2452848.06, 'u_0': 0.133,
     't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120,
     'alpha': 223.8})

my_1S2L_model.set_magnification_methods([2452833., 'VBBL', 2452845.])
t=my_1S2L_model.set_times()
my_data = mm.MulensData(data_list=[t, my_1S2L_model.magnification(t), my_1S2L_model.magnification(t)*0+0.003]) #orignally 0.03

print(my_1S2L_model.magnification(t))

def func(t_0, u_0, t_E, rho, q, s, alpha, my_data):
    my_f_model = mm.Model({'t_0': t_0, 'u_0': u_0,
     't_E': t_E, 'rho': rho, 'q': q, 's': s,
     'alpha': alpha})
    my_f_model.set_magnification_methods([2452833., 'VBBL', 2452845.])

    my_event = mm.Event(datasets=my_data, model=my_f_model)
    return my_event.get_chi2()

de=15
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
plt.savefig('teto.png')

qaxis = np.linspace(0.001, 0.2, de)
saxis = np.linspace(1, 1.3, de)
result=np.zeros((de,de))
x=-1
y=-1
for i in qaxis:
    x=x+1
    y=-1
    for j in saxis:
        y=y+1
        result[x][y] = func(2452848.06, 0.133, 61.5, 0.00096, i, j, 223.8, my_data)

plt.imshow(result, cmap='hot', interpolation='none', extent=[1, 1.3, 0.2, 0.001,])
plt.xlabel('s [Einstein Ring Radius]') # Set the y axis label of the current axis.
plt.ylabel('q') # Set a title.
plt.title('Separation/Mass Fraction Chi Squared over parameter space')
plt.savefig('sq.png')


#parameters_to_fit = ['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha']
#initial_guess = [t_0, u_0, t_E, rho, q, s, alpha]
#initial_guess = [2452848.06, 0.133, 61.5, 0.00096, 0.0039, 1.120, 223.8]


#print(chi2_for_model(initial_guess, my_event, parameters_to_fit))

#plt.figure()
#my_1S2L_model.plot_magnification(t_range=[2452810, 2452890],
#    subtract_2450000=True, color='black')
#plt.savefig('books_read.png')