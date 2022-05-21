from magnetno_nihalo import *
from scipy.fft import rfft, rfftfreq

dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)
# z = 1.5 - L #height of the magnet above the plain of the other magnets

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# magnets = extra_magnets([[0, 0], [0.3, 0.4]], x, y, 5, 0.7) 
magnets = configuration_of_magnets(x, y, 6, 0.7, False) 


# Initial conditions 
v = 1.2
# phi1 = 2*np.pi*np.random.random()
phi = np.pi/3
x0 = 0.0
y0 = 0.5
v_x0 = v * np.cos(phi)
v_y0 = v * np.sin(phi)
initial_conditions1 = [x0, y0, v_x0, v_y0]


# ############################  STATISTICS ############################
def average_of_solution(solution_x, solution_y):
    average_x = []
    average_y = []
    array_x = np.array(solution_x)
    array_y = np.array(solution_y)
    for i in range(len(solution_x[0])):
        average_x.append( np.sum(array_x[:, i]) / len(array_x[:, i]) )
        average_y.append( np.sum(array_y[:, i]) / len(array_y[:, i]) )
    matrix = np.array([average_x, average_y])
    return matrix.reshape(len(average_x), 2)


#10 repetitions for statistics
def solution_different_initial_v(list_of_initial_velocity, repetition, t):
    different_initial_v = dict()
    n = len(t)
    dt = t[1] - t[0]
    freq = rfftfreq(n, dt)
    # magnets = configuration_of_magnets(x, y, 4, 0.7, False)
    for i in list_of_initial_velocity:
        all_x = []
        all_y = []
        for j in range(repetition):
            v = i
            phi = 2*np.pi*np.random.random()
            x0 = 0.0
            y0 = 0.5
            v_x0 = v * np.cos(phi)
            v_y0 = v * np.sin(phi)
            initial_conditions = [x0, y0, v_x0, v_y0]
            solution = odeint(pendulum, initial_conditions, t, args=(magnets,))

                #adding the fft of solutions
            all_x.append( np.abs(rfft(solution[:, 0])) )
            all_y.append( np.abs(rfft(solution[:, 1])) )
            print(np.shape(all_x))
            average = average_of_solution(all_x, all_y)
            print(np.shape(average))
            different_initial_v[i] = average
    return different_initial_v, freq



inicial_velocity = [0, 0.25, 0.5, 1, 1.5, 1.75]
resitev, frekvenca = solution_different_initial_v(inicial_velocity, 5, t)

plt.plot(frekvenca, resitev[1][:, 0] )
plt.show()

for i in inicial_velocity:
    plt.plot(frekvenca, resitev[i][:, 0], label="v = {}".format(i))
    # plt.plot(frekvenca, resitev[i][:, 1])

plt.xlabel(r"$\nu$", size=10)
plt.ylabel("FFT(x)", size=10)
plt.title("Fourier transformation of the magnetic pendulum solution (averaged over 5 simulations)", size=15)
plt.grid()
plt.legend(loc="best")
plt.show()





    
############################  FFT ############################
# x_fft = fft.fft(x)
# nu_x = fft.fftfreq(len(x), 5/len(x))      

# plt.plot(nu_x, np.abs(x_fft)**2)
# plt.show()
