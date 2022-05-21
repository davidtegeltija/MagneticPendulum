from turtle import color
from magnetno_nihalo import *

dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# magnets = extra_magnets([[0, 0], [0.3, 0.4]], x, y, 5, 0.7) 
magnets = configuration_of_magnets(x, y, 6, 0.7, False) 


# Initial conditions 
v = 0
# phi1 = 2*np.pi*np.random.random()
phi1 = np.pi/3
x0 = 0.0
y0 = 0.5
v_x0 = v * np.cos(phi1)
v_y0 = v * np.sin(phi1)

possible_x = np.arange(-L, L, 0.25)
possible_y = np.arange(-L, L, 0.25)
for i in possible_x:
    for j in possible_y:
        initial_conditions = [i, j, v_x0, v_y0]
        solution = odeint(pendulum, initial_conditions, t, args=(magnets,))
        plt.plot(solution[:, 0], solution[:, 1], label="$x_0$ = {} \n $y_0$ = {}".format(i, j))
        plt.plot(solution[0, 0], solution[0, 1], "o", color="purple", label="Start position")
        plt.plot(solution[-1, 0], solution[-1, 1], "o", color="red", label="End position")
        plt.plot(magnets[:, 0], magnets[:, 1], "o", color="black", markersize=5)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel("x", size=10)
        plt.ylabel("y", size=10)
        plt.title(r"Simulation of motion for $\mathbf{v}=0$")
        plt.legend(loc="best")
        plt.grid()
        plt.show()
