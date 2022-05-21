from magnetno_nihalo import configuration_of_magnets, F
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


mu_0 = 4*np.pi * 1e-7
g = 9.81
L = 1. #visina magneta na kateri visi
h = 0.5  # višina med ravnino magnetov ter nihalom pod theta = 0
M = 1
omega = np.sqrt(g/L)


def force(x, y, magnets, mag_moment):
    # x = position[0]
    # y = position[1]
    # z = position[2]
    combined_force_of_magnets = np.zeros(3)
    # pendulum_force = np.zeros(3)
    r = (x**2 + y**2)**0.5
    # theta = np.arctan(r/L)
    # alpha = np.arctan(y/x)
    m1 = mag_moment * np.array([0, 0, 1]) #magnetic moment of stationary magnets
    # m2 = magnetic_moment * np.array([np.sin(theta) * x/r, np.sin(theta) * y/r, np.cos(theta)])
    m2 = mag_moment * np.array([0, 0, 1])
    for i in range(len(magnets)):
        r_vec = np.array([ x - magnets[i][0], y - magnets[i][1], h + L - np.sqrt(L**2 - x**2 - y**2)])
        force = F(r_vec, m1, m2)
        combined_force_of_magnets[0] += force[0]
        combined_force_of_magnets[1] += force[1]
        combined_force_of_magnets[2] += force[2]

    # pendulum_force[0] = omega**2 * x/r * np.sin(theta) 
    # pendulum_force[1] = omega**2 * y/r * np.sin(theta) 
    # pendulum_force[2] = omega**2 * np.cos(theta)
    gravity_force = [0, 0, - g * M]
    force = gravity_force + combined_force_of_magnets
    vector = [x, y, -L - h + np.sqrt(L**2 - x**2 - y**2)]
    projection = [np.dot(force, vector) * vector[0], np.dot(force, vector) * vector[1], np.dot(force, vector) * vector[2]]
    force = force - projection
    return force[: -1] #- pendulum_force + combined_force_of_magnets #I AM NOT SURE ABOUT SIGNS +/-


# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state, t, magnets, mag_moment): 
    x, y, v_x, v_y = state
    derivitives = np.zeros_like(state)
    derivitives[0] = v_x                #dxdt[0] = derivites[0] = state[1] = v_x
    derivitives[1] = v_y                #dydt[0] = derivites[1] = state[2] = v_y
    # derivitives[2] = v_z
    derivitives[2] = force(x, y, magnets, mag_moment)[0]     #dxdt[1] = derivites[2] = F_x
    derivitives[3] = force(x, y, magnets, mag_moment)[1]     #dydt[1] = derivites[3] = F_y
    # derivitives[5] = force(x, y, magnets)[2]
    return derivitives

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


# Initial conditions 1
v = 1.5
# phi1 = 2*np.pi*np.random.random()
phi1 = np.pi/3
x0 = 0.0
y0 = 0.5
v_x0 = v * np.cos(phi1)
v_y0 = v * np.sin(phi1)
initial_conditions1 = [x0, y0, v_x0, v_y0]




moment = [500, 1000, 1500, 2000]
colors = ["blue", "red", "green", "purple"]
plt.plot
# for i in colors:
for j in zip(moment, colors):
    solution = odeint(pendulum, initial_conditions1, t, args=(magnets, j[0]))
    plt.plot(solution[0, 0], solution[0, 1], "o", color=j[1])
    plt.plot(solution[len(t)//10, 0], solution[len(t)//10, 1], "o", color=j[1])
    plt.plot(solution[len(t)//5, 0], solution[len(t)//5, 1], "o", color=j[1])
    plt.plot(solution[len(t)//2, 0], solution[len(t)//2, 1], "o", color=j[1])
    # plt.plot(solution[len(t)//1.5, 0], solution[len(t)//1.5, 1], color=i)
    plt.plot(solution[-1, 0], solution[-1, 1], "o", color=j[1])
plt.legend(labels=moment)
plt.grid()
plt.show()
    