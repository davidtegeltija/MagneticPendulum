import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
from scipy.integrate import odeint
import matplotlib.animation as animation
import scipy.integrate as integrate
import scipy.special as special



############################ CONSTANTS and FUNCTIONS ############################
mu_0 = 4*np.pi * 1e-7
g = 9.81
L = 1. #visina magneta na kateri visi
R = 10 #radij magneta
omega = np.sqrt(g/L)

m1 = R*np.array([0, 0, 1]) #magnetic moment 
m2 = R*np.array([0, 0, 1])
# m2 = R*np.array([0, 0, 1])
# N = 100
# x = np.linspace(-50.0, 50.0, N)
# y = np.linspace(-50.0, 50.0, N)
# z = np.linspace(-10.0, 10.0, N)
dx = 0.01
dy = 0.01
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)
# N = 1000
# r = np.arange(0, 1, N)
# phi = np.linspace(0, 2*np.pi, N)
# R, PHI = np.meshgrid(r, phi) 
# X = R * np.cos(PHI)
# Y = R * np.sin(PHI)

# theta_x0 = 1.
# theta_y0 = 0.
x0 = 1.
y0 = 0.
v_x0 = 0.
v_y0 = 0.
initial_conditions = [x0, y0, v_x0, v_y0]

dt = dx
t = np.arange(0.0, len(x) * dt, dt)


#magnetic potencial
def A(r, m):
    return mu_0/4*np.pi * (np.cross(m, r) / r**3)

#magnetic field
def B(r, m):
    return mu_0/4*np.pi * (3. * r * np.dot(m, r)/np.linalg.norm(r)**5 - m/np.linalg.norm(r)**3)

#magnetic force
def F(r, m1, m2):
    return 3*mu_0 / (4*np.pi*np.linalg.norm(r)**5) * ((np.dot(m1, r)*m2 + np.dot(m2, r)*m1 + np.dot(m1, m2)*r - 5*np.dot(m1, r)*np.dot(m2, r) / np.linalg.norm(r)**2 * r ))


############################  POSITION OF MAGNETS  ############################
def configuration_of_magnets(x, y, N, distance=1, random_positions=False):
    if random_positions == False:
        theta = 2*np.pi / N
        phi = 0
        dx = np.abs(round(x[1] - x[0], 2))
        dy = np.abs(round(y[1] - y[0], 2))
        multiplication_factor = distance / dx
        positions = np.zeros([N, 2])
        # positions[0] = [0, 0]
        for i in range(1, N):
            x_projection = multiplication_factor * np.cos(phi)
            y_projection = multiplication_factor * np.sin(phi)
            x_direction = round(round(x_projection, 2) * dx, 2)
            y_direction = round(round(y_projection, 2) * dy, 2)

            positions[i] = positions[i - 1][0] + x_direction, positions[i - 1][1] + y_direction
            phi += theta

        centroid = positions.mean(axis=0)
        move_in_x = centroid[0]
        move_in_y = centroid[1]

        final_positions = np.zeros([N, 2])
        for i in range(N):
            final_positions[i] = positions[i][0] - move_in_x, positions[i][1] - move_in_y
        return final_positions

    else:
        positions = np.zeros([N, 2])
        for i in range(N):
            theta = 2 * np.pi * random.random()
            lenght = random.random()
            positions[i] = lenght*np.cos(theta), lenght*np.sin(theta)
        return positions



def magnetic_field(x, y, number_of_magnets):
    N = len(x)
    M = len(y)
    positions = configuration_of_magnets(x, y, number_of_magnets)
    mag_field = np.zeros([N, M])
    for magnet in positions:
        for i in range(N):
            for j in range(M):
                r = np.array([magnet[0] - x[i], magnet[1] - y[j], 0]) 
                mag_field[i, j] += np.linalg.norm( B(r, m1) ) 
    return mag_field, positions



# mag_field = magnetic_field(x, y, 6)[0]
# position_of_magnets = magnetic_field(x, y, 3)[1]




############################  FORCE and PENDULUM STATE  ############################
def force(x, y, magnets):
    # x = position[0]
    # y = position[1]
    # z = position[2]
    combined_force_of_magnets = np.zeros(2)
    pendulum_force = np.zeros(2)
    r = (x**2 + y**2)**0.5
    theta = np.arctan(r/L)
    alpha = np.arctan(y/x)
    for i in range(len(magnets)):
        r_vec = np.array([ np.abs(x - magnets[i][0]), np.abs(y - magnets[i][1]), L * np.cos(theta)])
        force = F(r_vec, m1, m2)
        combined_force_of_magnets[0] += force[0]
        combined_force_of_magnets[1] += force[1]
    

    pendulum_force[0] = omega**2 * np.sin(theta) * np.cos(alpha)
    pendulum_force[1] = omega**2 * np.sin(theta) * np.sin(alpha)
    return - pendulum_force - combined_force_of_magnets #I AM NOT SURE ABOUT SIGNS +/-


# def force_y(theta_y, x, y):
#     m1 = R*np.array([0, 0, 1])
#     m2 = R*np.array([0, 0, 1])
#     r = (x**2 + y**2)**0.5
#     theta = np.arcsin(r/L)
#     return -omega**2 * np.sin(theta) -F(r, m1, m2)[1]


# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state, t, magnets): 
    x, y, v_x, v_y = state
    derivitives = np.zeros_like(state)
    derivitives[0] = v_x                #dxdt[0] = derivites[0] = state[1] = v_x
    derivitives[1] = v_y                #dydt[0] = derivites[1] = state[2] = v_y
    derivitives[2] = force(x, y, magnets)[0]     #dxdt[1] = derivites[2] = F_x
    derivitives[3] = force(x, y, magnets)[1]     #dydt[1] = derivites[3] = F_y
    # dydt[0] = state[1][1] # x' = v 
    # dydt[1] = force(state[1][0])  # v' = F(x)
    return derivitives


magnets = configuration_of_magnets(x, y, 5, 1, False)
# print(magnets[1])
solution = odeint(pendulum, initial_conditions, t, args=(configuration_of_magnets(x, y, 5), ))

plt.plot(magnets[:, 0], magnets[:, 1], "o")
# plt.plot(magnets[1][0], magnets[1][1], "o")
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.grid()
plt.show()

plt.plot(t, solution[:, 0], label="x") #resitev za x
plt.plot(t, solution[:, 1], label="y") #resitev za y
plt.plot(t, solution[:, 2])
plt.legend()
plt.show()

for i in [0, 10, 20, 50, 100, 120, 150, 175, 200]:
    plt.plot(solution[i, 0], solution[i, 1], "o", label="{}s".format(i))
plt.plot(solution[175, 0], solution[175, 1], "o")
plt.plot(magnets[:, 0], magnets[:, 1], "o", markersize=10, color="black", label="magnets")
# plt.xlim(-border - 1, border + 1)
# plt.ylim(-border - 1, border + 1)
plt.grid()
plt.legend()
plt.show()






############################  2D plot of field strenght ############################
# def combine_two_arrays(array1, array2):
# 	new = []
# 	for i in range(len(array1)):
# 		new.append([array1[i], array2[i]])
# 	return np.array(new)

# def expand_solution(x, y, solution):
#     N = len(x)
#     M = len(y)


# X, Y = np.meshgrid(x, y)
# # X, Y = np.meshgrid(solution[:, 0], solution[:, 1])

# plt.contourf(X, Y, combine_two_arrays(solution[:, 0], solution[:, 1]), levels=len(X[0])//2, cmap="cividis") 
# # plt.contourf(X, Y, position_of_magnets, levels=len(X[0])//2, cmap="cividis") 

# cbar = plt.colorbar(orientation="vertical", aspect=10)
# # cbar.set_label(label="Hitrost tekočine v cevi", size = 15)
# plt.xlabel("x", fontsize=12)
# plt.ylabel("y", fontsize=12)
# # plt.title("Hitrostni profil toka tekočine skozi presek cevi", fontweight="bold", fontsize=15)
# # plt.xlim((-1.05, 1.05))
# # plt.ylim((0, 1.05))
# plt.show()



############################  3D plot of strenght of B  ############################
# ax = plt.axes(projection ="3d")
# ax.plot_surface(X, Y, mag_field, cmap ="cividis")  #viridis   #, edgecolor ="green"  To je barva mreže po, ki teče čez graf
# ax.set_xlabel("x", fontsize=12)
# ax.set_ylabel("y", fontsize=12)
# ax.set_zlabel("B (x, y)", fontsize=12)
# ax.set_title("3D representaion of B in xy plain", fontweight="bold", fontsize=15)
# plt.show()



############################  3D plot of vector field ############################
# X, Y, Z = np.meshgrid(x, y, z)

# u = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z)
# v = -np.cos(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.pi * Z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.sin(np.pi * Z))

# ax = plt.axes(projection='3d')
# ax.quiver(X, Y, Z, u, v, w, length=0.1)
# ax.set_xlabel("x", fontsize=12)
# ax.set_ylabel("y", fontsize=12)
# ax.set_zlabel("z", fontsize=12)
# plt.show()

