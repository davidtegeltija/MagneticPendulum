import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.integrate import odeint
import matplotlib.animation as animation
import scipy.integrate as integrate

############################  CONSTANTS  ############################
mu_0 = 4*np.pi * 1e-7
g = 9.81
L = 1. #visina magneta na kateri visi
h = 0.5  # višina med ravnino magnetov ter nihalom pod theta = 0
magnetic_moment = 2000
M = 1
omega = np.sqrt(g/L)


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

def extra_magnets(list_of_extra_magnets, x, y, number_of_magnets, distance):
    magnets = configuration_of_magnets(x, y, number_of_magnets, distance, False)
    for i in list_of_extra_magnets:
        magnets = np.append(magnets, i)
    magnets = magnets.reshape([number_of_magnets + len(list_of_extra_magnets), 2])
    return magnets


############################  FUNCTIONS  ############################
#magnetic potencial
def A(r, m):
    return mu_0/4*np.pi * (np.cross(m, r) / r**3)

#magnetic field
def B(r, m):
    return mu_0/4*np.pi * (3. * r * np.dot(m, r)/np.linalg.norm(r)**5 - m/np.linalg.norm(r)**3)

#magnetic force between two magnetic dipoles
def F(r, m1, m2):
    force = 3*mu_0 / (4*np.pi*np.linalg.norm(r)**5) * ((np.dot(m1, r)*m2 + np.dot(m2, r)*m1 + np.dot(m1, m2)*r - 5*np.dot(m1, r)*np.dot(m2, r) / np.linalg.norm(r)**2 * r ))
    return force



############################  FORCE and PENDULUM STATE  ############################
def force(x, y, magnets):
    # x = position[0]
    # y = position[1]
    # z = position[2]
    combined_force_of_magnets = np.zeros(3)
    pendulum_force = np.zeros(3)
    r = (x**2 + y**2)**0.5
    theta = np.arctan(r/L)
    # alpha = np.arctan(y/x)
    m1 = magnetic_moment * np.array([0, 0, 1]) #magnetic moment of stationary magnets
    # m2 = magnetic_moment * np.array([np.sin(theta) * x/r, np.sin(theta) * y/r, np.cos(theta)])
    m2 = magnetic_moment * np.array([0, 0, 1])
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
def pendulum(state, t, magnets): 
    x, y, v_x, v_y = state
    derivitives = np.zeros_like(state)
    derivitives[0] = v_x                #dxdt[0] = derivites[0] = state[1] = v_x
    derivitives[1] = v_y                #dydt[0] = derivites[1] = state[2] = v_y
    # derivitives[2] = v_z
    derivitives[2] = force(x, y, magnets)[0]     #dxdt[1] = derivites[2] = F_x
    derivitives[3] = force(x, y, magnets)[1]     #dydt[1] = derivites[3] = F_y
    # derivitives[5] = force(x, y, magnets)[2]
    return derivitives




#dont need this 
# m1 = magnetic_moment*np.array([0, 0, 1]) #magnetic moment of stationary magnets
# m2 = magnetic_moment*np.array([0, 0, 1]) #magnetic moment of stationary magnets
# for razdalja in [[0.01, 0, 0], [0.1, 0, 0], [0.5, 0, 0], [0.75, 0, 0], [1, 0, 0], [1.2, 0, 0], [1.5, 0, 0], [1.7, 0, 0], [2, 0, 0]]:
#     print(F(np.array(razdalja), m1, m2))
#     r = (razdalja[0]**2 + razdalja[1]**2)**0.5
#     theta = np.arctan(r/L)
#     pendulum_force = omega**2 * np.sin(theta) * razdalja[0]/r
#     plt.plot(razdalja[0], pendulum_force, "o", label="pendulum", color="b")
#     plt.plot(razdalja[0], F(np.array(razdalja), m1, m2)[0], "o", label="magnets", color="r")
# plt.legend(["pendulum", "magnets"])
# plt.show()




# for i in [0, 10, 20, 50, 100, 120, 150, 175, 200]:
#     plt.plot(solution[i, 0], solution[i, 1], "o", label="{}s".format(i))
# plt.plot(solution[175, 0], solution[175, 1], "o")
# plt.plot(magnets[:, 0], magnets[:, 1], "o", markersize=10, color="black", label="magnets")
# # plt.xlim(-border - 1, border + 1)
# # plt.ylim(-border - 1, border + 1)
# plt.grid()
# plt.legend()
# plt.show()



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



