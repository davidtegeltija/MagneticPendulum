import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.integrate import odeint
import matplotlib.animation as animation
import sympy as smp



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




############################  USING SYMPY  ############################
t = smp.symbols("t")
g = smp.symbols("g")
l = smp.symbols("L")
M = smp.symbols("M")
mu_0 = smp.symbols("mu_0")
magnetic_moment = smp.symbols("magnetic_moment")

theta, phi = smp.symbols("theta, phi", cls=smp.Function)
theta = theta(t)
phi = phi(t)

theta_d = smp.diff(theta, t)
phi_d = smp.diff(phi, t)
theta_dd = smp.diff(theta_d, t)
phi_dd = smp.diff(phi_d, t)

x = l * smp.sin(theta) * smp.cos(phi)
y = l * smp.sin(theta) * smp.sin(phi)
z = -l * smp.cos(theta)

# m_vector_dinamic = magnetic_moment * smp.Matrix([smp.sin(theta) * smp.cos(phi), smp.sin(theta) * smp.sin(phi), smp.cos(theta)]) #tuki sm mogoče pozabu upoštevat - pri mag momentih
m_vector_dinamic = magnetic_moment * smp.Matrix([0, 0, 1])
m_vector_static = magnetic_moment * smp.Matrix([0, 0, 1])

# for i in range(len(position_of_magnets)//2):
# 	x_magnet, y_magnet = position_of_magnets.row(i)
#     z_magnet = 0
x_magnet = smp.symbols("x_magnet")
y_magnet = smp.symbols("y_magnet")
z_magnet = 0
r_vector = smp.Matrix([x - x_magnet, y - y_magnet, z])
r = r_vector.norm() #((x - x_magnet)**2 + (y - y_magnet)**2 + z**2)**0.5
B_vector = mu_0/(4*smp.pi*r**5) * (3 * r_vector * m_vector_static.dot(r_vector) - r**2 * m_vector_static)


print(m_vector_dinamic.dot(B_vector))

#Kinetic energy
T = 1/2 * M * (smp.diff(x, t)**2 + smp.diff(y, t)**2 + smp.diff(z, t)**2)

#Potencial energy with magnetic potencial
V = M * g * z - m_vector_dinamic.dot(B_vector)

#Lagrangian
L = T - V

EL1 = smp.diff(L, theta) - smp.diff(smp.diff(L, theta_d), t).simplify()
EL2 = smp.diff(L, phi) - smp.diff(smp.diff(L, phi_d), t).simplify()
print("\n", EL1, "\n")
print(EL2)

equations = smp.solve([EL1, EL2], (theta_dd, phi_dd), simplify=False, rational=False)
print(equations[theta_dd])
dtheta_dt_fun = smp.lambdify(theta_d, theta_d)
dphi_dt_fun = smp.lambdify(phi_d, phi_d)
omega_theta_fun = smp.lambdify((t, g, l, M, mu_0, magnetic_moment, x_magnet, y_magnet, theta, phi, theta_d, phi_d) , equations[theta_dd])
omega_phi_fun = smp.lambdify((t, g, l, M, mu_0, magnetic_moment, x_magnet, y_magnet, theta, phi, theta_d, phi_d) , equations[phi_dd])


# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state, t, g, l, M, mu_0, magnetic_moment, magnets): 
    x_magnet = magnets[0]
    y_magnet = magnets[1]

    theta, phi, dtheta_dt, dphi_dt = state
    derivitives = np.zeros_like(state)
    derivitives[0] = dtheta_dt_fun(dtheta_dt)                #dxdt[0] = derivites[0] = state[1] = v_x
    derivitives[1] = dphi_dt_fun(dphi_dt)                #dydt[0] = derivites[1] = state[2] = v_y
    # derivitives[2] = v_z
    derivitives[2] = omega_theta_fun(t, g, l, M, mu_0, magnetic_moment, x_magnet, y_magnet, theta, phi, dtheta_dt, dphi_dt)     #dxdt[1] = derivites[2] = F_x
    derivitives[3] = omega_phi_fun(t, g, l, M, mu_0, magnetic_moment, x_magnet, y_magnet, theta, phi, dtheta_dt, dphi_dt)     #dydt[1] = derivites[3] = F_y
    # derivitives[5] = force(x, y, z, magnets)[2]
    return derivitives

    #t, g, mu_0, magnetic_moment, L, M, x_magnet, y_magnet, theta, phi, theta_d, phi_d

############################  SOLUTION  ############################
mu_0 = 4*np.pi * 1e-7
g = 9.81
l = 1. #visina magneta na kateri visi
M = 1.
R = 0.1 #radij magneta
omega = np.sqrt(g/l)

theta = np.pi/2
phi = 0
dtheta_dt = 0.
dphi_dt = 0.8

initial_conditions = [theta, phi, dtheta_dt, dphi_dt]

dt = 0.01
t = np.arange(0.0, 10, dt)

magnets=[0.2, 0.2]
# magnets = configuration_of_magnets(x, y, 6, 0.8, False)
# for i in [ [0.2, 0.2], [0.4, 0.1], [-0.1, 0.6] ]:
#     magnets = np.append(magnets, i)
# magnets = magnets.reshape([9, 2])

solution_of_EL = odeint(pendulum, initial_conditions, t, args=(g, l, M, mu_0, magnetic_moment, magnets))

print(solution_of_EL.shape)





############################  ANIMATION ############################


x_koordinata = np.sin(solution_of_EL[:, 0]) * np.cos(solution_of_EL[:, 1])
y_koordinata = np.sin(solution_of_EL[:, 0]) * np.sin(solution_of_EL[:, 1])
# x_magnets = magnets[:, 0]
# y_magnets = magnets[:, 1]

fig = plt.figure()
axis = fig.add_subplot(xlim = (-2, 2), ylim =(-2, 2))
# axis.grid()   ne rabim več, ker uporabljam plt.style.use("bmh")
line, = axis.plot([], [], "o-", linewidth=2) 
time_text = axis.text(0.05, 0.9, "", transform=axis.transAxes)
# plt.plot(x_magnets, y_magnets, "o", color="black", markersize=5)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x = x_koordinata[i]
    y = y_koordinata[i]
    line.set_data(x, y)
    time_text.set_text("time =%.1fs" %(i * 0.002))    #"time=%.1f" % čas je enako kot da napisemo "time={}s".format(round(i*dt, 1))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t)), interval=2, blit=False, init_func=init)
plt.grid()

# writervideo = animation.PillowWriter(fps=26) 
# ani.save("C:/Users/David/Documents/IPT/animacija3.gif", writer=writervideo)
plt.show()




































# ############################ CONSTANTS and FUNCTIONS ############################
# mu_0 = 4*np.pi * 1e-7
# g = 9.81
# L = 1. #visina magneta na kateri visi
# R = 0.1 #radij magneta
# omega = np.sqrt(g/L)


# # N = 100
# # x = np.linspace(-50.0, 50.0, N)
# # y = np.linspace(-50.0, 50.0, N)
# # z = np.linspace(-10.0, 10.0, N)
# dx = 0.01
# dy = 0.01
# border = L
# x = np.arange(-border, border + dx, dx)
# y = np.arange(-border, border + dy, dy)


# def B(r, m):
#     return mu_0/4*np.pi * (3. * r * np.dot(m, r)/np.linalg.norm(r)**5 - m/np.linalg.norm(r)**3)

# def energy_of_dipole_in_B(theta, phi, static_m):
#     x = np.sin(theta) * np.cos(phi)
#     y = np.sin(theta) * np.sin(phi)
#     z = 1 - np.cos(theta)
#     r_vec = np.array([x, y, z])
#     mag_field = B(r_vec, static_m)



# def second_order_theta(theta, phi, dthetadt, dphidt, magnets):
#     return np.sin(theta) * np.cos(theta) * dphidt **2 - omega**2 * np.sin(theta)

# def second_order_phi(theta, phi, dthetadt, dphidt, magnets):
#     return -2 * dthetadt * dphidt * np.cos(theta)/np.sin(theta)



# # a simple pendulum y''= F(y) , state = (y,v)
# def pendulum(state, t, magnets): 
#     theta, phi, dtheta_dt, dphi_dt = state
#     derivitives = np.zeros_like(state)
#     derivitives[0] = dtheta_dt                #dxdt[0] = derivites[0] = state[1] = v_x
#     derivitives[1] = dphi_dt                #dydt[0] = derivites[1] = state[2] = v_y
#     # derivitives[2] = v_z
#     derivitives[2] = second_order_theta(theta, phi, dtheta_dt, dphi_dt, magnets)     #dxdt[1] = derivites[2] = F_x
#     derivitives[3] = second_order_phi(theta, phi, dtheta_dt, dphi_dt, magnets)     #dydt[1] = derivites[3] = F_y
#     # derivitives[5] = force(x, y, z, magnets)[2]
#     return derivitives



# ############################ SOLUTION ############################
# theta = np.pi/3
# phi = np.pi/2
# dtheta_dt = 0.
# dphi_dt = 0.5

# initial_conditions = [theta, phi, dtheta_dt, dphi_dt]

# dt = 0.01
# t = np.arange(0.0, 5, dt)


# magnets = configuration_of_magnets(x, y, 6, 0.8, False)
# for i in [ [0.2, 0.2], [0.4, 0.1], [-0.1, 0.6] ]:
#     magnets = np.append(magnets, i)
# magnets = magnets.reshape([9, 2])

# solution = odeint(pendulum, initial_conditions, t, args=(magnets,))




# plt.plot(magnets[:, 0], magnets[:, 1], "o")
# plt.grid()
# plt.show()

# plt.plot(t, solution[:, 0], label="x") #resitev za x
# plt.plot(t, solution[:, 1], label="y") #resitev za y
# plt.plot(t, solution[:, 2])
# plt.legend()
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



