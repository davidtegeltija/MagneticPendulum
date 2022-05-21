from magnetno_nihalo import *

dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)
# z = 1.5 - L #height of the magnet above the plain of the other magnets

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# Initial conditions 
v = 1.5
phi1 = 2*np.pi*np.random.random()

x0 = 0.0
y0 = 0
v_x0 = v * np.cos(phi1)
v_y0 = v * np.sin(phi1)
initial_conditions1 = [x0, y0, v_x0, v_y0]

magnets = extra_magnets([[0.2, 0.7]], x, y, 3, 0.7) 
m1 = magnetic_moment * np.array([0, 0, 1]) #magnetic moment of stationary magnets
# m2 = magnetic_moment * np.array([np.sin(theta) * x/r, np.sin(theta) * y/r, np.cos(theta)])
m2 = magnetic_moment * np.array([0, 0, 1])

def calculate_the_energy(solution, magnets):
    x = solution[:, 0]
    y = solution[:, 1]
    v_x = solution[:, 2]
    v_y = solution[:, 3]
    # z = h + L - np.sqrt(L**2 - x**2 - y**2)
    z = -h - L + np.sqrt(L**2 - x**2 - y**2)
    combined_B = np.zeros(3)
    #for j in range(len())
    #for i in range(len(magnets)):

    #    r_vec = np.array([ x - magnets[i][0], y - magnets[i][1], h + L - np.sqrt(L**2 - x**2 - y**2)])
    #    magnetic_field = B(r_vec, m2)
    #    combined_B[0] += magnetic_field[0]
    #    combined_B[1] += magnetic_field[1]
    #    combined_B[2] += magnetic_field[2]

    Wk = 1/2 * M * (v_x**2 + v_y**2)
    Wp = M * g * z
    Wm = 1 - Wk - Wp
    #Wm = - np.dot(m2, combined_B)
    return Wk, Wp, Wm

resitev = odeint(pendulum, initial_conditions1, t, args=(magnets,))
energije = calculate_the_energy(resitev, magnets)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

ax1.plot(t, energije[0], label="$W_k$")
ax1.plot(t, energije[1], label="$W_p$")
ax1.plot(t, energije[2], label="$W_m$")
ax1.set_xlabel("Time")
ax1.set_ylabel("Individual energy")
ax1.set_title("Exchange of energy over time")
ax1.legend()


x_koordinata1 = resitev[:, 0]
y_koordinata1 = resitev[:, 1]
x_magnets = magnets[:, 0]
y_magnets = magnets[:, 1]

# fig = plt.subplot()
# ax2 = fig.add_subplot(xlim = (-2, 2), ylim =(-2, 2))
# axis.grid()   ne rabim več, ker uporabljam plt.style.use("bmh")
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
line, = ax2.plot([], [], "o-", linewidth=2, color="b") 
time_text = ax2.text(0.05, 0.9, "", transform=ax2.transAxes)
ax2.plot(x_magnets, y_magnets, "o", color="black", markersize=5)
# plt.plot(solution1[:, 0], solution1[:, 1], color="b") #plotanje trajektorije
# plt.plot(solution2[:, 0], solution2[:, 1], color="r")

def init():
    line.set_data([x0], [y0])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x1 = x_koordinata1[i]
    y1 = y_koordinata1[i]
    line.set_data(x1, y1)
    time_text.set_text("time =%.1fs" %(i * 0.002))    #"time=%.1f" % čas je enako kot da napisemo "time={}s".format(round(i*dt, 1))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t)), interval=2, blit=False)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Simulation responsible for the left graph")
ax2.legend(loc="best")
ax2.grid()
# writervideo = animation.PillowWriter(fps=26) 
# ani.save("C:/Users/David/Documents/IPT/animacija3.gif", writer=writervideo)
plt.show()

