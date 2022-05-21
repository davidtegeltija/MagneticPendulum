from magnetno_nihalo import *


############################  ANIMATION FOR DIFFERENT INITIAL ANGLES OF v VECTORS  ############################
dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)
# z = 1.5 - L #height of the magnet above the plain of the other magnets

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# magnets = extra_magnets([[0, 0], [0.3, 0.4]], x, y, 5, 0.7) 
magnets = configuration_of_magnets(x, y, 5, 0.7, False) 


# Initial conditions 1
v = 1.2
# phi1 = 2*np.pi*np.random.random()
phi1 = np.pi/3
x0 = 0.0
y0 = 0.5
v_x0 = v * np.cos(phi1)
v_y0 = v * np.sin(phi1)
initial_conditions1 = [x0, y0, v_x0, v_y0]


# Initial conditions 2
v = 1.2
# phi2 = 2*np.pi*np.random.random()
phi2 = phi1 + np.pi/18
x0 = 0.0
y0 = 0.5
# z0 = L + h - np.sqrt(L**2 - x**2 - y**2)
v_x0 = v * np.cos(phi2)
v_y0 = v * np.sin(phi2)
# v_z0 = 0.
initial_conditions2 = [x0, y0, v_x0, v_y0]


solution1 = odeint(pendulum, initial_conditions1, t, args=(magnets,))
solution2 = odeint(pendulum, initial_conditions2, t, args=(magnets,))

x_koordinata1 = solution1[:, 0]
y_koordinata1 = solution1[:, 1]
x_koordinata2 = solution2[:, 0]
y_koordinata2 = solution2[:, 1]
x_magnets = magnets[:, 0]
y_magnets = magnets[:, 1]

fig = plt.figure()
axis = fig.add_subplot(xlim = (-2, 2), ylim =(-2, 2))
# axis.grid()   ne rabim več, ker uporabljam plt.style.use("bmh")
line1, = axis.plot([], [], "o-", linewidth=2, color="b", label=r"$\phi = 60\degree$") #.format(round(phi1, 1))) 
line2, = axis.plot([], [], "o-", linewidth=2, color="r", label=r"$\phi = 70\degree$") #.format(round(phi2, 1)))
time_text = axis.text(0.05, 0.9, "", transform=axis.transAxes)
plt.plot(x_magnets, y_magnets, "o", color="black", markersize=5)
print(solution1[:, 0])
# plt.plot(solution1[:, 0], solution1[:, 1], color="b") #plotanje trajektorije
# plt.plot(solution2[:, 0], solution2[:, 1], color="r")

def init():
    line1.set_data([x0], [y0])
    line2.set_data([x0], [y0])
    time_text.set_text('')
    return line1, line2, time_text

def animate(i):
    x1 = x_koordinata1[i]
    y1 = y_koordinata1[i]
    x2 = x_koordinata2[i]
    y2 = y_koordinata2[i]
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    time_text.set_text("time =%.1fs" %(i * 0.002))    #"time=%.1f" % čas je enako kot da napisemo "time={}s".format(round(i*dt, 1))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t)), interval=2, blit=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Animation for different starting angles of velocity vector")
plt.legend(loc="best")
plt.grid()
writervideo = animation.PillowWriter(fps=26) 
ani.save("C:/Users/David/Documents/IPT/razlicni_zacetni_koti.gif", writer=writervideo)
plt.show()







############################  ANIMATION FOR DIFFERENT INITIAL VELOCITY VECTORS ############################
dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# magnets = extra_magnets([[0, 0], [0.3, 0.4]], x, y, 5, 0.7) 
magnets = configuration_of_magnets(x, y, 5, 0.7, False) 


# Initial conditions 1
v1 = 1.0
phi = np.pi/6
x0 = 0.0
y0 = 0.0
v_x0 = v1 * np.cos(phi)
v_y0 = v1 * np.sin(phi)
initial_conditions1 = [x0, y0, v_x0, v_y0]


# Initial conditions 2
v2 = 1.5
# phi2 = 2*np.pi*np.random.random()
phi = np.pi/6
x0 = 0.0
y0 = 0.0
# z0 = L + h - np.sqrt(L**2 - x**2 - y**2)
v_x0 = v2 * np.cos(phi)
v_y0 = v2 * np.sin(phi)
# v_z0 = 0.
initial_conditions2 = [x0, y0, v_x0, v_y0]


solution1 = odeint(pendulum, initial_conditions1, t, args=(magnets,))
solution2 = odeint(pendulum, initial_conditions2, t, args=(magnets,))

x_koordinata1 = solution1[:, 0]
y_koordinata1 = solution1[:, 1]
x_koordinata2 = solution2[:, 0]
y_koordinata2 = solution2[:, 1]
x_magnets = magnets[:, 0]
y_magnets = magnets[:, 1]

fig = plt.figure()
axis = fig.add_subplot(xlim = (-2, 2), ylim =(-2, 2))
# axis.grid()   ne rabim več, ker uporabljam plt.style.use("bmh")
line1, = axis.plot([], [], "o-", linewidth=2, color="b", label=r"$\mathbf{v}$ = %.1f" %v1) 
line2, = axis.plot([], [], "o-", linewidth=2, color="r", label=r"$\mathbf{v}$ = %.1f" %v2)
time_text = axis.text(0.05, 0.9, "", transform=axis.transAxes)
plt.plot(x_magnets, y_magnets, "o", color="black", markersize=5)
# plt.plot(solution1[:, 0], solution1[:, 1], color="b") #plotanje trajektorije
# plt.plot(solution2[:, 0], solution2[:, 1], color="r")

def init():
    line1.set_data([x0], [y0])
    line2.set_data([x0], [y0])
    time_text.set_text('')
    return line1, line2, time_text

def animate(i):
    x1 = x_koordinata1[i]
    y1 = y_koordinata1[i]
    x2 = x_koordinata2[i]
    y2 = y_koordinata2[i]
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    time_text.set_text("time =%.1fs" %(i * 0.002))    #"time=%.1f" % čas je enako kot da napisemo "time={}s".format(round(i*dt, 1))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t)), interval=2, blit=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Animation for different initial velocity vectors")
plt.legend(loc="best")
plt.grid()
writervideo = animation.PillowWriter(fps=26) 
ani.save("C:/Users/David/Documents/IPT/razlicna_zacetna_hitrost.gif", writer=writervideo)
plt.show()

