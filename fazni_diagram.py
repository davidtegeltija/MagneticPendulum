from magnetno_nihalo import *

dx = 0.1
dy = 0.1
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)

dt = dx
t = np.arange(0.0, 10*len(x)*dt, dt)

# Initial conditions 

magnets = extra_magnets([[0.2, 0.7]], x, y, 3, 0.7) 
m1 = magnetic_moment * np.array([0, 0, 1]) #magnetic moment of stationary magnets
# m2 = magnetic_moment * np.array([np.sin(theta) * x/r, np.sin(theta) * y/r, np.cos(theta)])
m2 = magnetic_moment * np.array([0, 0, 1])



velocity = np.arange(0, 3, 0.5)
for i in velocity:
    v = i
    phi = np.pi/4
    x0 = 0.0
    y0 = 0.0
    v_x0 = v * np.cos(phi)
    v_y0 = v * np.sin(phi)
    # v_x0 = 0
    # v_y0 = 0
    initial_conditions = [x0, y0, v_x0, v_y0]
    y = odeint(pendulum, initial_conditions, t, args=(magnets,))

    plt.plot([y[0,0]], [y[0,1]], "o", color="red") # start
    plt.plot([y[-1,0]], [y[-1,1]], "o", color="purple") # end
    # plt.plot(y[:,0], y[:,1],'-', color="orange")
    # plt.plot(y[:,0] + 2*np.pi, y[:,1], color="orange") 
    plt.plot([y[0,0]], [y[0,1]], "o", color="red") # start
    plt.plot([y[-1,0]], [y[-1,1]], "o", color="purple") # end
    
plt.plot(magnets[:, 0], magnets[:, 1], "o", color="black")
plt.xlabel(r'$\theta(t)$')
plt.ylabel(r"$\theta' (t)$")
plt.title("Fazni portret matematičnega nihala", size=21)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.grid()
plt.legend(labels=["Začetna pozicija", "Končna pozicija"], fontsize=15)
plt.show()




velocity = np.arange(0, 3, 0.5)
for i in velocity:
    v = i
    phi = np.pi/4
    x0 = 0.0
    y0 = 0.0
    v_x0 = v * np.cos(phi)
    v_y0 = v * np.sin(phi)
    # v_x0 = 0
    # v_y0 = 0
    initial_conditions = [x0, y0, v_x0, v_y0]
    y = odeint(pendulum, initial_conditions, t, args=(magnets,))

    plt.plot([y[0,0]], [y[0,1]], "o", color="red") # start
    plt.plot([y[-1,0]], [y[-1,1]], "o", color="purple") # end
    # plt.plot(y[:,0], y[:,1],'-', color="orange")
    # plt.plot(y[:,0] + 2*np.pi, y[:,1], color="orange") 
    plt.plot([y[0,0]], [y[0,1]], "o", color="red") # start
    plt.plot([y[-1,0]], [y[-1,1]], "o", color="purple") # end
    
plt.plot(magnets[:, 0], magnets[:, 1], "o", color="black")
plt.xlabel(r'$\theta(t)$')
plt.ylabel(r"$\theta' (t)$")
plt.title("Fazni portret matematičnega nihala", size=21)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.grid()
plt.legend(labels=["Začetna pozicija", "Končna pozicija"], fontsize=15)
plt.show()