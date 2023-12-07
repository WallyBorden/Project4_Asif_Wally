#Radiation_therapy

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parameters for the logistic growth and diffusion
r = 0.1  # Initial growth rate
K = 100  # Carrying capacity
D = 0.01  # Diffusion coefficient
radiation_effect = 0.02  # Decrease in growth rate per radiation session

# Spatial domain parameters
L = 10  # Length of the domain
N = 100  # Number of points in the domain
dx = L / N  # Spatial step size
x = np.linspace(0, L, N)

# Logistic growth function
def logistic_growth(P, r, K):
    return r * P * (1 - P / K)

# Reaction-diffusion model with radiation effect
def reaction_diffusion_radiation(t, P):
    global r  # global growth rate variable
    # Apply radiation at specific intervals (for example, every 10 time units)
    if int(t) % 10 == 0 and r > 0:
        r -= radiation_effect  # reduce growth rate due to radiation
    dPdt = np.zeros_like(P)
    # Diffusion term
    dPdt[1:-1] = D * (P[2:] - 2 * P[1:-1] + P[:-2]) / dx**2
    # Logistic growth term
    dPdt[1:-1] += logistic_growth(P[1:-1], r, K)
    return dPdt

# Initial condition (small tumor at the center)
P0 = np.zeros(N)
P0[N//2 - 5:N//2 + 5] = 10

# Time points
t_span = (0, 50)
t = np.linspace(t_span[0], t_span[1], 200)

# Solve the PDE with radiation effect
sol = solve_ivp(reaction_diffusion_radiation, t_span, P0, t_eval=t)

# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(x, sol.y[:, 0], color='blue')
ax.set_ylim(0, np.max(sol.y))

def animate(i):
    line.set_ydata(sol.y[:, i])  # Update the data
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50)

plt.xlabel('Position')
plt.ylabel('Tumor Density')
plt.title('Tumor Growth and Diffusion with Cumulative Radiation Effect')
plt.show()

