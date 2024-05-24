###############
### GRAVIPY ###
###############

# by R. Bhargav (https://github.com/bhargavrajath)
# with additional edits from H. Dawe. Version H.0

# This code simulates the trajectories of 3 bodies under their mutual gravitational influences, in 3 - dimensions, for given initial conditions.
# Gravity has been modelled using Netwonian law of gravitation and Newtonian mechanics. Relativistic effects ignored.
# The numerical integration used is of the first order. Hence it is advised to keep an eye on the time step variable to keep integration errors under check.
# Outputs are position and velocity arrays the length of simulation duration, with nested structure for each body and the 3D components.
# The example in this code simulates a 'Sun - Earth - near Earth asteroid' system. Figure 1 plots the 3D trajectories;
# Figure 2 plots the separation between Earth and the asteroid against time.
# It is recommended to keep the code for Fig 1 intact as it serves verification. The code for data analysis can be edited as required.

##################################
# Imports #

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import json

##################################
# Resources and Constants #

# The Pi
pi = np.pi

# Gravitation constant in SI
G = 6.6726e-11

# Gravitational acceleration due to earth at MSL
g0 = 9.8066

# coordinate indices
x = 0
y = 1
z = 2

# Units for position (can be used for velocity, ex: AU/YR)
KM = 1e3  # Kilometres in m
AU = 1.496e11  # Astronomical Unit in m
LY = 9.461e+15  # Light Year in m
PC = 3.086e+16  # Parsec in m

# Units for time
S = 1  # seconds (s)
HR = 3600  # Hour in s
DY = 86162.4  # Sidereal day in s
YR = 31.47e6  # Sidereal year in s

# Units for mass
KG = 1  # Kilogram (kg)
TN = 1e3  # Ton in kg
MSOL = 1.988e30  # Solar mass in kg

# Units also stored as a dictionary for config
units = {
'km' : 1e3,  # Kilometres in m
'AU' : 1.496e11,  # Astronomical Unit in m
'LY' : 9.461e+15,  # Light Year in m
'PC' : 3.086e+16,  # Parsec in m

# Units for time
's' : 1,  # seconds (s)
'hr' : 3600,  # Hour in s
'dy' : 86162.4,  # Sidereal day in s
'yr' : 31.47e6,  # Sidereal year in s

# Units for mass
'kg' : 1,  # Kilogram (kg)
'tn' : 1e3,  # Ton in kg
'MSOL' : 1.988e30  # Solar mass in kg
}

################################################################################
# Input Variables - edit config file as needed #
################################################################################

with open('config.JSON') as config_file:
    data = json.load(config_file)
# T = data['Simulation length']*YR
# dt = data['Time step']*DY
# N = data['Number of bodies']
# M = np.array(data['Mass of bodies'])
# s0 = np.array(data['Initial positions'])
# v0 = np.array(data['Initial velocities'])

T = data['Simulation length'] * units[data['Simulation length unit']]
dt = data['Time step'] * units[data['Time step unit']]
N = data['Number of bodies']
M = np.array(data['Mass of bodies']) * units[data['Mass unit']]
s0 = np.array(data['Initial positions']) * units[data['Position unit']]
v0 = np.array(data['Initial velocities']) * units[data['Velocity position unit']] / units[data['Velocity time unit']]

print(s0)
print(v0)
################################################################################
# Initialisation #

# Time array
t = np.arange(0, T, dt)

# Velocity array
# Indexing order - [body, time, xyz]
# First create full length zero array (faster computation)
v = np.zeros((N, len(t), 3))

# Position array
# Indexing order - [body, time, xyz]
# First create full length zero array (faster computation)
s = np.zeros((N, len(t), 3))

# Set initial positions
for n in range(0, N):
    s[n, 0, :] = np.array([s0[n, x], s0[n, y], s0[n, z]])

# Set initial velocities
for n in range(0, N):
    v[n, 0, :] = np.array([v0[n, x], v0[n, y], v0[n, z]])


# Gravitational Acceleration Function #

# calc_acc(body, position, velocity, index)            # velocity will be unused here but defined for generality

def calc_acc(n, x, u, i):
    # Currently this function for 3 bodies is hardcoded and will not work for N > 3.

    # body n is accelerated by sum of forces from remaining two bodies

    # Separation vectors to the two other bodies
    r_1 = s[n, i, :] - s[n - 1, i, :]
    if n == 2:
        r_2 = s[n, i, :] - s[0, i, :]
    else:
        r_2 = s[n, i, :] - s[n + 1, i, :]

    # Magnitudes of the separation vectors
    r1 = np.linalg.norm(r_1)
    r2 = np.linalg.norm(r_2)

    # Acceleration due to gravity - vector form
    acc1 = -G * M[n - 1] * (r_1) / (r1 ** 3)
    if n == 2:
        acc2 = -G * M[0] * (r_2) / (r2 ** 3)
    else:
        acc2 = -G * M[n + 1] * (r_2) / (r2 ** 3)

    # Total acceleration
    acc = acc1 + acc2

    return acc


# Integrator method #

# integ(body, position, velocity, time step size, index)

def integ(n, x, u, dt, i):
    # 1st order Euler integrator
    v = u + calc_acc(n, x, u, i) * dt
    s = x + v * dt

    return s, v


# Simulation #

for i in range(1, len(t)):

    # Integration on all bodies' states
    for n in range(0, N):
        s[n, i, :], v[n, i, :] = integ(n, s[n, i - 1, :], v[n, i - 1, :], dt, i - 1)

# Trajectory visualisation plots #

plt.figure(1)
ax = plt.axes(projection='3d')

# making life slightly easier
u = AU

# Graphical output axis label name
ulabel = '(AU)'

for n in range(0, N):
    # ax.plot(s[n, 0, x] / u, s[n, 0, y] / u, s[n, 0, z] / u, 'x')  # Plot initial positions
    ax.plot(s[n, :, x] / u, s[n, :, y] / u, s[n, :, z] / u)  # Plot trajectories
    ax.plot(s[n, -1, x] / u, s[n, -1, y] / u, s[n, -1, z] / u, 'o')  # Plot final positions

ax.grid(True)
plt.title('Trajectories')
ax.set_xlabel('X ' + ulabel)
ax.set_ylabel('Y ' + ulabel)
ax.set_zlabel('Z ' + ulabel)
ax.legend(['Initial Position', 'Trajectory', 'Final Position'])
ax.set_aspect('equal')
plt.tight_layout()

########################################################################################################################
# Data Analysis - edit as needed #
########################################################################################################################

plt.figure(2)
yu = 1e6 * KM  # Million kms
xu = YR

# Generate data of interest for analysis - example separation between Earth and the asteroid,
# requires norm on the 2nd level array (xyz) of the output position array
data = np.linalg.norm(s[1, :, :] - s[2, :, :], axis=1)

plt.plot(t / xu, data / yu)
plt.title('Separation v/s Time')
plt.xlabel('Time (Years)')
plt.ylabel('Separation (Million km)')
plt.grid(True)
plt.tight_layout()

########################################################################################################################

plt.show()

########################################################################################################################
# Hannah Animation Additions xoxo <3
########################################################################################################################

from matplotlib import animation

# Data set
numDataPoints = len(t)

# animation loop
def animate_func(num):
    ax.clear()
    ax.plot3D(s[1, :num + 1, x] / u, s[1, :num + 1, y] / u, s[1, :num + 1, z] / u, c='blue')  # Earth trajectory
    ax.scatter(s[1, num, x] / u, s[1, num, y] / u, s[1, num, z] / u, c='blue', marker='o')  # Earth Current Position

    ax.plot3D(s[2, :num + 1, x] / u, s[2, :num + 1, y] / u, s[2, :num + 1, z] / u, c='red')  # Asteroid Trajectory
    ax.scatter(s[2, num, x] / u, s[2, num, y] / u, s[2, num, z] / u, c='red', marker='o')  # Asteroid Current Position

    # ax.plot3D(s[0, :num + 1, x] / u, s[0, :num + 1, y] / u, s[0, :num + 1, z] / u, c='yellow')  # Sun Trajectory
    ax.scatter(s[0, num, x] / u, s[0, num, y] / u, s[0, num, z] / u, c='yellow', marker='o')  # Sun Current Position

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-0.5, 0.5])

    ax.set_title('Trajectory \nTime = ' + str(np.round(t[num]/xu, decimals=2)) + ' Years')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani = animation.FuncAnimation(fig, animate_func, interval=0.1, frames=numDataPoints)
plt.show()
