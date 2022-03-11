import numpy as np
import matplotlib.pyplot as plt
π = 2*np.arcsin(1)

Nt = 2**18
Tt = 2**6

δ = 0.5
γ12 = 20
γ34 = 0
γ35 = 5
γ45 = 20
γ67 = 20

dt = Tt/Nt
print(dt)

Ne = 3
N2 = Ne**2

H = [δ*x for x in range(Ne)]
V = 10*np.ones((2, Ne))
V[1][1] = 5

ρ = np.zeros((Ne,Ne))


def pi(b, syst):
    Z = 0
    R = []
    for i in range(Ne):
        Z += V[syst][i]*np.exp(-b*H[i])
    
    for i in range(Ne):
        R.append(V[syst][i]*np.exp(-b*H[i])/Z)
    
    return(R)

def temp(E, syst):
    return(0)


# ODEs that will be solved using RK4.

W = np.zeros((N2,N2))

W[1][2] = γ12/(V[0][1]*V[1][0])
W[2][1] = γ12/(V[0][0]*V[1][1])

W[3][4] = γ34/(V[0][1]*V[1][1])
W[3][5] = γ35/(V[0][2]*V[1][0])
W[4][3] = γ34/(V[0][0]*V[1][2])
W[4][5] = γ45/(V[0][2]*V[1][0])
W[5][3] = γ35/(V[0][0]*V[1][2])
W[5][4] = γ45/(V[0][1]*V[1][1])

W[6][7] = γ67/(V[0][2]*V[1][1])
W[7][6] = γ67/(V[0][1]*V[1][2])

for i in range(N2):
    for j in range(N2):
        if j != i:
            W[i][i] += -W[i][j]

print(W)

def ODE(im):
    out = np.zeros((N2))
    for i in range(N2):
        for j in range(N2):
            out[i] += W[i][j] * im[j]
        
    return(out)


# Runge-Kutta solver

K = np.zeros((N2,N2))
p = np.zeros((Nt,N2))
    
def solve(Ts, Tb):
    for i in range(Ne):
        for j in range(Ne):
            ρ[i][j] = pi(1/Tb, 1)[i] * pi(1/Ts, 0)[j]
    
    p[0][0] = ρ[0][0]
    p[0][1] = ρ[0][1]
    p[0][2] = ρ[1][0]
    p[0][3] = ρ[0][2]
    p[0][4] = ρ[1][1]
    p[0][5] = ρ[2][0]
    p[0][6] = ρ[1][2]
    p[0][7] = ρ[2][1]
    p[0][8] = ρ[2][2]
    
    for t in range(Nt-1):
        K[0] = ODE(p[t])
        K[1] = ODE(p[t] + (dt/2)*K[0])
        K[2] = ODE(p[t] + (dt/2)*K[1])
        K[3] = ODE(p[t] + dt*K[2])
        
        for i in range(N2):
            p[t+1][i] = p[t][i]+(dt/6)*(K[0][i]+2*K[1][i]+2*K[2][i]+K[3][i])
        
        if t%(2**16) == 0: print(1)

Time = np.zeros(Nt)

for t in range(Nt):
    Time[t] = t*dt


# Setting up and solving.

TS = 5
TB = 5

solve(TS, TB)

print('simulation finished')


# Energies and entropies.

QS = np.zeros((Nt))
QB = np.zeros((Nt))

# DS = np.zeros((Nt))
# DB = np.zeros((Nt))

for t in range(Nt):
    QS[t]  = H[1]*(p[t][1] + p[t][4] + p[t][7])
    QS[t] += H[2]*(p[t][3] + p[t][6] + p[t][8])
    QB[t]  = H[1]*(p[t][2] + p[t][4] + p[t][6])
    QB[t] += H[2]*(p[t][5] + p[t][7] + p[t][8])
    
    # DS[t] = temp(QS[t], 0)
    # DB[t] = temp(QB[t], 1)

print('energies computed')


# Plotting functions.

def plot_energy():
    plt.xlim(0, Nt*dt)
    plt.ylim(min([QS[0], QB[0]]), max([QS[0], QB[0]]))
    
    plt.plot(Time, QS, color = 'r')
    plt.plot(Time, QB, color = 'b')

    print(QS[-1], QB[-1], QS[0] + QB[0], QS[-1]+QB[-1])

def plot_populations(total_energy):
    if total_energy == 1:
        plt.xlim(0, Nt*dt)
        plt.ylim(min([p[0][1], p[0][2]]), max([p[0][1], p[0][2]]))
        
        plt.plot(Time, [row[1] for row in p], color = 'r')
        plt.plot(Time, [row[2] for row in p], color = 'b')
    
    elif total_energy == 2:
        plt.xlim(0, Nt*dt)
        plt.ylim(min([p[0][3], p[0][4], p[0][5]]), max([p[0][3], p[0][4], p[0][5]]))
        
        plt.plot(Time, [row[3] for row in p], color = 'r')
        plt.plot(Time, [row[4] for row in p], color = 'g')
        plt.plot(Time, [row[5] for row in p], color = 'b')
    
    elif total_energy == 3:
        plt.xlim(0, Nt*dt)
        plt.ylim(min([p[0][6], p[0][7]]), max([p[0][6], p[0][7]]))
        
        plt.plot(Time, [row[6] for row in p], color = 'r')
        plt.plot(Time, [row[7] for row in p], color = 'b')
        
    else: return(0)

"""
def plot_temp():
    plt.xlim(0, Nt*dt)
    plt.ylim(min([TB, TS]), max([TB, TS]))
    
    # EE = r'$ EE$'
    plt.plot(Time, DS, color = 'r')

    # EF = b'$ EF$'
    plt.plot(Time, DB, color = 'b')

    print(TS, TB, (DS[-1]+DB[-1])/2, (DS[-1]-DB[-1])/2)
"""