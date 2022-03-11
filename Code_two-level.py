## Hello! And welcome to the EMME Python Solver v1. As soon as you run this,
## it'll begin to compute the evolution of two baths with two energy levels,
## with the initial temperatures, degeneracies and time-steps you set.
## After it's done computing everything, just execute one of the plot_
## functions and you'll see the time evolution of probabilities, energies,
## temperatures, observational entropy, canonical entropy and/or
## mutual information.

import numpy as np
import matplotlib.pyplot as plt
π = 2*np.arcsin(1)

Nt = 2**19 # time-steps.
Tt = 2**7  # total actual time.

δ = 5
γ = 15
dt = Tt/Nt
print(dt)

E0 = 0
E1 = δ

V0 = 300
V1 = 400

ρ = np.zeros((2,2,Nt))


# Auxiliary function ζ(t) computed through trapezoid numerical integration.

def F(τ):
    return(np.sinc(δ*τ/(2*π))**2)

def ζ(initial, final):
    dτ = dt*(2**(-6))
    j = 1
    if initial == 0:
        I = (F(initial*dt) + F(final*dt))*dτ
    else:
        I = F(final*dt)*dτ
    while j*dτ < (final-initial)*dt:
        I += F(initial*dt + j*dτ) * dτ
        j += 1
    return(I*δ/π)


# ODEs that will be solved using RK4.

def A(G, w, x):
    return(γ * G * (x/V1 - w/V0))

def B(G, w, x):
    return(γ * G * (w/V0 - x/V1))


# Thermodynamics

def pi(b, syst):
    if syst == 0:
        M0 = 1
        M1 = 1
    else:
        M0 = V0
        M1 = V1
    
    Z = M0*np.exp(-b*E0) + M1*np.exp(-b*E1)
    return([M0*np.exp(-b*E0)/Z, M1*np.exp(-b*E1)/Z])

def temp(E, syst):
    if E == E0:
        return(0)
    elif syst == 0:
        return((E1-E0)/(np.log(((E1-E)/(E-E0)))))
    else:
        M0 = V0
        M1 = V1
        return((E1-E0)/(np.log((M1/M0)*((E1-E)/(E-E0)))))


# Runge-Kutta solver

K = np.zeros((4,2))

def solve(Ts, Tb):
    ρ[0][0][0] = pi(1/Tb, 1)[0] * pi(1/Ts, 0)[0]
    ρ[0][1][0] = pi(1/Tb, 1)[0] * pi(1/Ts, 0)[1]
    ρ[1][0][0] = pi(1/Tb, 1)[1] * pi(1/Ts, 0)[0]
    ρ[1][1][0] = pi(1/Tb, 1)[1] * pi(1/Ts, 0)[1]
    
    GE = 1 # 0
    
    for t in range(Nt-1):
        w = ρ[0][1][t]
        x = ρ[1][0][t]
        
        GB = 1 # GE
        GM = 1 # GE + ζ(t, t + 1/2)
        GE = 1 # GM + ζ(t + 1/2, t + 1)
        
        K[0][0] = A(GB, w, x)
        K[0][1] = B(GB, w, x)
        
        for k in range(1, 3):
            K[k][0] = A(GM, w + (dt/2)*K[k-1][0], x + (dt/2)*K[k-1][1])
            K[k][1] = B(GM, w + (dt/2)*K[k-1][0], x + (dt/2)*K[k-1][1])
        
        K[3][0] = A(GE, w + dt * K[2][0], x + dt * K[2][1])
        K[3][1] = B(GE, w + dt * K[2][0], x + dt * K[2][1])
        
        ρ[0][0][t+1] = ρ[0][0][t]
        ρ[0][1][t+1] = w + (dt/6)*(K[0][0] + 2*(K[1][0]+K[2][0]) + K[3][0])
        ρ[1][0][t+1] = x + (dt/6)*(K[0][1] + 2*(K[1][1]+K[2][1]) + K[3][1])
        ρ[1][1][t+1] = ρ[1][1][t]
        
        if t%(2**16) == 0: print(GB) 

Time = np.zeros(Nt)

for t in range(Nt):
    Time[t] = t*dt


# Setting up and solving.

TS = 2**(-4)
TB = 100

solve(TS, TB)

print('simulation finished')


# Entropies.

QS = np.zeros((Nt))
QB = np.zeros((Nt))

DS = np.zeros((Nt))
DB = np.zeros((Nt))

bS = np.zeros((Nt))
bB = np.zeros((Nt))

for t in range(Nt):
    QS[t] = E1*(ρ[0][1][t]+ρ[1][1][t])+E0*(ρ[0][0][t]+ρ[1][0][t])
    QB[t] = E1*(ρ[1][0][t]+ρ[1][1][t])+E0*(ρ[0][0][t]+ρ[0][1][t])
    
    DS[t] = temp(QS[t], 0)
    DB[t] = temp(QB[t], 1)
    
    bS[t] = 1/DS[t]
    bB[t] = 1/DB[t]

print('energies computed')


SS = np.zeros((2,Nt))  # 0 observacional, 1 Gibbs.
SB = np.zeros((2,Nt))
S = np.zeros((2,Nt))
I = np.zeros((Nt))

for t in range(Nt):
    SS[0][t] = (ρ[0][0][t]+ρ[1][0][t])*np.log(1/(ρ[0][0][t]+ρ[1][0][t]))
    SS[0][t] += (ρ[0][1][t]+ρ[1][1][t])*np.log(1/(ρ[0][1][t]+ρ[1][1][t]))
    
    SB[0][t] = (ρ[0][0][t]+ρ[0][1][t])*np.log(V0/(ρ[0][0][t]+ρ[0][1][t]))
    SB[0][t] += (ρ[1][0][t]+ρ[1][1][t])*np.log(V1/(ρ[1][0][t]+ρ[1][1][t]))
    
    S[0][t] = SS[0][t] + SB[0][t]
    
    SS[1][t] = np.log(np.exp(-E0/DS[t]) + np.exp(-E1/DS[t]))
    SS[1][t] += (1/DS[t])*QS[t]
    
    SB[1][t] = np.log(V0*np.exp(-E0/DB[t]) + V1*np.exp(-E1/DB[t]))
    SB[1][t] += (1/DB[t])*QB[t]
    
    S[1][t] = SS[1][t] + SB[1][t]
    
    I[t]  = ρ[0][0][t]*np.log(ρ[0][0][t]/((ρ[0][0][t]+ρ[0][1][t])*(ρ[0][0][t]+ρ[1][0][t])))
    I[t] += ρ[0][1][t]*np.log(ρ[0][1][t]/((ρ[0][0][t]+ρ[0][1][t])*(ρ[0][1][t]+ρ[1][1][t])))
    I[t] += ρ[1][0][t]*np.log(ρ[1][0][t]/((ρ[1][0][t]+ρ[1][1][t])*(ρ[0][0][t]+ρ[1][0][t])))
    I[t] += ρ[1][1][t]*np.log(ρ[1][1][t]/((ρ[1][0][t]+ρ[1][1][t])*(ρ[0][1][t]+ρ[1][1][t])))
    
St  = ρ[0][0][0]*np.log((V0)/ρ[0][0][0])
St += ρ[1][1][0]*np.log((V1)/ρ[1][1][0])
St += (ρ[0][1][0]+ρ[1][0][0])*np.log((V1+V0)/(ρ[0][1][0]+ρ[1][0][0]))

print('entropies computed')


# Plotting functions.

def plot_energy():
    plt.xlim(0, Nt*dt)
    plt.ylim(min([QS[0], QB[0]]), max([QS[0], QB[0]]))
    
    plt.plot(Time, QS, color = 'r')
    plt.plot(Time, QB, color = 'b')

    print(QS[-1], QB[-1], QS[0] + QB[0], QS[-1]+QB[-1])

def plot_temp():
    plt.xlim(0, Nt*dt)
    plt.ylim(min([TB, TS]), max([TB, TS]))
    
    # EE = r'$ EE$'
    plt.plot(Time, DS, color = 'r')

    # EF = b'$ EF$'
    plt.plot(Time, DB, color = 'b')

    print(TS, TB, (DS[-1]+DB[-1])/2, (DS[-1]-DB[-1])/2)

def plot_beta():
    plt.xlim(0, Nt*dt)
    plt.ylim(min([bB[0], bS[0]]), max([bB[0], bS[0]]))
    
    # EE = r'$ EE$'
    plt.plot(Time, bS, color = 'r')

    # EF = b'$ EF$'
    plt.plot(Time, bS, color = 'b')

def plot_entropy(opt, ent):
    if opt == 0:
        plt.xlim(0, Nt*dt)
        plt.ylim(min(SS[ent]), max(SS[ent]))
        
        plt.plot(Time, SS[ent], color = 'g')
    
    elif opt == 1:
        plt.xlim(0, Nt*dt)
        plt.ylim(min(SB[ent]), max(SB[ent]))
        
        plt.plot(Time, SB[ent], color = 'g')
    
    else:
        plt.xlim(0, Nt*dt)
        plt.ylim(min(S[ent]), max(S[ent]))
        
        plt.plot(Time, S[ent], color = 'g')
    
def plot_entropy_all():
    plt.xlim(0, Nt*dt)
    plt.ylim(min(S[0]), max(S[1]))
    
    plt.plot(Time, S[0], color = 'g')
    plt.plot(Time, S[1], color = 'b')

def plot_corr():
    plt.xlim(0, Nt*dt)
    plt.ylim(min(I), max(I))
    
    plt.plot(Time, I, color = 'g')
    plt.plot(Time, I, color = 'b')