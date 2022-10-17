import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dydt(t, y):      #defines dydt
   return (-3) * y * np.sin(t)

def ytrue(t):     #defines ytrue
   return (np.pi * np.exp(3 * (np.cos(t) - 1)) / np.sqrt(2))

t0 = 0      #initializes t0, y0, the array of all dt values, and the ytrue value at t=5
y0 = np.pi / 2
dt = np.array([2**(-2), 2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
ans = ytrue(5)

def forward_euler(t0, y0, dt, dydt, ans):    #returns a tuple of the y value for 2^(-8) and the array of errors
   errorlist = np.empty(len(dt))    #initialize the array of 7 error values corresponding to the 7 dt values

   for j in range(len(dt)):      #iterates through each dt value
      dtvals = np.arange(0, 5 + dt[j], dt[j])      #creates an array of t values each spaced apart from the previous value by dt
      yvals = np.array(y0)

      for i in range(len(dtvals)):     #iterates thru the list of dt values and calculates the next y value using FE
         if i == 0:
            y = y0 + dt[j] * dydt(t0, y0)
         else:
            y = y + dt[j] * dydt(dtvals[i], y)
         yvals = np.append(yvals, y)

      errorlist[j] = np.abs(ans - y)      #adds the error for this value of dt to the error list
      
   return yvals, errorlist

solfe = forward_euler(t0, y0, dt, dydt, ans)
A1 = np.transpose(np.array([solfe[0]]))
A2 = np.array([solfe[1]])

coefficientsfe = np.polyfit(np.log(dt), np.log(solfe[1]), 1)
A3 = coefficientsfe[0]

xaxis = np.linspace(0.001, 1, 1000)
'''plt.loglog(dt, solfe[1], '.r')
coefficientsfe = np.polyfit(np.log(dt), np.log(solfe[1]), 1)
polynomialfe = np.poly1d(coefficientsfe)
ylinefe = polynomialfe(np.log(xaxis))
plt.plot(xaxis, np.exp(yline), '-b')
plt.show()'''

def heun(t0, y0, dt, dydt, ans):
   errorlist = np.empty(len(dt))

   for j in range(len(dt)):
      dtvals = np.arange(0, 5 + dt[j], dt[j])
      yvals = np.array([y0])

      for i in range(len(dtvals)):
         if i == 0:
            y = y0 + (dt[j] / 2) * (dydt(t0, y0) + dydt(t0 + dt[j], y0 + dt[j] * dydt(t0, y0)))
         else:
            y = y + (dt[j] / 2) * (dydt(dtvals[i], y) + dydt(dtvals[i] + dt[j], y + dt[j] * dydt(dtvals[i], y)))
         yvals = np.append(yvals, y)

      errorlist[j] = np.abs(ans - y)

   return yvals, errorlist

solheun = heun(t0, y0, dt, dydt, ans)
A4 = np.transpose(np.array([solheun[0]]))
A5 = np.array([solheun[1]])

coefficientsheun = np.polyfit(np.log(dt), np.log(solheun[1]), 1)
A6 = coefficientsheun[0]


'''plt.loglog(dt, solheun[1], '.r')
polynomialheun = np.poly1d(coefficientsheun)
ylineheun = polynomialheun(np.log(xaxis))
plt.plot(xaxis, np.exp(ylineheun), '-b')
plt.show()
A6 = coefficientsheun[1]'''

def RK2(t0, y0, dtval, dydt):
   return y0 + dtval * dydt(t0 + (dtval/2), y0 + (dtval/2) * dydt(t0, y0))

def adams(t0, y0, dt, dydt, ans):
   errorlist = np.empty(len(dt))

   for j in range(len(dt)):
      dtvals = np.arange(0, 5 + dt[j], dt[j])
      yvals = np.array([y0])

      for i in range(len(dtvals)):
         if i == 0:
            yprev = y0
            y = RK2(t0, y0, dt[j], dydt)
         else:
            ypred = y + (dt[j]/2) * (3*dydt(dtvals[i], y) - dydt(dtvals[i-1], yprev))
            y = y + (dt[j]/2) * (dydt(dtvals[i] + dt[j], ypred) + dydt(dtvals[i], y))
         yvals = np.append(yvals, y)

      errorlist[j] = np.abs(y - ans)

   return yvals, errorlist

soladams = adams(t0, y0, dt, dydt, ans)
A7 = np.transpose(np.array([soladams[0]]))
A8 = np.array([soladams[1]])

coefficientsadams = np.polyfit(np.log(dt), np.log(soladams[1]), 1)
A9 = coefficientsadams[0]

'''yfe = solfe[0]
yheun = solheun[0]
yadams = soladams[0]
xaxis = np.linspace(0, 5, len(yheun))
plt.plot(xaxis, yfe, '-g')
plt.plot(xaxis, yheun, '-r')
plt.plot(xaxis, yadams, '-y')
plt.plot(xaxis, ytrue(xaxis), '-b')
plt.show()'''

t0 = 0
y0 = np.sqrt(3)
dydt0 = 1
dt = 0.5

z0 = np.array([np.sqrt(3), 1])
tlist = np.arange(0, 32 + dt, dt)

def dzdte1(t, z):
   w1 = z[0]
   w2 = z[1]
   return np.array([w2, (-0.1) * ((w1 ** 2) - 1) * w2 + w1])

ep1 = solve_ivp(dzdte1, [0, 32], z0, method='RK45', t_eval=tlist)
ep1col = np.transpose(np.array([ep1.y[0]]))

def dzdte2(t, z):
   w1 = z[0]
   w2 = z[1]
   return np.array([w2, (-1) * ((w1 ** 2) - 1) * w2 + w1])

ep2 = solve_ivp(dzdte2, [0, 32], z0, method='RK45', t_eval=tlist)
ep2col = np.transpose(np.array([ep2.y[0]]))

def dzdte3(t, z):
   w1 = z[0]
   w2 = z[1]
   return np.array([w2, (-20) * ((w1 ** 2) - 1) * w2 + w1])

ep3 = solve_ivp(dzdte3, [0, 32], z0, method='RK45', t_eval=tlist)
ep3col = np.transpose(np.array([ep3.y[0]]))

A10 = np.transpose(np.array([ep1col, ep2col, ep3col]))[0]

tolerances = np.array([10 ** (-4), 10 ** (-5), 10 ** (-6), 10 ** (-7), 10 ** (-8), 10 ** (-9), 10 ** (-10)])
z0 = np.array([2, np.pi ** 2])

for i in range(len(tolerances)):
   solRK45 = solve_ivp(dzdte2, [0, 32], z0, method='RK45', atol=tolerances[i], rtol=tolerances[i])
   solRK23 = solve_ivp(dzdte2, [0, 32], z0, method='RK23', atol=tolerances[i], rtol=tolerances[i])
   solBDF = solve_ivp(dzdte2, [0, 32], z0, method='BDF', atol=tolerances[i], rtol=tolerances[i])

   TRK45 = solRK45.t
   YRK45 = solRK45.y
   dtlistRK45 = np.diff(TRK45)
   dtavgRK45 = np.mean(dtlistRK45)

   TRK23 = solRK23.t
   YRK23 = solRK23.y
   dtlistRK23 = np.diff(TRK23)
   dtavgRK23 = np.mean(dtlistRK23)

   TBDF = solBDF.t
   YRKBDF = solBDF.y
   dtlistBDF = np.diff(TBDF)
   dtavgBDF = np.mean(dtlistBDF)

   if i == 0:
      avglistRK45 = np.array([dtavgRK45])
      avglistRK23 = np.array([dtavgRK23])
      avglistBDF = np.array([dtavgBDF])
   else:
      avglistRK45 = np.append(avglistRK45, dtavgRK45)
      avglistRK23 = np.append(avglistRK23, dtavgRK23)
      avglistBDF = np.append(avglistBDF, dtavgBDF)

coefficientRK45 = np.polyfit(np.log(avglistRK45), np.log(tolerances), 1)
A11 = coefficientRK45[0]

coefficientRK23 = np.polyfit(np.log(avglistRK23), np.log(tolerances), 1)
A12 = coefficientRK23[0]

coefficientBDF = np.polyfit(np.log(avglistBDF), np.log(tolerances), 1)
A13 = coefficientBDF[0]

y0 = np.array([0.1, 0, 0.1, 0])

def sys_rhs1(t, y):
   v1, w1, v2, w2 = y
   dydt = [- v1 ** 3 + (1 + 0.05) * (v1 ** 2) - 0.05 * v1 - w1 + 0.1 + 0 * v2, 0.1 * v1 - 0.1 * w1, 
   - v2 ** 3 + (1 + 0.25) * (v2 ** 2) - 0.25 * v2 - w2 + 0.1 + 0 * v1, 0.1 * v2 - 0.1 * w2]
   return  dydt

def sys_rhs2(t, y):
   v1, w1, v2, w2 = y
   dydt = [- v1 ** 3 + (1 + 0.05) * (v1 ** 2) - 0.05 * v1 - w1 + 0.1 + 0 * v2, 0.1 * v1 - 0.1 * w1, 
   - v2 ** 3 + (1 + 0.25) * (v2 ** 2) - 0.25 * v2 - w2 + 0.1 + 0.2 * v1, 0.1 * v2 - 0.1 * w2]
   return  dydt

def sys_rhs3(t, y):
   v1, w1, v2, w2 = y
   dydt = [- v1 ** 3 + (1 + 0.05) * (v1 ** 2) - 0.05 * v1 - w1 + 0.1 - 0.1 * v2, 0.1 * v1 - 0.1 * w1, 
   - v2 ** 3 + (1 + 0.25) * (v2 ** 2) - 0.25 * v2 - w2 + 0.1 + 0.2 * v1, 0.1 * v2 - 0.1 * w2]
   return  dydt

def sys_rhs4(t, y):
   v1, w1, v2, w2 = y
   dydt = [- v1 ** 3 + (1 + 0.05) * (v1 ** 2) - 0.05 * v1 - w1 + 0.1 - 0.3 * v2, 0.1 * v1 - 0.1 * w1, 
   - v2 ** 3 + (1 + 0.25) * (v2 ** 2) - 0.25 * v2 - w2 + 0.1 + 0.2 * v1, 0.1 * v2 - 0.1 * w2]
   return  dydt

def sys_rhs5(t, y):
   v1, w1, v2, w2 = y
   dydt = [- v1 ** 3 + (1 + 0.05) * (v1 ** 2) - 0.05 * v1 - w1 + 0.1 - 0.5 * v2, 0.1 * v1 - 0.1 * w1, 
   - v2 ** 3 + (1 + 0.25) * (v2 ** 2) - 0.25 * v2 - w2 + 0.1 + 0.2 * v1, 0.1 * v2 - 0.1 * w2]
   return  dydt

tlist = np.arange(0, 100 + 0.5, 0.5)

solFH1 = solve_ivp(sys_rhs1, [0, 100], y0, method='BDF', t_eval=tlist)
A14 = np.transpose(np.array(solFH1.y))

solFH2 = solve_ivp(sys_rhs2, [0, 100], y0, method='BDF', t_eval=tlist)
A15 = np.transpose(np.array(solFH2.y))

solFH3 = solve_ivp(sys_rhs3, [0, 100], y0, method='BDF', t_eval=tlist)
A16 = np.transpose(np.array(solFH3.y))

solFH4 = solve_ivp(sys_rhs4, [0, 100], y0, method='BDF', t_eval=tlist)
A17 = np.transpose(np.array(solFH4.y))

solFH5 = solve_ivp(sys_rhs5, [0, 100], y0, method='BDF', t_eval=tlist)
A18 = np.transpose(np.array(solFH5.y))