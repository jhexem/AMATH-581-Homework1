from tkinter import Y
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
A6 = coefficientsheun[1]


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
A9 = coefficientsadams[1]

'''yfe = solfe[2]
yheun = solheun[2]
yadams = soladams[2]
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

