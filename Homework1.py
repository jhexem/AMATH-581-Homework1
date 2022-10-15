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
      yvals = np.empty(1)

      for i in range(len(dtvals)):     #iterates thru the list of dt values and calculates the next y value using FE
         if i == 0:
            y = y0 + dt[j] * dydt(t0, y0)
         else:
            y = y + dt[j] * dydt(dtvals[i], y)
         yvals = np.append(yvals, y)

      errorlist[j] = np.abs(ans - y)      #adds the error for this value of dt to the error list
      
   return y, errorlist, yvals

solfe = forward_euler(t0, y0, dt, dydt, ans)
A1 = np.array([[solfe[0]]])
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
      yvals = np.empty(1)

      for i in range(len(dtvals)):
         if i == 0:
            y = y0 + (dt[j] / 2) * (dydt(t0, y0) + dydt(t0 + dt[j], y0 + dt[j] * dydt(t0, y0)))
         else:
            y = y + (dt[j] / 2) * (dydt(dtvals[i], y) + dydt(dtvals[i] + dt[j], y + dt[j] * dydt(dtvals[i], y)))
         yvals = np.append(yvals, y)

      errorlist[j] = np.abs(ans - y)

   print(errorlist)
   return y, errorlist, yvals

solheun = heun(t0, y0, dt, dydt, ans)
A4 = np.array([solheun[0]])
A5 = np.array([solheun[1]])

plt.loglog(dt, solheun[1], '.r')
coefficientsheun = np.polyfit(np.log(dt), np.log(solheun[1]), 1)
A6 = coefficientsheun[1]

polynomialheun = np.poly1d(coefficientsheun)
ylineheun = polynomialheun(np.log(xaxis))
plt.plot(xaxis, np.exp(ylineheun), '-b')
plt.show()
A6 = coefficientsheun[1]


'''yfe = solfe[2]
yheun = solheun[2]
xaxis = np.linspace(0, 5, len(yheun))
plt.plot(xaxis, yfe, '-g')
plt.plot(xaxis, yheun, '-r')
plt.plot(xaxis, ytrue(xaxis), '-b')
plt.show()'''