import numpy as np
import matplotlib.pyplot as plt

def dydt(t, y):
   return (-3) * y * np.sin(t)

def ytrue(t):
   return (np.pi * np.exp ** (3 * (np.cos(t) + 1))) / np.sqrt(2)

dt = np.array([2**(-2), 2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])

