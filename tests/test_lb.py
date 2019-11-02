import numpy as np

x = np.array((58.3, 62.1))
b = 35.177

def method_a(b, x, j=1):
    return x[j] * ((x[0]/x[j])**b)

def method_b(b, x, j=1):
    # return x[1] - (b*(x[1]-x[0]))
    return b*x[0] + (1-b)*x[j]

print(method_a(b, x), method_b(b, x))

s = np.load('single_test.npy')
s.sort()
import toleranceinterval as ti
k = 2.49660

print(method_a(k, s, j=-1), method_b(k, s, j=-1))
