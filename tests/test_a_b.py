import numpy as np
import toleranceinterval as ti

x = np.load('single_test.npy')

print('A basis')
print('normal', ti.oneside.normal(x, 0.01, 0.95))
print('non_parametric', ti.oneside.non_parametric(x, 0.01, 0.95))
print('HansonKoopmans', ti.oneside.hanson_koopmans(x, 0.01, 0.95))

print('B basis')
print('normal', ti.oneside.normal(x, 0.1, 0.95))
print('non_parametric', ti.oneside.non_parametric(x, 0.1, 0.95))
print('HansonKoopmans', ti.oneside.hanson_koopmans(x, 0.1, 0.95))

print('A basis High')
print('normal', ti.oneside.normal(x, 0.99, 0.95))
print('non_parametric', ti.oneside.non_parametric(x, 0.99, 0.95))
print('HansonKoopmans', ti.oneside.hanson_koopmans(x, 0.99, 0.95))

print('B basis High')
print('normal', ti.oneside.normal(x, 0.9, 0.95))
print('non_parametric', ti.oneside.non_parametric(x, 0.9, 0.95))
print('HansonKoopmans', ti.oneside.hanson_koopmans(x, 0.9, 0.95))
