import numpy as np
import toleranceinterval as ti

x = np.random.random(300)
lb = ti.non_parametric(x, 0.01, 0.95)
ub = ti.non_parametric(x, 0.99, 0.95)