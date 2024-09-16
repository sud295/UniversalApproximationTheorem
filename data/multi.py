import numpy as np
import pandas as pd

def f(x, y, z):
    return (x**2 + 4*z, np.exp(y) + x, y*x - z**2)

num_points = 1000
x_vals = np.random.uniform(-10, 10, num_points)
y_vals = np.random.uniform(-10, 10, num_points)
z_vals = np.random.uniform(-10, 10, num_points)

output_1 = []
output_2 = []
output_3 = []

for x, y, z in zip(x_vals, y_vals, z_vals):
    out1, out2, out3 = f(x, y, z)
    output_1.append(out1)
    output_2.append(out2)
    output_3.append(out3)

data = pd.DataFrame({
    'x': x_vals,
    'y': y_vals,
    'z': z_vals,
    'f1': output_1,
    'f2': output_2,
    'f3': output_3
})

data.to_csv('multi.csv', index=False)
