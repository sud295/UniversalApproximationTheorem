import numpy as np
import pandas as pd

def poly(x):
    return 3*x**2 + 12*x + 4

x = np.linspace(-10, 10, 1000)
y = poly(x)

data = pd.DataFrame({'x': x, 'y': y})
data.to_csv('small_poly.csv', index=False)
