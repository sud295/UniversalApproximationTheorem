import numpy as np
import pandas as pd

def poly(x):
    return 9*x**4 + 123*x**2 - 4*x + 902

x = np.linspace(-10, 10, 1000)
y = poly(x)

data = pd.DataFrame({'x': x, 'y': y})
data.to_csv('big_poly.csv', index=False)
