import numpy as np
import pandas as pd

x = np.linspace(-5, 5, 1000)
y = np.exp(x)

data = pd.DataFrame({'x': x, 'y': y})
data.to_csv('exp.csv', index=False)
