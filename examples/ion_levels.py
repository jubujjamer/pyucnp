import pandas as pd
import matplotlib.pyplot as plt
from pyucnp import data
from pyucnp import math

er_ion = data.load_ion_states()

observed = pd.to_numeric(er_ion['observed'], errors='coerce')
calculated = pd.to_numeric(er_ion['calculated'], errors='coerce')
print(er_ion.state.unique())

print(math.mean_energy(er_ion, '2H2', '11/2'))
# plt.plot(observed, calculated, 'ok')
# plt.show()
