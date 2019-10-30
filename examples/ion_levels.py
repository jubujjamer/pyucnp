import pandas as pd
import matplotlib.pyplot as plt
from pyucnp import data
from pyucnp.ions import Erbium

er = Erbium()

print(er.energy_levels())
