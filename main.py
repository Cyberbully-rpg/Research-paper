import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample housing dataset
data = {
    'house_size':        [800, 1200, 1500, 1800, 2200, 900, 1600, 2000, 1100, 2500],
    'distance_from_city':[15,  10,   8,    5,    2,    12,  6,    3,    11,   1],
    'num_bedrooms':      [2,   3,    3,    4,    4,    2,   3,    4,    3,    5],
    'age_of_house':      [20,  15,   10,   8,    5,    18,  7,    3,    12,   1],
    'house_price':       [3500000, 5500000, 7000000, 9000000, 12000000,
                          4000000, 8000000, 11000000, 5000000, 14000000]
}

df = pd.DataFrame(data)