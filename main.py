import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



# New Antecedent/Consequent objects hold universe variables and membership
# functions
wind_speed = ctrl.Antecedent(np.arange(0, 60, 1), 'WindSpeed ')
board_length = ctrl.Antecedent(np.arange(132, 166, 1), 'BoardLength') #długość w cm
board_width = ctrl.Consequent(np.arange(39, 50, 1), 'BoardWeight')  #szerokość w cm
# można dać też wagę
kite_size = ctrl.Consequent(np.arange(2, 15, 1), 'KiteSize')


# Auto-membership function population is possible with .automf(3, 5, or 7)
wind_speed.automf(5)
board_length.automf(5)
board_width.automf(5)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
kite_size['very small'] = fuzz.trimf(kite_size.universe, [3, 6, 9])
kite_size['small'] = fuzz.trimf(kite_size.universe, [6, 9, 12])
kite_size['medium'] = fuzz.trimf(kite_size.universe, [9, 12, 15])
kite_size['large'] = fuzz.trimf(kite_size.universe, [12, 15, 18])
kite_size['very large'] = fuzz.trimf(kite_size.universe, [15, 18, 18])
