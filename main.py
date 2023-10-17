import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



# New Antecedent/Consequent objects hold universe variables and membership
# functions
WindSpeed = ctrl.Antecedent(np.arange(0, 50, 1), 'WindSpeed ')
BoardLength = ctrl.Antecedent(np.arange(132, 166, 1), 'BoardLength') #długość w cm
BoardWeight = ctrl.Consequent(np.arange(39, 50, 1), 'BoardWeight')  #szerokość w cm
# można dać też wagę
KiteSize = ctrl.Consequent(np.arange(2, 15, 1), 'KiteSize')


# Auto-membership function population is possible with .automf(3, 5, or 7)
WindSpeed.automf(5)
BoardLength.automf(5)
BoardWeight.automf(5)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
KiteSize['very small'] = fuzz.trimf(KiteSize.universe, [3, 6, 9])
KiteSize['small'] = fuzz.trimf(KiteSize.universe, [6, 9, 12])
KiteSize['medium'] = fuzz.trimf(KiteSize.universe, [9, 12, 15])
KiteSize['large'] = fuzz.trimf(KiteSize.universe, [12, 15, 18])
KiteSize['very large'] = fuzz.trimf(KiteSize.universe, [15, 18, 18])
