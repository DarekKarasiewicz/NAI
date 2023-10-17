import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



# New Antecedent/Consequent objects hold universe variables and membership
# functions
wind_speed = ctrl.Antecedent(np.arange(0, 60, 1), 'wind_speed ')
wind_speed_names = ["week","avarge","strong","very_strong","armagedon"]
board_length = ctrl.Antecedent(np.arange(132, 166, 1), 'board_length') #długość w cm
board_length_names = ["kid_size","short","avarge","long","big_foot"]
board_width = ctrl.Consequent(np.arange(39, 50, 1), 'board_weight')  #szerokość w cm
board_width_names = ["kid_size","narrow","avarge","wide","big_foot"]
# można dać też wagę
kite_size = ctrl.Consequent(np.arange(3, 18, 1), 'kite_size')

# Auto-membership function population is possible with .automf(3, 5, or 7)
wind_speed.automf(names=wind_speed_names)
board_length.automf(names=board_length_names)
board_width.automf(names=board_length_names)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
kite_size['very_small'] = fuzz.trimf(kite_size.universe, [3, 6, 9])
kite_size['small'] = fuzz.trimf(kite_size.universe, [6, 9, 12])
kite_size['medium'] = fuzz.trimf(kite_size.universe, [9, 12, 15])
kite_size['large'] = fuzz.trimf(kite_size.universe, [12, 15, 18])
kite_size['very_large'] = fuzz.trimf(kite_size.universe, [15, 18, 18])






rule1 = ctrl.Rule(wind_speed['week'] | board_length['kid_size'] | board_width['kid_size'], kite_size['very_large'])
rule2 = ctrl.Rule(wind_speed['week'] | board_length['kid_size'] | board_width['narrow'], kite_size['very_large'])
rule3 = ctrl.Rule(wind_speed['week'] | board_length['short'] | board_width['narrow'], kite_size['very_large'])
rule4 = ctrl.Rule(wind_speed['week'] | board_length['short'] | board_width['avarge'], kite_size['very_large'])
rule5 = ctrl.Rule(wind_speed['week'] | board_length['avarge'] | board_width['avarge'], kite_size['very_large'])
rule6 = ctrl.Rule(wind_speed['week'] | board_length['avarge'] | board_width['wide'], kite_size['very_large'])
rule7 = ctrl.Rule(wind_speed['week'] | board_length['long'] | board_width['wide'], kite_size['very_large'])
rule8 = ctrl.Rule(wind_speed['week'] | board_length['long'] | board_width['big_foot'], kite_size['large'])
rule9 = ctrl.Rule(wind_speed['week'] | board_length['big_foot'] | board_width['big_foot'], kite_size['large'])
rule10 = ctrl.Rule(wind_speed['avarge'] | board_length['kid_size'] | board_width['kid_size'], kite_size[""])
rule11 = ctrl.Rule(wind_speed['avarge'] | board_length['kid_size'] | board_width['narrow'], kite_size[""])
rule12 = ctrl.Rule(wind_speed['avarge'] | board_length['short'] | board_width['kid_size'], kite_size[""])
rule13 = ctrl.Rule(wind_speed['avarge'] | board_length['short'] | board_width['narrow'], kite_size[""])
rule14 = ctrl.Rule(wind_speed['avarge'] | board_length['short'] | board_width['avarge'], kite_size[""])
rule15 = ctrl.Rule(wind_speed['avarge'] | board_length['avarge'] | board_width['narrow'], kite_size[""])
rule16 = ctrl.Rule(wind_speed['avarge'] | board_length['avarge'] | board_width['avarge'], kite_size[""])
rule17 = ctrl.Rule(wind_speed['avarge'] | board_length['avarge'] | board_width['wide'], kite_size[""])
rule18 = ctrl.Rule(wind_speed['avarge'] | board_length['long'] | board_width['wide'], kite_size[""])
rule19 = ctrl.Rule(wind_speed['avarge'] | board_length['big_foot'] | board_width['big_foot'], kite_size[""])
rule20 = ctrl.Rule(wind_speed['strong'] | board_length['kid_size'] | board_width['kid_size'], kite_size['medium'])
rule21 = ctrl.Rule(wind_speed['strong'] | board_length['kid_size'] | board_width['narrow'], kite_size['medium'])
rule22 = ctrl.Rule(wind_speed['strong'] | board_length['short'] | board_width['narrow'], kite_size['medium'])
rule23 = ctrl.Rule(wind_speed['strong'] | board_length['short'] | board_width['avarge'], kite_size['medium'])
rule24 = ctrl.Rule(wind_speed['strong'] | board_length['avarge'] | board_width['avarge'], kite_size['medium'])
rule25 = ctrl.Rule(wind_speed['strong'] | board_length['avarge'] | board_width['wide'], kite_size['medium'])
rule26 = ctrl.Rule(wind_speed['strong'] | board_length['long'] | board_width['wide'], kite_size['small'])
rule27 = ctrl.Rule(wind_speed['strong'] | board_length['long'] | board_width['big_foot'], kite_size['small'])
rule28 = ctrl.Rule(wind_speed['strong'] | board_length['big_foot'] | board_width['big_foot'], kite_size['small'])
rule29 = ctrl.Rule(wind_speed['strong'] | board_length['big_foot'] | board_width['big_foot'], kite_size['small'])
rule30 = ctrl.Rule(wind_speed['very_strong'] | board_length['kid_size'] | board_width['kid_size'], kite_size[""])
rule31 = ctrl.Rule(wind_speed['very_strong'] | board_length['kid_size'] | board_width['narrow'], kite_size[""])
rule32 = ctrl.Rule(wind_speed['very_strong'] | board_length['short'] | board_width['kid_size'], kite_size[""])
rule33 = ctrl.Rule(wind_speed['very_strong'] | board_length['short'] | board_width['narrow'], kite_size[""])
rule34 = ctrl.Rule(wind_speed['very_strong'] | board_length['short'] | board_width['avarge'], kite_size[""])
rule35 = ctrl.Rule(wind_speed['very_strong'] | board_length['avarge'] | board_width['narrow'], kite_size[""])
rule36 = ctrl.Rule(wind_speed['very_strong'] | board_length['avarge'] | board_width['avarge'], kite_size[""])
rule37 = ctrl.Rule(wind_speed['very_strong'] | board_length['avarge'] | board_width['wide'], kite_size[""])
rule38 = ctrl.Rule(wind_speed['very_strong'] | board_length['long'] | board_width['wide'], kite_size[""])
rule39 = ctrl.Rule(wind_speed['very_strong'] | board_length['big_foot'] | board_width['big_foot'], kite_size[""])
rule40 = ctrl.Rule(wind_speed['armagedon'] | board_length['kid_size'] | board_width['kid_size'], kite_size['small'])
rule41 = ctrl.Rule(wind_speed['armagedon'] | board_length['kid_size'] | board_width['narrow'], kite_size['small'])
rule42 = ctrl.Rule(wind_speed['armagedon'] | board_length['short'] | board_width['narrow'], kite_size['small'])
rule43 = ctrl.Rule(wind_speed['armagedon'] | board_length['short'] | board_width['avarge'], kite_size['very_small'])
rule44 = ctrl.Rule(wind_speed['armagedon'] | board_length['avarge'] | board_width['avarge'], kite_size['very_small'])
rule45 = ctrl.Rule(wind_speed['armagedon'] | board_length['avarge'] | board_width['wide'], kite_size['very_small'])
rule46 = ctrl.Rule(wind_speed['armagedon'] | board_length['long'] | board_width['wide'], kite_size['very_small'])
rule47 = ctrl.Rule(wind_speed['armagedon'] | board_length['long'] | board_width['big_foot'], kite_size['very_small'])
rule48 = ctrl.Rule(wind_speed['armagedon'] | board_length['big_foot'] | board_width['big_foot'], kite_size['very_small'])
rule49 = ctrl.Rule(wind_speed['armagedon'] | board_length['big_foot'] | board_width['big_foot'], kite_size['very_small'])


"""
.. image:: PLOT2RST.current_figure

Control System Creation and Simulation
---------------------------------------

Now that we have our rules defined, we can simply create a control system
via:
"""

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``.  Think of this object representing our controller
applied to a specific set of cirucmstances.  For tipping, this might be tipping
Sharon at the local brew-pub.  We would create another
``ControlSystemSimulation`` when we're trying to apply our ``tipping_ctrl``
for Travis at the cafe because the inputs would be different.
"""

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the quality 6.5 out of 10
and the service 9.8 of 10.
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['quality'] = 2.5
tipping.input['service'] = 9.8

# Crunch the numbers
tipping.compute()

"""
Once computed, we can view the result as well as visualize it.
"""
print (tipping.output['tip'])
tip.view(sim=tipping)

"""
.. image:: PLOT2RST.current_figure

The resulting suggested tip is **20.24%**.

Final thoughts
--------------

The power of fuzzy systems is allowing complicated, intuitive behavior based
on a sparse system of rules with minimal overhead. Note our membership
function universes were coarse, only defined at the integers, but
``fuzz.interp_membership`` allowed the effective resolution to increase on
demand. This system can respond to arbitrarily small changes in inputs,
and the processing burden is minimal.
"""
plt.show()