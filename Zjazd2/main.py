import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


"""
The Kite size Problem
-------------------

Control system which models how you might choose what size of kite you should
take om session. When you choosing you should consider wind speed, and size
of your board height and width. Base on that fuzzy control system gives you 
size you should consider to take (+2/-2 base on your weight).

We would formulate this problem as:

* Antecedents (Inputs)
   - `Wind speed`
      * Wind speed in knots  on a scale of 0 to 60?(Yes wind speed could be higher than 60 but then you should 
      consider why you try kill yourself)
      * Fuzzy set  week, average, strong, very_strong, armagedon
   - `board height`
      * Board height in cm on scale 132 to 166?
      * Fuzzy set: kid_size, short, average, long, big_foot
   - `board wight`
      * Board wight in cm on scale 39 to 50?
      * Fuzzy set: kid_size, short, average, long, big_foot
      
* Consequents (Outputs)
   - `Kite size`
      * What kite size you should take on session on scale 3m2 to 18m2 
      * Fuzzy set: very_small, small, medium, large, very_large, extreme_large
* Rules
    We should take only few rules by example:
   - IF the *wind* is week  *and* the *board_high* is big_foot *and* the *board_wight* is  big_foot,
     THEN the kite_size will be very_large.
   - IF the *wind* is strong  *and* the *board_high* is kid_size *and* the *board_wight* is  kid_size,
     THEN the kite_size will be medium.
   - IF the *wind* is very_strong  *and* the *board_high* is average *and* the *board_wight* is  wide,
     THEN the kite_size will be very_small.
   - IF the *wind* is average  *and* the *board_high* is short *and* the *board_wight* is  narrow,
     THEN the kite_size will be large.
     
* Usage
   - If I tell this controller that I rate:
      * the wind_speed as 9 and
      * the board_high as 136,
      * the board_width as 45,
   - it would recommend kite size:
      * a 15m2
"""

wind_speed = ctrl.Antecedent(np.arange(0, 60, 1), 'wind_speed')
wind_speed_names = ["week","average","strong","very_strong","armagedon"]
board_length = ctrl.Antecedent(np.arange(132, 166, 1), 'board_length')
board_length_names = ["kid_size","short","average","long","big_foot"]
board_width = ctrl.Antecedent(np.arange(39, 50, 1), 'board_weight')
board_width_names = ["kid_size","narrow","average","wide","big_foot"]
kite_size = ctrl.Consequent(np.arange(3, 18, 1), 'kite_size')

wind_speed.automf(names=wind_speed_names)
board_length.automf(names=board_length_names)
board_width.automf(names=board_width_names)

def kite_size_trifunction(kite_size):
    """
    This function returns array with Triangular membership function.

    param:
    kite_size array:
        Consequent (output/control) variable for a fuzzy control system.

    return:
    kite_size array:
        Triangular membership function.
    """
    kite_size['very_small'] = fuzz.trimf(kite_size.universe, [3, 6, 9])
    kite_size['small'] = fuzz.trimf(kite_size.universe, [6, 9, 12])
    kite_size['medium'] = fuzz.trimf(kite_size.universe, [9, 12, 15])
    kite_size['large'] = fuzz.trimf(kite_size.universe, [12, 15, 18])
    kite_size['very_large'] = fuzz.trimf(kite_size.universe, [15, 16, 18])
    kite_size['extreme_large'] = fuzz.trimf(kite_size.universe, [16, 17, 18])

    return kite_size


def kite_size_trapfunction(kite_size):
    """
    This function returns array with Trapezoidal membership function.

    param:
    kite_size array:
        Consequent (output/control) variable for a fuzzy control system.

    return:
    kite_size array:
        Trapezoidal membership function.
    """

    kite_size['very_small'] = fuzz.trapmf(kite_size.universe, [3,4,8,9])
    kite_size['small'] = fuzz.trapmf(kite_size.universe, [6,7,11,12])
    kite_size['medium'] = fuzz.trapmf(kite_size.universe, [9,10,14,15])
    kite_size['large'] = fuzz.trapmf(kite_size.universe, [12,13, 17, 18])
    kite_size['very_large'] = fuzz.trapmf(kite_size.universe, [15, 16,17, 18])
    kite_size['extreme_large'] = fuzz.trapmf(kite_size.universe, [16,17,18, 18])

    return kite_size

def kite_size_sfunction(kite_size):
    """
    This function returns array with S-function membership function.

    param:
    kite_size array:
        Consequent (output/control) variable for a fuzzy control system.

    return:
    kite_size array:
        S-function membership function.
    """

    kite_size['very_small'] = fuzz.smf(kite_size.universe, 3, 9)
    kite_size['small'] = fuzz.smf(kite_size.universe, 6, 12)
    kite_size['medium'] = fuzz.smf(kite_size.universe, 9, 15)
    kite_size['large'] = fuzz.smf(kite_size.universe, 12, 18)
    kite_size['very_large'] = fuzz.smf(kite_size.universe, 15, 18)
    kite_size['extreme_large'] = fuzz.smf(kite_size.universe, 16, 18)

    return kite_size

def write_diagram(wind_speed,board_length, board_width,kite_size):
    """
    This function calculate and generate final results of given membership function.
    
    param 
    wind_speed array:
        Antecedent (input/sensor) variable for a fuzzy control system.
        
    param 
    board_length:
        Antecedent (input/sensor) variable for a fuzzy control system.
    
    param 
    board_width array:
        Antecedent (input/sensor) variable for a fuzzy control system.
    
    param 
    kite_size array:
        Given membership function
    
    """
    rule1 = ctrl.Rule(wind_speed['week'] & board_length['kid_size'] & board_width['kid_size'], kite_size['extreme_large'])
    rule2 = ctrl.Rule(wind_speed['week'] & board_length['kid_size'] & board_width['narrow'], kite_size['extreme_large'])
    rule3 = ctrl.Rule(wind_speed['week'] & board_length['short'] & board_width['narrow'], kite_size['extreme_large'])
    rule4 = ctrl.Rule(wind_speed['week'] & board_length['short'] & board_width['average'], kite_size['extreme_large'])
    rule5 = ctrl.Rule(wind_speed['week'] & board_length['average'] & board_width['average'], kite_size['extreme_large'])
    rule6 = ctrl.Rule(wind_speed['week'] & board_length['average'] & board_width['wide'], kite_size['extreme_large'])
    rule7 = ctrl.Rule(wind_speed['week'] & board_length['long'] & board_width['wide'], kite_size['extreme_large'])
    rule8 = ctrl.Rule(wind_speed['week'] & board_length['long'] & board_width['big_foot'], kite_size['very_large'])
    rule9 = ctrl.Rule(wind_speed['week'] & board_length['big_foot'] & board_width['big_foot'], kite_size['very_large'])
    rule10 = ctrl.Rule(wind_speed['average'] & board_length['kid_size'] & board_width['kid_size'], kite_size["very_large"])
    rule11 = ctrl.Rule(wind_speed['average'] & board_length['kid_size'] & board_width['narrow'], kite_size["large"])
    rule12 = ctrl.Rule(wind_speed['average'] & board_length['short'] & board_width['kid_size'], kite_size["large"])
    rule13 = ctrl.Rule(wind_speed['average'] & board_length['short'] & board_width['narrow'], kite_size["large"])
    rule14 = ctrl.Rule(wind_speed['average'] & board_length['short'] & board_width['average'], kite_size["large"])
    rule15 = ctrl.Rule(wind_speed['average'] & board_length['average'] & board_width['narrow'], kite_size["large"])
    rule16 = ctrl.Rule(wind_speed['average'] & board_length['average'] & board_width['average'], kite_size["medium"])
    rule17 = ctrl.Rule(wind_speed['average'] & board_length['average'] & board_width['wide'], kite_size["medium"])
    rule18 = ctrl.Rule(wind_speed['average'] & board_length['long'] & board_width['wide'], kite_size["medium"])
    rule19 = ctrl.Rule(wind_speed['average'] & board_length['big_foot'] & board_width['big_foot'], kite_size["medium"])
    rule20 = ctrl.Rule(wind_speed['strong'] & board_length['kid_size'] & board_width['kid_size'], kite_size['medium'])
    rule21 = ctrl.Rule(wind_speed['strong'] & board_length['kid_size'] & board_width['narrow'], kite_size['medium'])
    rule22 = ctrl.Rule(wind_speed['strong'] & board_length['short'] & board_width['narrow'], kite_size['medium'])
    rule23 = ctrl.Rule(wind_speed['strong'] & board_length['short'] & board_width['average'], kite_size['medium'])
    rule24 = ctrl.Rule(wind_speed['strong'] & board_length['average'] & board_width['average'], kite_size['medium'])
    rule25 = ctrl.Rule(wind_speed['strong'] & board_length['average'] & board_width['wide'], kite_size['medium'])
    rule26 = ctrl.Rule(wind_speed['strong'] & board_length['long'] & board_width['wide'], kite_size['small'])
    rule27 = ctrl.Rule(wind_speed['strong'] & board_length['long'] & board_width['big_foot'], kite_size['small'])
    rule28 = ctrl.Rule(wind_speed['strong'] & board_length['big_foot'] & board_width['big_foot'], kite_size['small'])
    rule29 = ctrl.Rule(wind_speed['very_strong'] & board_length['kid_size'] & board_width['kid_size'], kite_size["small"])
    rule30 = ctrl.Rule(wind_speed['very_strong'] & board_length['kid_size'] & board_width['narrow'], kite_size["small"])
    rule31 = ctrl.Rule(wind_speed['very_strong'] & board_length['short'] & board_width['kid_size'], kite_size["small"])
    rule32 = ctrl.Rule(wind_speed['very_strong'] & board_length['short'] & board_width['narrow'], kite_size["small"])
    rule33 = ctrl.Rule(wind_speed['very_strong'] & board_length['short'] & board_width['average'], kite_size["small"])
    rule34 = ctrl.Rule(wind_speed['very_strong'] & board_length['average'] & board_width['narrow'], kite_size["small"])
    rule35 = ctrl.Rule(wind_speed['very_strong'] & board_length['average'] & board_width['average'], kite_size["small"])
    rule36 = ctrl.Rule(wind_speed['very_strong'] & board_length['average'] & board_width['wide'], kite_size["very_small"])
    rule37 = ctrl.Rule(wind_speed['very_strong'] & board_length['long'] & board_width['wide'], kite_size["very_small"])
    rule38 = ctrl.Rule(wind_speed['very_strong'] & board_length['big_foot'] & board_width['big_foot'], kite_size["very_small"])
    rule39 = ctrl.Rule(wind_speed['armagedon'] & board_length['kid_size'] & board_width['kid_size'], kite_size['small'])
    rule40 = ctrl.Rule(wind_speed['armagedon'] & board_length['kid_size'] & board_width['narrow'], kite_size['small'])
    rule41 = ctrl.Rule(wind_speed['armagedon'] & board_length['short'] & board_width['narrow'], kite_size['small'])
    rule42 = ctrl.Rule(wind_speed['armagedon'] & board_length['short'] & board_width['average'], kite_size['very_small'])
    rule43 = ctrl.Rule(wind_speed['armagedon'] & board_length['average'] & board_width['average'], kite_size['very_small'])
    rule44 = ctrl.Rule(wind_speed['armagedon'] & board_length['average'] & board_width['wide'], kite_size['very_small'])
    rule45 = ctrl.Rule(wind_speed['armagedon'] & board_length['long'] & board_width['wide'], kite_size['very_small'])
    rule46 = ctrl.Rule(wind_speed['armagedon'] & board_length['long'] & board_width['big_foot'], kite_size['very_small'])
    rule47 = ctrl.Rule(wind_speed['armagedon'] & board_length['big_foot'] & board_width['big_foot'], kite_size['very_small'])

    sizing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                      rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
                                      rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
                                      rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
                                      rule41, rule42, rule43, rule44, rule45, rule46, rule47])

    sizing = ctrl.ControlSystemSimulation(sizing_ctrl)

    sizing.input['wind_speed'] = 9
    sizing.input['board_length'] = 136
    sizing.input['board_weight'] = 45

    sizing.compute()
    print(sizing.output['kite_size'])
    kite_size.view(sim=sizing)
    plt.show()

write_diagram(wind_speed=wind_speed,board_length=board_length,board_width=board_width,kite_size=kite_size_trifunction(kite_size))
write_diagram(wind_speed=wind_speed,board_length=board_length,board_width=board_width,kite_size=kite_size_trapfunction(kite_size))
write_diagram(wind_speed=wind_speed,board_length=board_length,board_width=board_width,kite_size=kite_size_sfunction(kite_size))

"""
Documentation made with help of https://scikit-fuzzy.readthedocs.io/en/latest/index.html
"""