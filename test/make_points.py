import json
import numpy as np
from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators

result = {}
for o in operators:
    result[o] = list(np.hstack([np.array([0.0]), np.random.uniform(-1.0, 1.0, 999).round(3)]))

with open('random_points.json', 'w') as f:
    json.dump(result, f)

result = {}
for o in operators:
    # be sure to include the NP=0 point
    lows = np.linspace(-1.0, 0.0, num=30, endpoint=False).round(3)
    highs = np.linspace(0.0, 1.0, num=30).round(3)
    result[o] = list(np.hstack([lows, highs]))

with open('linspace_points.json', 'w') as f:
    json.dump(result, f)
