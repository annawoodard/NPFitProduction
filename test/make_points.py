import json
import numpy as np
from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators

result = {}

for o in operators:
    result[o] = list(np.hstack([np.array([0.0]), np.random.uniform(-0.5, 0.5, 999)]))

with open('points.json', 'w') as f:
    json.dump(result, f)
