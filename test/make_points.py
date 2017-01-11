import json
import numpy as np
from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators

result = {}
for o in operators:
    result[o] = list(np.hstack([np.array([0.0]), np.random.uniform(-1.0, 1.0, 999).round(3)]))

with open('random_points.json', 'w') as f:
    json.dump(result, f)

points = {}
for o in operators:
    # convergence of the loop expansion requires c < (4 * pi)^2
    # see section 7 in https://arxiv.org/pdf/1205.4231.pdf
    cutoff = (4 * np.pi) ** 2
    # be sure to include the NP=0 point
    lows = np.linspace(-1 * cutoff, 0.0, num=10, endpoint=False)
    highs = np.linspace(0.0, cutoff, num=10)
    points[o] = np.hstack([lows, highs])

np.save('linspace_20_points.npy', points)

points = {}
for o, low, high in [
        ('cuW', -0.076897, 0.075463),
        ('cuB', -0.111924, 0.11056667),
        ('cu', -2.02494876, 0.55177),
        ('cHu', -1.452384, 0.363096)]:
    # be sure to include the NP=0 point
    # lows = np.linspace(low, 0.0, num=10, endpoint=False)
    l = np.linspace(low, high, num=29)
    points[o] = np.hstack([l, np.array([0.0])])

np.save('linspace_30_points.npy', points)
