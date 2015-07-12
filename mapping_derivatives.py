import sympy as sm

y, L, s=sm.symbols('y L s')

mapping=sm.sqrt((y**2 *(1+s))/(L**2 + y**2))

d_map=sm.diff(mapping, y)

sm.pprint(sm.simplify(d_map))
print (sm.simplify(d_map))


dd_map=sm.diff(d_map, y)

sm.pprint(sm.simplify(dd_map))
print (sm.simplify(dd_map))


ddd_map=sm.diff(dd_map, y)

sm.pprint(sm.simplify(ddd_map))
print (sm.simplify(ddd_map))


dddd_map=sm.diff(ddd_map, y)
sm.pprint(sm.simplify(dddd_map))
print (sm.simplify(dddd_map))


