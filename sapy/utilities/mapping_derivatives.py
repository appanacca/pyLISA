import sympy as sm

y, L, x=sm.symbols('y L x')

# mapping=sm.sqrt((y**2 *(1+s))/(L**2 + y**2))
mapping = (L*(x +1) / sm.sqrt(1 -((x+1)/2)**2))
#y, a, b = sm.symbols('y a b')

#mapping = a*((1+y)/(b-y))

d_map=sm.diff(mapping, x)

#sm.pprint(sm.simplify(d_map))
print (sm.simplify(d_map))
print '-------------------'
"""
dd_map=sm.diff(d_map, y)

#sm.pprint(sm.simplify(dd_map))
print (sm.simplify(dd_map))
print '-------------------'

ddd_map=sm.diff(dd_map, y)

#sm.pprint(sm.simplify(ddd_map))
print (sm.simplify(ddd_map))
print '-------------------'

dddd_map=sm.diff(ddd_map, y)
#sm.pprint(sm.simplify(dddd_map))
print (sm.simplify(dddd_map))

"""
