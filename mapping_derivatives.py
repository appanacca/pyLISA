import sympy as sm

y, L=sm.symbols('y L')

mapping=-1 +((y/L)/(1+(y/L)**2 /4)**0.5)

d_map=sm.diff(mapping, y)

print d_map
