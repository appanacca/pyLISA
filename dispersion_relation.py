import sympy as sm
a, b, c, d= sm.symbols('a b c d')

M=sm.Matrix(([a,b],[c,d]))
M.det()

U1, U2, y1, y2, m, alpha, omega= cm.symbols('U1, U2, y1, y2, m, alpha, omega')

# equation in y_1:
# (1)=0  so the terms in B and C change sign due to the left transport

(U1*alpha -omega)*(sm.exp(alpha*y1) +sm.exp(-alpha*y1)) #G

-((U1*alpha -omega)*(sm.exp(alpha*y1)) -m*sm.exp(-alpha*y1)) #B

-((U1*alpha -omega)*(-sm.exp(-alpha*y1)) -m*sm.exp(-alpha*y1)) #C


#(2)=0  

(sm.exp(alpha*y1)-sm.exp(-alpha*y1)) #G

-sm.exp(alpha*y1) #B

-sm.exp(-alpha*y1) #C


# equation in y_2:
# (1)=0

-(U2*alpha -omega)  #A

-((U2*alpha -omega)*(sm.exp(alpha*y2)) #B

((U2*alpha -omega)*(sm.exp(-alpha*y2)) #C

#(2)

1  #A

-(sm.exp(alpha*y2)) #B

-(sm.exp(-alpha*y2)) #C
