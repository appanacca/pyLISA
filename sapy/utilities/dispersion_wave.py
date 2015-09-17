import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

#a, b, c, d= sm.symbols('a b c d')
#M=sm.Matrix(([a,b],[c,d]))
#M.det()

U1, U2, y1, y2, m, alpha, omega, Fr, ym, slope= sm.symbols('U1, U2, y1, y2, m, alpha, omega, Fr, ym, slope')

# equation in y_1:
# (1)=0  so the terms in B and C change sign due to the left transport

M14=(U1*alpha -omega)*(sm.exp(alpha*y1) +sm.exp(-alpha*y1)) #G

M12=-((U1*alpha -omega)*(sm.exp(alpha*y1)) -m*sm.exp(alpha*y1)) #B

M13=((U1*alpha -omega)*(sm.exp(-alpha*y1)) +m*sm.exp(-alpha*y1)) #C

M11=0  #A

M15=0  #F 

#(2)=0  

M24=(sm.exp(alpha*y1)-sm.exp(-alpha*y1)) #G

M22=-sm.exp(alpha*y1) #B

M23=-sm.exp(-alpha*y1) #C

M21=0 #A

M25=0 #F

# equation in y_2:
# (1)=0

M31=(U2*alpha -omega)*sm.exp(alpha*y2)  #A

M32=-((U2*alpha -omega)*(sm.exp(alpha*y2)) -m*(sm.exp(alpha*y2)))  #B

M33= ((U2*alpha -omega)*(sm.exp(-alpha*y2)) +m*(sm.exp(-alpha*y2)))  #C

M34=0 #G

M35=-(U2*alpha -omega)*sm.exp(-alpha*y2) #F 

#(2)


M41=(sm.exp(alpha*y2)) #A

M42=-(sm.exp(alpha*y2)) #B

M43=-(sm.exp(-alpha*y2)) #C

M45=(sm.exp(-alpha*y2))  #F

M44=0  #G


# equations in y_max:

M51=(sm.exp(alpha*ym))*((U2*alpha -omega)**2 -slope/Fr**2 )  #A

M52=0   #B

M53= 0 #C

M54= 0 #G

M55=-(sm.exp(-alpha*ym))*((U2*alpha -omega)**2 +slope/Fr**2 )   #F

#------- MATRIX SYSTEM BUILDING----------

M=sm.Matrix(([M11, M12, M13, M14, M15],[M21, M22, M23, M24, M25],[M31, M32, M33, M34, M35],[M41, M42, M43, M44, M45], [M51, M52, M53, M54, M55]))

eq=M.det()

sm.pprint(sm.simplify(eq))


sol=sm.solve(eq, omega)

sm.pprint(sm.simplify(sol))

#------- NUMERICAL SOLVER--------------

U1n=0.4
U2n=1.6
y1n=0.7
y2n= 1.8
ymn=3.5
Frn=0.02
slopen=1
mn=(U2-U1)/(y2-y1)

alp_num=np.linspace(0,1.4,200)
om_num_1=np.zeros(len(alp_num),'D')
om_num_2=np.zeros(len(alp_num),'D')

for i in np.arange(len(alp_num)):
	om_num_1[i]=sol[0].evalf(subs={alpha:alp_num[i],U1:U1n,U2:U2n,y1:y1n,y2:y2n,m:mn,ym:ymn,Fr:Frn,slope:slopen})
	om_num_2[i]=sol[1].evalf(subs={alpha:alp_num[i],U1:U1n,U2:U2n,y1:y1n,y2:y2n,m:mn,ym:ymn,Fr:Frn,slope:slopen})
	
	#print om_num[i]


fig, ay = plt.subplots(dpi=150)
ay.plot(alp_num,np.imag(om_num_1),'b',alp_num,np.imag(om_num_2),'r',lw=2)
ay.set_ylabel(r'$\omega_i$',fontsize=32)
ay.set_xlabel(r'$\alpha$',fontsize=32)
#ay.set_ylim([-1,0.1])
#ay.set_xlim([0, 1.8])
#plt.tight_layout()
#fig.savefig('ci_cr.png', bbox_inches='tight',dpi=50)     
#plt.hold(True)
ay.grid()
#fig.savefig('ci_cr.png', bbox_inches='tight',dpi=150)
plt.show()
