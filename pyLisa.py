# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:37:38 2014

@author: appanacca


"""

from __future__  import division
import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import chebdif as cb
import scipy.linalg as lin
import scipy.interpolate as intp

import scipy.io

import blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd



class fluid(object):
	"""
	fluid: Perform a linear stability analysis after building the operator(ex.Orr-Sommerfeld)
	"""
	def __init__(self,option,**kwargs):
		self.option=option
		self.N=option['n_points']
		self.y=np.linspace(-1,1,self.option['n_points'])
		self.U=np.zeros(len(self.y))
		self.aCD=np.zeros(len(self.y))
		self.dU=np.zeros(len(self.y))
		self.ddU=np.zeros(len(self.y))
		self.alpha=option['perturbation']['alpha']
		self.Re=option['perturbation']['Re']

		self.Fr=option['Froude']
		self.slope=option['slope']


	def read_velocity_profile(self):
		""" read from a file the velocity profile store in a .txt file and set the variable_data members""" 
		in_txt=np.genfromtxt(self.option['flow'], delimiter=' ',skiprows=1) 
		self.y_data=in_txt[:,0]
		self.U_data=in_txt[:,1]
		self.dU_data=in_txt[:,2]
		self.ddU_data=in_txt[:,3]
		self.aCD_data=in_txt[:,4]
		self.daCD_data=in_txt[:,5]
		
		

		self.lc=option['lc'] #0.16739  #lc*=0.22*(h-z1) / h 

	def set_poiseuille(self):
		"""set the members velocity and its derivatives as couette flow"""
		Upoiseuille=(lambda y: 1-y**2)
		dUpoiseuille=(lambda y: -y*2)
		ddUpoiseuille=-np.ones(len(self.y))*2
		self.U=Upoiseuille(self.y)
		self.dU=dUpoiseuille(self.y)
		self.ddU=ddUpoiseuille

	def set_hyptan(self):
		"""set the members velocity and its derivatives as hyperbolic tangent flow"""		
		Uhyptan=(lambda y: 0.5*(1+np.tanh(y)))
		dUhyptan=(lambda y: 1/(2*np.cosh(y)**2))
		ddUhyptan=(lambda y: (1/np.cosh(y))*(-np.tanh(y)/np.cosh(y) )  )
		self.U=Uhyptan(self.y)
		self.dU=dUhyptan(self.y)
		self.ddU=ddUhyptan(self.y)
		self.aCD=np.zeros(self.N)



	def set_blasisus(self,y_gl):
		"""set the members velocity and its derivatives as boundary layer flow"""	
		self.U, self.dU, self.ddU = bl.blasius(y_gl) 
		self.CD=np.zeros(len(self.y))
		


	def choose_variables(self):
		""" read the 'variable' option in the option dictionary and select the operator to solve"""

		if self.option['variables']=='v_eta':
			self.v_eta_operator()
		elif self.option['variables']=='p_u_v':
			self.LNS_operator()


	def plot_velocity(self):
		"""plot the velocity profiles"""
		fig, ay = plt.subplots(figsize=(10,10), dpi=50)
		lines = ay.plot(self.U,self.y,'b',self.dU,self.y,'g',self.ddU,self.y,'r',self.aCD,self.y,'m',self.daCD,self.y,'c',lw=2)
		ay.set_ylabel(r'$y$',fontsize=32)
		lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$',r'$a^* \dot C_D$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
		#ay.set_ylim([0,5])
		#ax.set_xlim([np.min(time[2*T:3*T]),np.max(time[2*T:3*T])])
		ay.grid()                    
		#plt.tight_layout()
		#fig.savefig('RESULTS'+'couette.png', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=50)     
		plt.show(lines)		
		
	def diff_matrix(self):
		"build the differenziation matrix with chebichev discretization  [algoritmh from Reddy & W...]"
		self.y, self.D= cb.chebdif(self.N,4)  #in this line we re-instanciate the y in gauss lobatto points
		self.D=self.D + 0j  # summing 0j is needed in order to make the D matrices immaginary
		
  
     #def mapping(self,method):
         #più tardi implementa metodi diversi per fare il mapping
     
         
  
	def v_eta_operator(self):
	        """ this member build the stability operator in the variable v, so you have to eliminate pressure from the equation and get the u=f(v,alpha) from the continuity eq. """ 
		I=np.identity(self.N)
		i=(0+1j)
		delta=self.D[1] -self.alpha**2 *I
		Z=np.zeros((self.N,self.N))


		CD=np.matrix(np.diag(self.aCD))
		dCD=np.matrix(np.diag(self.daCD))
		U=np.matrix(np.diag(self.U))
		D1=np.matrix(self.D[0])
		D2=np.matrix(self.D[1])
		D4=np.matrix(self.D[3])
		
		dU=np.matrix(np.diag(self.dU))
		ddU=np.matrix(np.diag(self.ddU))

		
		if self.option['equation']=='Euler':
			self.A= np.dot(np.diag(self.U),delta) -np.diag(self.ddU) 
			self.B=delta
		elif self.option['equation']=='Euler_CD':
			self.A= np.dot(np.diag(self.U),delta) -np.diag(self.ddU) -(i/self.alpha)*(dCD*U*D1 + CD*dU*D1 + CD*U*D2)
			self.B=delta
		elif self.option['equation']=='Euler_CD_turb':
			print "not implemented yet"
		elif self.option['equation']=='LNS':
			self.A=(i/(self.alpha*self.Re))*(D4 -2*self.alpha**2 *D2 + self.alpha**4 *I) -ddU +U*delta
			self.B=delta
		elif self.option['equation']=='LNS_CD':
			self.A=(i/(self.alpha*self.Re))*(D4 -2*self.alpha**2 *D2 + self.alpha**4 *I) -ddU +U*delta -(i/self.alpha)*(dCD*U*D1 + CD*dU*D1 + CD*U*D2)			 
			self.B=delta
		elif self.option['equation']=='LNS_turb':
			print "not implemented yet"			
		elif self.option['equation']=='LNS_turb_CD':
			print "not implemented yet"
		elif self.option['equation']=='Euler_wave':        # in this case the B.C. is of 2nd order in omega so the matrix problem should be reorganized
								# see the article of Jerome Hoepffner for details in the trick to transform polynomial eigenvalue problem in a single one
			self.A= np.dot(np.diag(self.U),delta) -np.diag(self.ddU) 
			self.B=delta
			self.C=Z

			A1=np.concatenate((self.A,Z),axis=1)
			A2=np.concatenate((Z,I),axis=1)
			self.A=np.concatenate((A1,A2))

			B1=np.concatenate((self.B,self.C),axis=1)
			B2=np.concatenate((I,Z),axis=1)
			self.B=np.concatenate((B1,B2))


		if self.option['equation']=='Euler':
			self.BC2()
		elif self.option['equation']=='Euler_CD':
			self.BC2()			
		elif self.option['equation']=='Euler_CD_turb':
			print "not implemented yet"
		elif self.option['equation']=='LNS':
			self.BC1()
		elif self.option['equation']=='LNS_CD':
			self.BC1()
		elif self.option['equation']=='LNS_turb':
			print "not implemented yet"			
		elif self.option['equation']=='LNS_turb_CD':
			print "not implemented yet"
		elif self.option['equation']=='Euler_wave':
			self.BC_wave_v_eta()



	def BC1(self):
		"""impose the boundary condition as specified in the paper "Modal Stability Theory" ASME 2014 from Hanifi in his examples codes
		   in v(0), v(inf) , Dv(0) , Dv(inf) all =0
		"""

		eps=1e-4*(0+1j)
		
		#v(inf)=0
		self.A[0,:]=np.zeros(self.N)
		self.A[0,0]=1
		self.B[0,:]=self.A[0,:]*eps

		#v'(inf)=0
		self.A[1,:]=self.D[0][0,:]
		self.B[1,:]=self.A[1,:]*eps

		#v'(0)=0
		self.A[-2,:] =self.D[0][-1,:]
		self.B[-2,:]= self.A[-2,:]*eps

		#v(0)=0
		self.A[-1,:]=np.zeros(self.N)
		self.A[-1,-1]=1
		self.B[-1,:]=self.A[-1,:]*eps

	def BC_wave_v_eta(self):
		eps=1e-4*(0+1j)
		
		#v(y_max)
		#self.A[0,:]=self.D[0][0,:]
		self.A[0,0:self.N]=self.D[0][0,:]*self.U[0]**2 -(np.identity(self.N)[0,:])*np.cos(self.slope)/self.Fr**2
		
		
		self.B[0,:]= np.concatenate((2*self.U[0]*self.D[0][0,:],-self.D[0][0,:]),axis=1)
		#self.B[0,0]=+2*self.U[0]*self.D[0][0,0]
		#self.B[1,self.N]=-self.D[0][0,0]  #conditoon on C**2 
		

		#v(0)=0
		self.A[self.N -1,:]=np.zeros(2*self.N)
		self.A[self.N -1,self.N -1]=1
		self.B[self.N -1,:]=self.A[self.N -1,:]*eps




	def BC2(self):
		"""impose the boundary condition as specified in the paper "Modal Stability Theory" ASME 2014 from Hanifi in his examples codes
		   only in the v(0) and v(inf) =0
		"""
		
		eps=1e-4*(0+1j)
		
		#v(inf)=0
		self.A[0,:]=np.zeros(self.N)
		self.A[0,0]=1
		self.B[0,:]=self.A[0,:]*eps

		#v(0)=0
		self.A[-1,:]=np.zeros(self.N)
		self.A[-1,-1]=1
		self.B[-1,:]=self.A[-1,:]*eps


         
        @nb.jit 
	def solve_eig(self):
		 """ solve the eigenvalues problem with the LINPACK subrutines"""
		 self.eigv, self.eigf  = lin.eig(self.A,self.B) 

		 #remove the infinite and nan eigenvectors, and their eigenfunctions
		 selector=np.isfinite(self.eigv)
		 self.eigv=self.eigv[selector]
		 self.eigf=self.eigf[:,selector]

		 self.eigv_re=np.real(self.eigv)
		 self.eigv_im=np.imag(self.eigv)


	def plot_spectrum(self):
		if self.option['variables']=='v_eta':
			self.plot_spectrum_v_eta()
		elif self.option['variables']=='p_u_v':
			self.plot_LNS()
		
	
	def plot_spectrum_v_eta(self):
		""" plot the spectrum """
		for i in np.arange(10):
			fig, ay = plt.subplots(figsize=(10,10), dpi=50)
			lines = ay.plot(self.eigv_re,self.eigv_im,'b*',lw=10)
			ay.set_ylabel(r'$c_i$',fontsize=32)
			ay.set_xlabel(r'$c_r$',fontsize=32)
			#lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay.set_xlim([0.4,1.6])
			ay.set_ylim([-0.02, 0.28])
			ay.grid()                                         
			#plt.tight_layout()
			fig.savefig('RESULTS'+'spectrum_couette.png', bbox_inches='tight',dpi=50)     
			#plt.show(lines)	

		
			sel_eig=plt.ginput(2)
		
			omega_r_picked=(sel_eig[0][0] +sel_eig[1][0])/2
			omega_i_picked=(sel_eig[0][1] +sel_eig[1][1])/2
		
			omega_picked=omega_r_picked*(1+0j)+ omega_i_picked*(0+1j)
			n=np.argmin(np.abs(self.eigv -omega_picked))

			eigfun_picked=self.eigf[:,n]#*(-0.13 -0.99j)
		
			print omega_picked, lin.norm(eigfun_picked)

	  # needed in the case "Euler_wave" because only the half of the point are in fact v the other part of the vector is alpha*v
			v=eigfun_picked[0:self.N]  
			u=np.dot((v/self.alpha),  self.D[0]) *(0+1j)

			fig2, (ay2,ay3) = plt.subplots(1,2)#, dpi=50)
			lines2 = ay2.plot(np.real(u),self.y,'r',np.imag(u),self.y,'g',np.sqrt(u*np.conjugate(u)),self.y,'m',lw=2)
			ay2.set_ylabel(r'$y$',fontsize=32)
			ay2.set_xlabel(r"$u$",fontsize=32)	
			#lgd=ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay2.set_ylim([0,5])
			#ay2.set_xlim([-1, 1])
			ay2.grid()  


			lines3 = ay3.plot(np.real(v),self.y,'r',np.imag(v),self.y,'g',np.sqrt(v*np.conjugate(v)),self.y,'m',lw=2)
			ay3.set_ylabel(r'$y$',fontsize=32)
			ay3.set_xlabel(r"$v$",fontsize=32)	
			#lgd=ay3.legend((lines3),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay3.set_ylim([0,5])
			#ay3.set_xlim([-1, 1])
			ay3.grid()  
			
			plt.show(lines)
		
	
		
	def mapping(self):
		if self.option['mapping'][0]=='semi_infinite':
			ymax=self.option['Ymax']
			s=self.y[1:-1]
			r=(s +1)/2
			L=(ymax*np.sqrt(1-r[0]**2) )/(2*r[0])
			self.y=(L*(s+1))/(np.sqrt((1- ((s+1)**2)/4)))
			y_inf=2000#(L*(1.999))/(np.sqrt((1- ((1.999)**2)/4)))
			self.y=np.concatenate([np.array([y_inf]), self.y])
			self.y=np.concatenate([self.y, np.array([0])])
			K=np.sqrt(self.y**2 +4* L**2)
			
			xi=np.zeros((self.N,4))
			xi[:, 0] =                           8 * L**2 / K**3 
			xi[:, 1] =                    - 24 * self.y * L**2 / K**5
			xi[:, 2] =           96 * (self.y**2 - L**2) * L**2 / K**7
			xi[:, 3]= 480 * self.y * (3 * L**2 - self.y**2) * L**2 / K**9		

		elif self.option['mapping'][0]=='infinite':
			L=10
			s_inf=20
			s=(L/s_inf)**2
			self.y=(-L*self.y)/(np.sqrt(1+s-self.y**2))
					
			xi=np.zeros((self.N,4))
			xi[:,0]= L**2*np.sqrt(self.y**2*(s + 1)/(L**2 + self.y**2))/(self.y*(L**2 + self.y**2))		
			xi[:,1]= -3*L**2*np.sqrt(self.y**2*(s + 1)/(L**2 + self.y**2))/(L**4 + 2*L**2*self.y**2 + self.y**4)
			xi[:,2]= 3*L**2*np.sqrt(self.y**2*(s + 1)/(L**2 + self.y**2))*(-L**2 + 4*self.y**2)/(self.y*(L**6 + 3*L**4*self.y**2 + 3*L**2*self.y**4 + self.y**6))
			xi[:,3]= L**2*np.sqrt(self.y**2*(s + 1)/(L**2 + self.y**2))*(45*L**2 - 60*self.y**2)/(L**8 + 4*L**6*self.y**2 + 6*L**4*self.y**4 + 4*L**2*self.y**6 + self.y**8)
			
		
		elif self.option['mapping'][0]=='finite':
			a=self.option['mapping'][1][0]
			b=self.option['mapping'][1][1]
			self.y=(b-a)*0.5*self.y +(a+b)*0.5

			xi=np.zeros((self.N,4))
			xi[:,0]=(2*self.y -a -b)/(b -a)
			xi[:,1]=np.zeros(self.N)
			xi[:,2]=np.zeros(self.N)
			xi[:,3]=np.zeros(self.N)


		self.D[0] = np.dot(np.diag(xi[:,0]),self.D[0])
		self.D[1] = np.dot(np.diag(xi[:,0]**2) , self.D[1]) + np.dot(np.diag(xi[:,1]) , self.D[0])			
		self.D[2] = np.dot(np.diag(xi[:,0]**3) , self.D[2]) + 3*np.dot(np.dot(np.diag(xi[:,0]),np.diag(xi[:,1])) ,self.D[1])   + np.dot(np.diag(xi[:,2]),self.D[0])
		self.D[3] = np.dot(np.diag(xi[:,0]**4),self.D[3])  + 6*np.dot(np.dot(np.diag(xi[:,1]),np.diag(xi[:,0]**2)),self.D[2])      + 4*np.dot(np.dot(np.diag(xi[:,2]),np.diag(xi[:,0])),self.D[1]) + 3*np.dot(np.diag(xi[:,1]**2),self.D[1]) + np.dot(np.diag(xi[:,3]),self.D[0]) 
			
		#scipy.io.savemat('test.mat', dict(x=self.D,y=xi))



	def LNS_operator(self):
		#----Matrix Construction-----------
		#  p |u |v
		# (       ) continuity 
		# (       ) x-momentum
		# (       ) y-momentum
		
		I=np.identity(self.N)
		i=(0+1j)
		delta=self.D[1] -self.alpha**2 *I
		
		AA1=np.zeros((self.N,self.N))
		AA2=i*self.alpha*I
		AA3=self.D[0]

		AB1=i*self.alpha*I

		if (self.option['equation']=='Euler_wave' or self.option['equation']=='Euler') :
			AB2=i*self.alpha*np.diag(self.U)   
			AC3=i*self.alpha*np.diag(self.U)
		elif (self.option['equation']=='Euler_CD' or self.option['equation']=='Euler_CD_wave'):
			AB2=i*self.alpha*np.diag(self.U)  +np.diag(self.aCD*self.U) 
			AC3=+ i*self.alpha*np.diag(self.U) 

		elif self.option['equation']=='Euler_CD_turb':
			AB2=i*self.alpha*np.diag(self.U)  -(2*self.lc**2 )*(np.dot(np.diag(self.dU),self.D[1]) + np.dot( self.D[0] ,np.diag(self.ddU))     )
			AC3=+ i*self.alpha*np.diag(self.U)  

		elif self.option['equation']=='LNS':
			AB2=i*self.alpha*np.diag(self.U)  -delta/self.Re 
			AC3=i*self.alpha*np.diag(self.U)  -delta/self.Re
		elif self.option['equation']=='LNS_CD' :
			AB2=i*self.alpha*np.diag(self.U)  -delta/self.Re +np.diag(self.aCD*self.U) 
			AC3=+ i*self.alpha*np.diag(self.U)  -delta/self.Re
		elif self.option['equation']=='LNS_turb':
			AB2=i*self.alpha*np.diag(self.U)  -delta/self.Re -(2*self.lc**2 )*(np.dot(np.diag(self.dU),self.D[1]) + np.dot( self.D[0] ,np.diag(self.ddU))     )
			AC3=+ i*self.alpha*np.diag(self.U)  -delta/self.Re
		elif self.option['equation']=='LNS_turb_CD':
			AB2=i*self.alpha*np.diag(self.U)  -delta/self.Re  +np.diag(self.aCD*self.U)  -(2*self.lc**2 )*(np.dot(np.diag(self.dU),self.D[1]) + np.dot( self.D[0] ,np.diag(self.ddU))     )
			AC3=+ i*self.alpha*np.diag(self.U)  -delta/self.Re
		

		AB3=np.diag(self.dU)

		AC1=self.D[0]
		AC2=np.zeros((self.N,self.N)) 

		BA1=BA2=BA3=BB1=BB3=BC1=BC2=np.zeros((self.N,self.N))
		BB2=BC3=i*I*self.alpha

		AA=np.concatenate((AA1,AA2,AA3),axis=1)
		AB=np.concatenate((AB1,AB2,AB3),axis=1)
		AC=np.concatenate((AC1,AC2,AC3),axis=1)

		self.A=np.concatenate((AA,AB,AC))
		
		BA=np.concatenate((BA1,BA2,BA3),axis=1)
		BB=np.concatenate((BB1,BB2,BB3),axis=1)
		BC=np.concatenate((BC1,BC2,BC3),axis=1)

		self.B=np.concatenate((BA,BB,BC))

		if self.option['equation']=='Euler':
			self.BC_LNS_neu_v()
		elif self.option['equation']=='Euler_wave':
			self.BC_LNS_wave()
		elif self.option['equation']=='Euler_CD':
			self.BC_LNS_neu_v()
		elif self.option['equation']=='Euler_CD_wave':
			self.BC_LNS_wave()
		elif self.option['equation']=='Euler_CD_turb':
			self.BC_LNS_neu_v()
		elif self.option['equation']=='LNS':
			self.BC_LNS_neu_u_v()
		elif self.option['equation']=='LNS_CD':
			self.BC_LNS_neu_u_v()
		elif self.option['equation']=='LNS_turb':
			self.BC_LNS_neu_u_v()
		elif self.option['equation']=='LNS_turb_CD':
			self.BC_LNS_neu_u_v()
			

	def BC_LNS_neu_u_v(self):
		idx_bc=np.array([self.N,2*self.N,2*self.N -1,3*self.N-1]) #index of the 
		
		self.A[idx_bc,:]=np.zeros(3*self.N)
		self.B[idx_bc,:]=np.zeros(3*self.N)
		
		self.A[self.N,self.N]=1
		self.A[2*self.N -1,2*self.N -1]=1
		
		self.A[2*self.N ,2*self.N ]=1
		self.A[3*self.N -1,3*self.N -1]=1
		
		#print self.A, self.B


	def BC_LNS_neu_v(self):
		idx_bc=np.array([2*self.N,3*self.N -1]) #index of the 
		
		self.A[idx_bc,:]=np.zeros(3*self.N)
		self.B[idx_bc,:]=np.zeros(3*self.N)
		
				
		self.A[2*self.N ,2*self.N ]=1
		self.A[3*self.N -1,3*self.N -1]=1
		
		#print self.A, self.B

	def BC_LNS_wave(self):
		idx_bc=np.array([2*self.N,3*self.N -1]) #index of the 
		
		self.A[idx_bc,:]=np.zeros(3*self.N)
		self.B[idx_bc,:]=np.zeros(3*self.N)
		
		#v(0)=0		
		self.A[3*self.N -1,3*self.N -1]=1
		
		#v(y_max) --> equation
		self.A[2*self.N ,2*self.N]= -( np.cos(self.slope)/self.Fr**2)
		self.A[2*self.N,0]= (0+1j)*self.alpha*self.U[0]
		self.B[2*self.N, 0]=(0+1j)*self.alpha


	def interpolate(self):
		f_U=intp.interp1d(self.y_data,self.U_data)
		idx=np.where(self.y<self.y_data[-1])
		y_int=self.y[idx]
		self.U=np.concatenate([(np.ones(len(self.y)-len(y_int)))*self.U_data[-1],f_U(y_int)]) 
		
		f_dU=intp.interp1d(self.y_data,self.dU_data)		
		self.dU=np.concatenate([(np.ones(len(self.y)-len(y_int)))*0,f_dU(y_int)]) 
		
		f_ddU=intp.interp1d(self.y_data,self.ddU_data)		
		self.ddU=np.concatenate([(np.ones(len(self.y)-len(y_int)))*0,f_ddU(y_int)]) 

		f_aCD=intp.interp1d(self.y_data,self.aCD_data)		
		self.aCD=np.concatenate([(np.ones(len(self.y)-len(y_int)))*0,f_aCD(y_int)]) 
		
		f_daCD=intp.interp1d(self.y_data,self.daCD_data)		
		self.daCD=np.concatenate([(np.ones(len(self.y)-len(y_int)))*0,f_daCD(y_int)]) 

	#	if self.option['equation']=='LNS_CD_wave':
	#		self.y=self.y[	self.y<= self.['mapping'][1]]
			


		
		

	def plot_LNS(self):    
		for i in np.arange(10):
			#plt.rcParams.update({'font.size': 32})
			fig, ay = plt.subplots(figsize=(10,10), dpi=50)
			lines = ay.plot(self.eigv_re,self.eigv_im,'b*',lw=10)
			ay.set_ylabel(r'$c_i$',fontsize=32)
			ay.set_xlabel(r'$c_r$',fontsize=32)
			#lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay.set_ylim(self.option['plot_lim'][0])
			ay.set_xlim(self.option['plot_lim'][1])
			ay.grid()                                         
			#plt.tight_layout()
			fig.savefig('RESULTS'+'spectrum_bla.png', bbox_inches='tight',dpi=50)     
			#plt.show(lines)	
				
			sel_eig=plt.ginput(2)
		
			omega_r_picked=(sel_eig[0][0] +sel_eig[1][0])/2
			omega_i_picked=(sel_eig[0][1] +sel_eig[1][1])/2
		
			omega_picked=omega_r_picked*(1+0j)+ omega_i_picked*(0+1j)
			n=np.argmin(np.abs(self.eigv -omega_picked))
		
			eigfun_picked=self.eigf[:,n]
		
			print omega_picked
			
			
			p=eigfun_picked[0:self.N]
			u=eigfun_picked[self.N:2*self.N]
			v=eigfun_picked[2*self.N:3*self.N]

		
			fig2, (ay1,ay2,ay3) = plt.subplots(1,3)#, dpi=50)
			lines1 = ay1.plot(np.real(p),self.y,'r',np.imag(p),self.y,'g',np.sqrt(p*np.conjugate(p)),self.y,'m',lw=2)
			ay1.set_ylabel(r'$y$',fontsize=32)
			ay1.set_xlabel(r"$p$",fontsize=32)	
			lgd=ay1.legend((lines1),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay1.set_ylim([0,5])
			#ay1.set_xlim([-1, 1])
			ay1.grid()                                         
		
			lines2 = ay2.plot(np.real(u),self.y,'r',np.imag(u),self.y,'g',np.sqrt(u*np.conjugate(u)),self.y,'m',lw=2)
			ay2.set_ylabel(r'$y$',fontsize=32)
			ay2.set_xlabel(r"$u$",fontsize=32)	
			#lgd=ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay2.set_ylim([0,5])
			#ay2.set_xlim([-1, 1])
			ay2.grid()  


			lines3 = ay3.plot(np.real(v),self.y,'r',np.imag(v),self.y,'g',np.sqrt(v*np.conjugate(v)),self.y,'m',lw=2)
			ay3.set_ylabel(r'$y$',fontsize=32)
			ay3.set_xlabel(r"$v$",fontsize=32)	
			#lgd=ay3.legend((lines3),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay3.set_ylim([0,5])
			#ay3.set_xlim([-1, 1])
			ay3.grid()  

			fig2.savefig('fun.png', bbox_inches='tight',dpi=150)     
				
			plt.show(lines)	

	@nb.jit
	def omega_alpha_curves(self,alpha_start,alpha_end, n_step):
		self.vec_alpha=np.linspace(alpha_start, alpha_end,n_step)
		self.vec_eigv_im=np.zeros(n_step)
		for i in np.arange(n_step):
			self.set_perturbation(self.vec_alpha[i],self.Re)
			self.choose_variables()			
			self.solve_eig()
			#self.vec_eigv_im[i]=np.max(self.eigv_im)
			
			self.vec_eigv_im[i]=self.vec_alpha[i]*np.max(self.eigv_im[self.eigv_im<1])
			


			#print self.eigv_im
		np.savez('euler_cd_turb',self.vec_alpha,self.vec_eigv_im)
	
		fig, ay = plt.subplots(dpi=150)
		lines = ay.plot(self.vec_alpha,self.vec_eigv_im,'b',lw=2)
		ay.set_ylabel(r'$\omega_i$',fontsize=32)
		ay.set_xlabel(r'$\alpha$',fontsize=32)
		#lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
		#ay.set_ylim([-1,0.1])
		#ay.set_xlim([0, 1.8])
		ay.grid()                                         
		#plt.tight_layout()
		fig.savefig('euler_cd_turb.png', bbox_inches='tight',dpi=150)     
		plt.show(lines)


	def set_perturbation(self,a,Re):
		 self.alpha=a
		 self.Re=Re


	def superpose_spectrum(self,alpha_start,alpha_end, n_step):
		self.vec_alpha=np.linspace(alpha_start, alpha_end,n_step)
					
		
		sp_re=np.array([])
		sp_im=np.array([])
		
		bkpl.output_file("spectrum.html")
		

		TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,hover"

		p = bkpl.figure(plot_width=1000, plot_height=600, tools=TOOLS, title="Superimposed spectrum",x_axis_label='c_r', y_axis_label='c_i',x_range=(0.8,0.96), y_range=(-0.05,0.1) )

		COLORS=['aqua', 'blue', 'fuchsia', 'gold', 'green', 'orange', 'red', 'sienna', 'yellow', 'lime']   #from css list
		j=-1
		for i in np.arange(n_step):
			j=j+1
			self.set_perturbation(self.vec_alpha[i],self.Re)
			self.choose_variables()
			self.solve_eig()
			#sp_re=np.concatenate((sp_re,self.eigv_re))
			#sp_im=np.concatenate((sp_im, self.eigv_im))
			sp_re=self.eigv_re.tolist()
			sp_im=self.eigv_im.tolist()
			if j>9:
				j=j-10  #these with the above inizialization is needed for the iteration in the COLOURS list
			
			p.circle(sp_re,sp_im, size=10,fill_color=COLORS[j] )
		bkpl.show(p)

							




option={'flow':'DATA/G.txt', \
	'n_points':200, \
	'lc':0.16739, \
	'Ymax':300, \
	'perturbation':{'alpha':0.03, \
			'Re':160}, \
	'variables':'primitives', \
	'equation':'Euler_CD', \
	'BC':'Neumann', \
	'plot_lim':[[-0.02,0.02],[0.83,0.85]]  }



"""	
cc=fluid(option)




#cc.set_perturbation()
cc.diff_matrix()
#cc.set_poiseuille()
cc.read_velocity_profile()
#cc.read_velocity_profile('DATA/blasius.txt')

"""
"""
cc.mapping(1000)
#cc.interpolate()
cc.set_blasisus(cc.y)
cc.build_operator()
cc.BC1()
cc.solve_eig()
cc.plot_spectrum()
"""

"""
cc.mapping()
cc.interpolate()
#cc.set_blasisus(cc.y)
#cc.plot_velocity()

a=np.linspace(0.01,2,20)
omega_sel=np.zeros(len(a))
for i in np.arange(len(a)):
	cc.set_perturbation(a[i],160)
	cc.LNS()
	cc.solve_eig()
	#cc.plot_LNS_eigspectrum()
	print cc.eigv[(cc.eigv.real>0.832) & (cc.eigv.real<0.844) & (cc.eigv.imag<0.02) & (cc.eigv.imag>-0.02)]

	omega=a[i]*cc.eigv[(cc.eigv.real>0.8) & (cc.eigv.real<0.825) & (cc.eigv.imag<0.02) & (cc.eigv.imag>-0.05)]
	omega_sel[i]=omega.imag
	print omega_sel[i], a[i]

	#cc.plot_LNS_eigspectrum()
	#cc.omega_alpha_curves(0.01,2,30)
fig, ay = plt.subplots(dpi=50)
ay.plot(a,omega_sel,lw=2)
ay.set_ylabel(r'$\omega_i$',fontsize=32)
ay.set_xlabel(r'$alpha$',fontsize=32)
#ay.set_ylim([-1,0.1])
#ay.set_xlim([0, 1.8])
#plt.tight_layout()
#fig.savefig('ci_cr.png', bbox_inches='tight',dpi=50)     
#plt.hold(True)
ay.grid()
#fig.savefig('ci_cr.png', bbox_inches='tight',dpi=150)
plt.show()


#cc.LNS()
#cc.solve_eig()
#cc.plot_LNS_eigspectrum()
#cc.omega_alpha_variab_curves(0,2,20)
#cc.omega_alpha_variab_curves_only_4(0.01,2,10)

"""

