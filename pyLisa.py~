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



class fluid(object):
	"""
	fluid: Perform a linear stability analysis after building the operator(ex.Orr-Sommerfeld)
	"""
	def __init__(self,option,**kwargs):
		self.option=option
		self.N=option['n_points']
		self.y=np.linspace(-1,1,self.option['n_points'])
		self.U=np.zeros(len(self.y))
		self.CD=np.zeros(len(self.y))
		self.dU=np.zeros(len(self.y))
		self.ddU=np.zeros(len(self.y))


	def read_velocity_profile(self):
		in_txt=np.genfromtxt(self.option['flow'], delimiter=' ',skiprows=1) 
		self.y_data=in_txt[:,0]
		self.U_data=in_txt[:,1]
		self.dU_data=in_txt[:,2]
		self.ddU_data=in_txt[:,3]
		self.aCD_data=in_txt[:,4]
		#self.daCD_data=in_txt[:,5]
		self.lc=option['lc'] #0.16739  #lc*=0.22*(h-z1) / h 

	def set_poiseuille(self):
		Upoiseuille=(lambda y: 1-y**2)
		dUpoiseuille=(lambda y: -y*2)
		ddUpoiseuille=-np.ones(len(self.y))*2
		self.U=Upoiseuille(self.y)
		self.dU=dUpoiseuille(self.y)
		self.ddU=ddUpoiseuille


	def set_blasisus(self,y_gl):
		self.U, self.dU, self.ddU = bl.blasius(y_gl) 


	def plot_velocity(self):
		fig, ay = plt.subplots(figsize=(10,10), dpi=50)
		lines = ay.plot(self.U,self.y,'b',self.dU,self.y,'g',self.ddU,self.y,'r',self.aCD,self.y,'m',lw=2)
		ay.set_ylabel(r'$y$',fontsize=32)
		lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$',r'$a^* \dot C_D$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
		ay.set_ylim([0,5])
		#ax.set_xlim([np.min(time[2*T:3*T]),np.max(time[2*T:3*T])])
		ay.grid()                    
		#plt.tight_layout()
		fig.savefig('RESULTS'+'couette.png', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=50)     
		plt.show(lines)		
		
	def diff_matrix(self):
		self.y, self.D= cb.chebdif(self.N,4)  #in this line we re-instanciate the y in gauss lobatto points
		self.D=self.D + 0j
		"""
		self.D1=self.D[0]
		self.D2=self.D[1]
		self.D4=self.D[3]
  		"""
  
     #def mapping(self,method):
         #più tardi implementa metodi diversi per fare il mapping
     
         
  
	def build_operator(self):
		 I=np.identity(self.N)
		 self.A= np.dot(np.diag(self.alpha*self.U),(self.D[1]-I*self.alpha**2)) -np.diag(self.alpha*self.ddU) +((1/self.Re)*(self.D[3] -(2*self.alpha**2)*self.D[1] +(self.alpha**4)*I ))*(0+1j)
		 self.B=(self.D[1]-I*self.alpha**2)


	def set_perturbation(self):
		 self.alpha=option['perturbation']['alpha']
		 self.Re=option['perturbation']['Re']


	def BC1(self):
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

	def BC2(self):
		self.A[0,:]=np.zeros(self.N)
		self.A[-1,:]=np.zeros(self.N)
		self.B[0,:]=np.zeros(self.N)
		self.B[-1,:]=np.zeros(self.N)


	def BC3(self):
		self.A=self.A[1:-2,1:-2]
		self.B=self.B[1:-2,1:-2]




         
         
	def solve_eig(self):
		 self.eigv, self.eigf  = lin.eig(self.A,self.B) #, left=True, right=True)

		 #remove the infinite and nan eigenvectors, and their eigenfunctions
		 selector=np.isfinite(self.eigv)
		 self.eigv=self.eigv[selector]
		 self.eigf=self.eigf[:,selector]

		 self.eigv_re=np.real(self.eigv)
		 self.eigv_im=np.imag(self.eigv)
		
	
	def plot_spectrum(self):
		for i in np.arange(10):
			fig, ay = plt.subplots(figsize=(10,10), dpi=50)
			lines = ay.plot(self.eigv_re,self.eigv_im,'b*',lw=2)
			ay.set_ylabel(r'$\omega_i$',fontsize=32)
			ay.set_xlabel(r'$\omega_r$',fontsize=32)
			#lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay.set_ylim([-1,0.1])
			ay.set_xlim([0, 1])
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

			v=eigfun_picked
			u=np.dot((v/self.alpha),  self.D[0]) *(0+1j)

			fig2, (ay2,ay3) = plt.subplots(1,2)#, dpi=50)
			lines2 = ay2.plot(np.real(u),self.y,'r',np.imag(u),self.y,'g',np.sqrt(u*np.conjugate(u)),self.y,'m',lw=2)
			ay2.set_ylabel(r'$y$',fontsize=32)
			ay2.set_xlabel(r"$u^'$",fontsize=32)	
			#lgd=ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay2.set_ylim([0,5])
			#ay2.set_xlim([-1, 1])
			ay2.grid()  


			lines3 = ay3.plot(np.real(v),self.y,'r',np.imag(v),self.y,'g',np.sqrt(v*np.conjugate(v)),self.y,'m',lw=2)
			ay3.set_ylabel(r'$y$',fontsize=32)
			ay3.set_xlabel(r"$v^'$",fontsize=32)	
			#lgd=ay3.legend((lines3),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay3.set_ylim([0,5])
			#ay3.set_xlim([-1, 1])
			ay3.grid()  
			
			plt.show(lines)
		
	
		
	def mapping(self):
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

		#self.D = np.zeros((4,self.N, self.N)) +0j

		#print self.D[0]
		

		self.D[0] = np.dot(np.diag(xi[:,0]),self.D[0])
		self.D[1] = np.dot(np.diag(xi[:,0]**2) , self.D[1]) + np.dot(np.diag(xi[:,1]) , self.D[0])
		self.D[2] = np.dot(np.diag(xi[:,0]**3) , self.D[2]) + 3*np.dot(np.dot(np.diag(xi[:,0]),np.diag(xi[:,1])) ,self.D[1])   + np.dot(np.diag(xi[:,2]),self.D[0])

		self.D[3] = np.dot(np.diag(xi[:,0]**4),self.D[3])  + 6*np.dot(np.dot(np.diag(xi[:,1]),np.diag(xi[:,0]**2)),self.D[2])      + 4*np.dot(np.dot(np.diag(xi[:,2]),np.diag(xi[:,0])),self.D[1]) + 3*np.dot(np.diag(xi[:,1]**2),self.D[1]) + np.dot(np.diag(xi[:,3]),self.D[0]) 
		
		#scipy.io.savemat('test.mat', dict(x=self.D,y=xi))

		#print self.D[0]
		
		


	def LNS(self):
		I=np.identity(self.N)
		i=(0+1j)
		delta=self.D[1] -self.alpha**2 *I
		
		AA1=np.zeros((self.N,self.N))
		AA2=i*self.alpha*I
		AA3=self.D[0]

		AB1=i*self.alpha*I

		if self.option['equation']=='Euler':
			AB2=i*self.alpha*np.diag(self.U)   
			AC3=+ i*self.alpha*np.diag(self.U)
		elif self.option['equation']=='Euler_CD':
			AB2=i*self.alpha*np.diag(self.U)  +np.diag(self.aCD*self.U) 
			AC3=+ i*self.alpha*np.diag(self.U) 
		elif self.option['equation']=='LNS':
			AB2=i*self.alpha*np.diag(self.U)  -delta/self.Re 
			AC3=+ i*self.alpha*np.diag(self.U)  -delta/self.Re
		elif self.option['equation']=='LNS_CD':
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
		AC3=+ i*self.alpha*np.diag(self.U) # -delta/self.Re 

		BA1=BA2=BA3=BB1=BB3=BC1=BC2=np.zeros((self.N,self.N))
		BB2=BC3=i*self.alpha*I

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
		elif self.option['equation']=='Euler_CD':
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
		
		##self.A[self.N,self.N]=1
		#self.A[2*self.N -1,2*self.N -1]=1
		
		self.A[2*self.N ,2*self.N ]=1
		self.A[3*self.N -1,3*self.N -1]=1
		
		#print self.A, self.B


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
		#plt.plot(self.aCD,self.y,'b')
		#plt.show()
		

	def plot_LNS_eigspectrum(self):    
		for i in np.arange(10):
			fig, ay = plt.subplots(figsize=(10,10), dpi=50)
			lines = ay.plot(self.eigv_re,self.eigv_im,'b*',lw=2)
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
		
			print omega_picked, lin.norm(eigfun_picked), len(eigfun_picked)

			p=eigfun_picked[0:self.N]
			u=eigfun_picked[self.N:2*self.N]
			v=eigfun_picked[2*self.N:3*self.N]

		
			fig2, (ay1,ay2,ay3) = plt.subplots(1,3)#, dpi=50)
			lines1 = ay1.plot(np.real(p),self.y,'r',np.imag(p),self.y,'g',np.sqrt(p*np.conjugate(p)),self.y,'m',lw=2)
			ay1.set_ylabel(r'$y$',fontsize=32)
			ay1.set_xlabel(r"$p^'$",fontsize=32)	
			lgd=ay1.legend((lines1),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay1.set_ylim([0,5])
			#ay1.set_xlim([-1, 1])
			ay1.grid()                                         
		
			lines2 = ay2.plot(np.real(u),self.y,'r',np.imag(u),self.y,'g',np.sqrt(u*np.conjugate(u)),self.y,'m',lw=2)
			ay2.set_ylabel(r'$y$',fontsize=32)
			ay2.set_xlabel(r"$u^'$",fontsize=32)	
			#lgd=ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay2.set_ylim([0,5])
			#ay2.set_xlim([-1, 1])
			ay2.grid()  


			lines3 = ay3.plot(np.real(v),self.y,'r',np.imag(v),self.y,'g',np.sqrt(v*np.conjugate(v)),self.y,'m',lw=2)
			ay3.set_ylabel(r'$y$',fontsize=32)
			ay3.set_xlabel(r"$v^'$",fontsize=32)	
			#lgd=ay3.legend((lines3),(r'$Re$',r'$Im$',r'$Mod$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
			ay3.set_ylim([0,5])
			#ay3.set_xlim([-1, 1])
			ay3.grid()  

			fig2.savefig('RESULTS'+'spfunrum_bla.png', bbox_inches='tight',dpi=50)     

			plt.show(lines)	


	def omega_alpha_curves(self,alpha_start,alpha_end, n_step):
		self.vec_alpha=np.linspace(alpha_start, alpha_end,n_step)
		self.vec_eigv_im=np.zeros(n_step)
		for i in np.arange(n_step):
			self.set_perturbation(self.vec_alpha[i],self.Re)
			self.LNS()
			self.BC_LNS_neu_v()
			self.solve_eig()
			#self.vec_eigv_im[i]=np.max(self.eigv_im)
			
			self.vec_eigv_im[i]=self.vec_alpha[i]*np.max(self.eigv_im[self.eigv_im<1])
			


			#print self.eigv_im
		fig, ay = plt.subplots(dpi=50)
		lines = ay.plot(self.vec_alpha,self.vec_eigv_im,'b',lw=2)
		ay.set_ylabel(r'$\omega_i$',fontsize=32)
		ay.set_xlabel(r'$\alpha$',fontsize=32)
		#lgd=ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
		#ay.set_ylim([-1,0.1])
		#ay.set_xlim([0, 1.8])
		ay.grid()                                         
		#plt.tight_layout()
		fig.savefig('os_all.png', bbox_inches='tight',dpi=50)     
		plt.show(lines)




option={'flow':'DATA/G.txt', \
	'n_points':200, \
	'lc':0.16739, \
	'Ymax':300, \
	'perturbation':{'alpha':0.6, \
			'Re':160}, \
	'variables':'primitives', \
	'equation':'Euler_CD', \
	'BC':'Neumann', \
	'plot_lim':[[-1,0.1],[0,1.8]]  }



		
cc=fluid(option)




cc.set_perturbation()
cc.diff_matrix()
#cc.set_poiseuille()
cc.read_velocity_profile()
#cc.read_velocity_profile('DATA/blasius.txt')

"""
cc.mapping(1000)
#cc.interpolate()
cc.set_blasisus(cc.y)
cc.build_operator()
cc.BC1()
cc.solve_eig()
cc.plot_spectrum()

"""
cc.mapping()
cc.interpolate()
#cc.set_blasisus(cc.y)
cc.plot_velocity()
cc.LNS()
#cc.BC_LNS_neu_v()
cc.solve_eig()
cc.plot_LNS_eigspectrum()
#cc.omega_alpha_curves(0.1,1.6,30)

