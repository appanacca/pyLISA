from pyLisa import *


a1, o1=np.load("mode_1.npz")

print a1[]



fig, ay = plt.subplots(dpi=150)
ay.plot(a1,o1,lw=2)
ay.set_ylabel(r'$\omega_i$',fontsize=32)
ay.set_xlabel(r'$\alpha$',fontsize=32)
ay.set_ylim([-0.005,0.1])
ay.set_xlim([0, 2])
ay.grid()
#fig.savefig('mode_3.png', bbox_inches='tight',dpi=150)     
plt.show()