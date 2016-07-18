import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb
import numpy as np

option = {'flow': 'DATA/H.txt',
          'a_ast': 1.104,  #0.552
          'n_points': 340,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 5,
          'alpha': 0.45,  #0.56552
          'Re': 1e5 ,   #157.922677   #1e5
          'variables': 'p_u_v', # v_eta
          'equation': 'LNS_CD',
          'mapping': ['semi_infinite_PB', [0, (46.7/13.8)]],
          'Froude': 0.02,
          'slope': 1.3e-5}


f = sa.fluid(option)

f.diff_matrix()
f.integ_matrix()
f.read_velocity_profile()
f.mapping()

f.interpolate()

# f.set_blasisus(f.y)

# f.infinite_mapping()
# f.set_hyptan()
# f.set_poiseuille()

f.set_operator_variables()

f.solve_eig()
f.adjoint_spectrum('cont')
f.solve_eig_adj()


file_name = option['flow'][-5]+"_"+str(option['Re'])

f.save_sim(file_name)
#f.check_adj()


v = po.viz(file_name)
v.plot_velocity()
v.plot_spectrum()

#f.omega_alpha_curves(0.1, 1, 20, 0.9, 1.1, 'G_RE_1e5')

idx = np.argmax(np.imag(f.eigv))
om = sn.sensitivity(file_name, idx, show_f=True)
a, b, c, d = om.c_per(obj='norm')
print (a, b, c,d)

f.omega_alpha_curves(0.1, 1, 20, 0.7, 1, name_file=file_name)
#om.sens_spectrum('ke_cd_N001_puv.png', 1e-7, 1e-4, 189, obj='u', shape='gauss') # eps, gamma
#om.validation(1, 1e-7, 1e-4, idx, 'gauss')
