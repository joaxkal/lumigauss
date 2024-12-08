import numpy as np
import torch
import scipy
import scipy.sparse
import numpy as np

# Snippets from OSR-NERF.
class Rotation(object):
	"""
		Example. Rotate an environment map using spherical harmonics:
		env = torch.randn(9, 3)
		angle = np.pi/2
		print(env)

		rotation = Rotation()
		rot = np.float32(np.dot(rotation.rot_y(angle), np.dot(rotation.rot_x(0.), rotation.rot_z(0.))))
		rot_sh = np.matmul(rot, env.clone().detach().cpu().numpy())
		print(rot_sh)
	"""

	def __init__(self):
		super(Rotation, self).__init__()
		rows_x = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8]
		cols_x = [0, 2, 1, 3, 7, 5, 6, 8, 4, 6, 8]

		data_x_p90 = [1,-1,1,1,-1,-1,-1/2,-np.sqrt(3)/2,1,-np.sqrt(3)/2,1/2]
		data_x_n90 = [1,1,-1,1,1,-1,-1/2,-np.sqrt(3)/2,-1,-np.sqrt(3)/2,1/2]

		self.Rot_X_p90 = scipy.sparse.coo_matrix((data_x_p90,(rows_x,cols_x)), shape=(9,9)).toarray()
		self.Rot_X_n90 = scipy.sparse.coo_matrix((data_x_n90,(rows_x,cols_x)), shape=(9,9)).toarray()

		self.rows_z = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
		self.cols_z = [0, 1, 3, 2, 1, 3, 4, 8, 5, 7, 6, 5, 7, 4, 8]


	def rot_SH(self, SH, thetaX, thetaY, thetaZ):


		Rot_y = []
		Rot_x = []
		Rot_z = []

		for rotx, roty, rotz in zip(thetaX, thetaY, thetaZ):
			Rot_z.append(self.rot_z(rotz))
			Rot_y.append(self.rot_y(roty))
			Rot_x.append(self.rot_x(rotx))

		Rot_y = np.stack(Rot_y, axis=0)
		Rot_x = np.stack(Rot_x, axis=0)
		Rot_z = np.stack(Rot_z, axis=0)


		Rot = np.matmul(Rot_z, np.matmul(Rot_y, Rot_x))
		rot_SH = np.matmul(Rot, SH)

		return rot_SH


	def rot_z(self, thetaZ):
		data_Z = [1,np.cos(thetaZ),np.sin(thetaZ),1,-np.sin(thetaZ),np.cos(thetaZ),np.cos(2*thetaZ),np.sin(2*thetaZ),np.cos(thetaZ),np.sin(thetaZ),1,-np.sin(thetaZ),np.cos(thetaZ),-np.sin(2*thetaZ),np.cos(2*thetaZ)]

		return scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()

	def rot_y(self, thetaY):
		data_Z = [1,np.cos(thetaY),np.sin(thetaY),1,-np.sin(thetaY),np.cos(thetaY),np.cos(2*thetaY),np.sin(2*thetaY),np.cos(thetaY),np.sin(thetaY),1,-np.sin(thetaY),np.cos(thetaY),-np.sin(2*thetaY),np.cos(2*thetaY)]
		rotM_z = scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()

		return np.matmul(self.Rot_X_p90, np.matmul(rotM_z, self.Rot_X_n90))


	def rot_x(self, thetaX):
		data_Z = [1,np.cos(thetaX),np.sin(thetaX),1,-np.sin(thetaX),np.cos(thetaX),np.cos(2*thetaX),np.sin(2*thetaX),np.cos(thetaX),np.sin(thetaX),1,-np.sin(thetaX),np.cos(thetaX),-np.sin(2*thetaX),np.cos(2*thetaX)]
		rotM_z = scipy.sparse.coo_matrix((data_Z,(self.rows_z, self.cols_z)), shape=(9,9)).toarray()
		
		return np.matmul(self.rot_y(np.pi/2), np.matmul(rotM_z, self.rot_y(-np.pi/2)))


def render_sphere_nm(radius, num):
	# nm is a batch of normal maps
	nm = []

	for i in range(num):
		### sphere (projected on circular image just like angular map)
		# span the regular grid for computing azimuth and zenith angular map
		height = 2*radius
		width = 2*radius
		centre = radius
		h_grid, v_grid = np.meshgrid(np.arange(1.,2*radius+1), np.arange(1.,2*radius+1))
		# grids are (-radius, radius)
		h_grid -= centre
		# v_grid -= centre
		v_grid = centre - v_grid
		# scale range of h and v grid in (-1,1)
		h_grid /= radius
		v_grid /= radius

		# z_grid is linearly spread along theta/zenith in range (0,pi)
		dist_grid = np.sqrt(h_grid**2+v_grid**2)
		dist_grid[dist_grid>1] = np.nan
		theta_grid = dist_grid * np.pi
		z_grid = np.cos(theta_grid)

		rho_grid = np.arctan2(v_grid,h_grid)
		x_grid = np.sin(theta_grid)*np.cos(rho_grid)
		y_grid = np.sin(theta_grid)*np.sin(rho_grid)

		# concatenate normal map
		nm.append(np.stack([x_grid,y_grid,z_grid],axis=2))

	# construct batch
	nm = np.stack(nm,axis=0)
	return nm

def sh_recon(nm, lighting):
	width = nm.shape[1]

	x = nm[:,:,:,0]
	y = nm[:,:,:,1]
	z = nm[:,:,:,2]

	# convert light probe to angular map(evenly distributed front and back environment), find light directions by new angular map
	azi = np.arctan2(y, x)
	zen = np.arccos(z)

	c1 = 0.282095
	c2 = 0.488603
	c3 = 1.092548
	c4 = 0.315392
	c5 = 0.546274

	# domega = 4*np.pi**2/width**2 * sinc(zen)
	domega = np.ones_like(zen)

	sh_basis = np.stack([c1 * domega, c2*y * domega, c2*z * domega, c2*x * domega, c3*x*y * domega, c3*y*z * domega, c4*(3*z*z-1) * domega, c3*x*z * domega, c5*(x*x-y*y) * domega], axis=1)
	sh_basis = np.expand_dims(sh_basis, axis=-1)
	lighting_recon = np.expand_dims(np.expand_dims(lighting,axis=-2),axis=-2) * sh_basis
	lighting_recon = np.sum(lighting_recon,axis=1)

	return lighting_recon


# Example: Visualize SH reconstruction
# from utils.sh_rotate_utils import sh_recon, render_sphere_nm
# import numpy as np
# import matplotlib.pyplot as plt
# img_sh = np.array([
#     [2.5, 2.3, 2.5],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0]
# ])
# lighting_recon = sh_recon(np.float32(render_sphere_nm(100, 1)), img_sh)
# lighting_validPix = lighting_recon[np.logical_not(np.isnan(lighting_recon))]
# lighting_recon = (lighting_recon - lighting_validPix.min()) / (lighting_validPix.max() - lighting_validPix.min())
# lighting_recon[np.isnan(lighting_recon)] = 1
# plt.figure()
# plt.imshow(lighting_recon[0])
