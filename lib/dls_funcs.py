import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time
import h5py
from tqdm import tqdm
from scipy.sparse.linalg import factorized

def random_patch_sampling(data, patch_size):
    num_patches = 10000
    num_images = 1
    nx = data.shape[0]
    ny = data.shape[1]
    sz = patch_size
    BUFF = 0
    totalsamples = 0
    X = np.zeros((sz ** 2, num_patches))
    
    for i in range(num_images):
        this_image = data

        # Determine how many patches to take
        getsample = num_patches // num_images
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(getsample):
            d1 = BUFF + np.random.randint(0, nx - sz - 2 * BUFF)
            d2 = BUFF + np.random.randint(0, ny - sz - 2 * BUFF)
            
            totalsamples += 1
            temp = this_image[d1:d1 + sz, d2:d2 + sz].reshape(sz ** 2, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X

def Modal_decomp_2D(data, patch_size):
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal

def FEM_shape_calculator_2D_ortho_gfemlr(x, y, xpt, ypt):
    sumxpt = np.sum(xpt) / 4
    sumypt = np.sum(ypt) / 4

    dxpt = (-xpt[0] + xpt[1] + xpt[2] - xpt[3]) / 2
    dypt = (ypt[0] + ypt[1] - ypt[2] - ypt[3]) / 2

    zeta_i = [-1, 1, 1, -1]
    eta_i = [1, 1, -1, -1]

    # Inverse transform for parallelogram elements, bilinear shape functions
    zeta = 2 * (x - sumxpt) / dxpt
    eta = 2 * (y - sumypt) / dypt

    N = np.zeros((4,1))
    # shape function values
    for i in range(4):
        N[i] = (1 / 4) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta)
    return N

def gfem_2d(data, patch_size, num_modes):
    num_snaps = data.shape[-1]
    nx = data.shape[1]
    ny = data.shape[2]
    grid_x = np.linspace(1, nx, nx)
    grid_y = np.linspace(1, ny, ny)
    [grid_x, grid_y] = np.meshgrid(grid_x, grid_y)
    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    Q_grid_u = data[0,:nx_t, :ny_t, :]
    Q_grid_v = data[1,:nx_t, :ny_t, :]

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element


    # Compute local modes
    local_modes_u, eigVal = Modal_decomp_2D(data[0,:,:,-1], patch_size)
    local_modes_v, eigVal = Modal_decomp_2D(data[1,:,:,-1], patch_size)
    # print(local_modes.shape)
    mode_grid_u = local_modes_u[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')
    mode_grid_v = local_modes_v[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')

    # Mode grid components for the four quadrants
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt-1, nskip_sample + 1))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt-1, nskip_sample + 1))

    modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

    modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

    for i in range(num_modes):
        modes_grid_1_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

        modes_grid_1_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

    modes_vec_1_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_u = np.zeros((mid_pt ** 2, num_modes))

    modes_vec_1_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_v = np.zeros((mid_pt ** 2, num_modes))

    for i in range(num_modes):
        modes_vec_1_comp_u[:, i] = modes_grid_1_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_u[:, i] = modes_grid_2_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_u[:, i] = modes_grid_3_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_u[:, i] = modes_grid_4_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')

        modes_vec_1_comp_v[:, i] = modes_grid_1_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_v[:, i] = modes_grid_2_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_v[:, i] = modes_grid_3_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_v[:, i] = modes_grid_4_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
    
    i = 1
    j = 1

    M_local_u = np.zeros((dof_elem, dof_elem))
    M_local_v = np.zeros((dof_elem, dof_elem))
    
    # x, y locations of the GFEM element nodes
    x1 = grid_x[i*nskip,       (j-1)*nskip]
    x2 = grid_x[i*nskip,       j*nskip]
    x3 = grid_x[(i-1)*nskip,   j*nskip]
    x4 = grid_x[(i-1)*nskip,   (j-1)*nskip]

    y1 = grid_y[i*nskip,       (j-1)*nskip]
    y2 = grid_y[i*nskip,       j*nskip]
    y3 = grid_y[(i-1)*nskip,   j*nskip]
    y4 = grid_y[(i-1)*nskip,   (j-1)*nskip]


    # Combining x, y nodal coordinates into vector form
    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((nskip+1)**2)
    N2 = np.zeros((nskip+1)**2)
    N3 = np.zeros((nskip+1)**2)
    N4 = np.zeros((nskip+1)**2)

    for kx in range(nskip+1):
        indx = (i-1)*nskip + kx
        for ky in range(nskip+1):
            indy = (j-1)*nskip + ky
            x_val = grid_x[indy,indx]
            y_val = grid_y[indy,indx]

            # shape functions over the grid points

            iind = ky*(nskip+1) + kx

            N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    modemat_local_u = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_u,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_u,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_u,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_u
    ])
    
    modemat_local_v = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_v,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_v,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_v,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_v
    ])

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    L_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
    L_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global GFEM matrices')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            L_local_u = np.zeros((dof_elem, 1))
            L_local_v = np.zeros((dof_elem, 1))

            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            indx_cell = (i-1) * nskip
            indy_cell = (j-1) * nskip

            M_u[np.ix_(lltogl, lltogl)] = M_u[np.ix_(lltogl, lltogl)] + M_local_u
            M_v[np.ix_(lltogl, lltogl)] = M_v[np.ix_(lltogl, lltogl)] + M_local_v

            for id in range(num_snaps):
                indx_cell = i * nskip
                indy_cell = j * nskip

                Q_local_u = Q_grid_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, id]
                Q_local_v = Q_grid_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, id]

                Q_local_u_vec = np.zeros((nskip+1)**2)
                Q_local_v_vec = np.zeros((nskip+1)**2)


                for kx in range(nskip+1):
                    for ky in range(nskip+1):
                        iind = ky*(nskip+1) + kx
                        
                        Q_local_u_vec[iind] = Q_local_u[kx, ky]
                        Q_local_v_vec[iind] = Q_local_v[kx, ky]

                L_local_u = modemat_local_u_wt.T @ Q_local_u_vec.T
                L_local_v = modemat_local_v_wt.T @ Q_local_v_vec.T

                L_u[lltogl, id] = L_u[lltogl, id] + L_local_u
                L_v[lltogl, id] = L_v[lltogl, id] + L_local_v

    
    print('Done constructing global matrices')
    # plt.spy(M, markersize=2)
    print(M_u.shape, L_u.shape)

    # Convert lilmatrix to csr matrix
    M_u = csr_matrix(M_u)
    M_v = csr_matrix(M_v)

    dof_u = spsolve(M_u, L_u)
    dof_v = spsolve(M_v, L_v)

    class Config:
        def __init__(self, data, patch_size, num_modes, modemat_local_u, modemat_local_v):
            self.nx = data.shape[1]
            self.ny = data.shape[2]
            self.num_snaps = data.shape[3]
            self.patch_size = patch_size
            self.num_modes = num_modes
            self.nskip = (patch_size - 1) // 2
            self.nskip_sample = patch_size - 1
            self.mid_pt = 1 + self.nskip_sample // 2
            self.sample_x = range(0, self.nx, self.nskip)
            self.sample_y = range(0, self.ny, self.nskip)
            self.nx_t = max(self.sample_x) + 1
            self.ny_t = max(self.sample_y) + 1
            self.nx_g = len(self.sample_x)
            self.ny_g = len(self.sample_y)
            self.num_gfem_nodes = self.nx_g * self.ny_g
            self.dof_node = num_modes + 1
            self.dof_elem = 4 * self.dof_node
            self.modemat_local_u = modemat_local_u
            self.modemat_local_v = modemat_local_v
            self.compression_ratio = data.shape[0]*num_snaps*self.nx*self.ny / (data.shape[0]*num_snaps*self.dof_node + data.shape[0] * num_modes * self.patch_size**2 )


    config = Config(data=data, patch_size=patch_size, num_modes=num_modes, modemat_local_u=modemat_local_u, modemat_local_v=modemat_local_v)

    return dof_u, dof_v, config



def gfem_recon(dof_u, dof_v, config):

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    nskip = config.nskip
    dof_node = config.dof_node # DOFs/node
    dof_elem = config.dof_elem # DOFs/element
    
    # if one dimensional, make 2d
    if len(dof_u.shape) == 1:
        dof_u = dof_u[:, np.newaxis]
        dof_v = dof_v[:, np.newaxis]

    Q_rec_u = np.zeros((config.nx_t, config.ny_t, dof_u.shape[-1]))
    Q_rec_v = np.zeros((config.nx_t, config.ny_t, dof_v.shape[-1]))

    for i in range(config.nx_g-1):
        for j in range(config.ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*config.ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)
            # print(lltogl)
            for id in range(dof_u.shape[-1]):
                Q_rec_local_u_vec = config.modemat_local_u @ dof_u[lltogl, id]
                Q_rec_local_v_vec = config.modemat_local_v @ dof_v[lltogl, id]
                
                Q_rec_local_u = Q_rec_local_u_vec.reshape((nskip+1, nskip+1), order='F')
                Q_rec_local_v = Q_rec_local_v_vec.reshape((nskip+1, nskip+1), order='F')

                Q_rec_u[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_u
                Q_rec_v[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_v
    Q_rec = np.zeros((2, config.nx_t, config.ny_t, dof_u.shape[-1]))
    Q_rec[0, :, :, :] = Q_rec_u
    Q_rec[1, :, :, :] = Q_rec_v
    return Q_rec

class dls_Config:
    def __init__(self, data, patch_size, num_modes, modemat_local_u, modemat_local_v):
        self.nx = data.shape[1]
        self.ny = data.shape[2]
        self.num_snaps = data.shape[3]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.num_gfem_nodes = self.nx_g * self.ny_g
        self.dof_node = num_modes + 1
        self.dof_elem = 4 * self.dof_node
        self.modemat_local_u = modemat_local_u
        self.modemat_local_v = modemat_local_v
        self.compression_ratio = data.shape[0]*self.num_snaps*self.nx*self.ny / (data.shape[0]*self.num_snaps*self.dof_node + data.shape[0] * num_modes * self.patch_size**2 )

def gfem_2d_1t(data, patch_size, num_modes):
    num_snaps = data.shape[-1]
    nx = data.shape[1]
    ny = data.shape[2]
    grid_x = np.linspace(1, nx, nx)
    grid_y = np.linspace(1, ny, ny)
    [grid_x, grid_y] = np.meshgrid(grid_x, grid_y)
    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    Q_grid_u = data[0,:nx_t, :ny_t, :]
    Q_grid_v = data[1,:nx_t, :ny_t, :]

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element


    # Compute local modes
    local_modes_u, eigVal = Modal_decomp_2D(data[0,:,:,0], patch_size)
    local_modes_v, eigVal = Modal_decomp_2D(data[1,:,:,0], patch_size)
    # print(local_modes.shape)
    mode_grid_u = local_modes_u[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')
    mode_grid_v = local_modes_v[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')

    # Mode grid components for the four quadrants
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt-1, nskip_sample + 1))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt-1, nskip_sample + 1))

    modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

    modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

    for i in range(num_modes):
        modes_grid_1_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

        modes_grid_1_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

    modes_vec_1_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_u = np.zeros((mid_pt ** 2, num_modes))

    modes_vec_1_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_v = np.zeros((mid_pt ** 2, num_modes))

    for i in range(num_modes):
        modes_vec_1_comp_u[:, i] = modes_grid_1_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_u[:, i] = modes_grid_2_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_u[:, i] = modes_grid_3_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_u[:, i] = modes_grid_4_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')

        modes_vec_1_comp_v[:, i] = modes_grid_1_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_v[:, i] = modes_grid_2_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_v[:, i] = modes_grid_3_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_v[:, i] = modes_grid_4_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
    
    i = 1
    j = 1

    M_local_u = np.zeros((dof_elem, dof_elem))
    M_local_v = np.zeros((dof_elem, dof_elem))
    
    # x, y locations of the GFEM element nodes
    x1 = grid_x[i*nskip,       (j-1)*nskip]
    x2 = grid_x[i*nskip,       j*nskip]
    x3 = grid_x[(i-1)*nskip,   j*nskip]
    x4 = grid_x[(i-1)*nskip,   (j-1)*nskip]

    y1 = grid_y[i*nskip,       (j-1)*nskip]
    y2 = grid_y[i*nskip,       j*nskip]
    y3 = grid_y[(i-1)*nskip,   j*nskip]
    y4 = grid_y[(i-1)*nskip,   (j-1)*nskip]


    # Combining x, y nodal coordinates into vector form
    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((nskip+1)**2)
    N2 = np.zeros((nskip+1)**2)
    N3 = np.zeros((nskip+1)**2)
    N4 = np.zeros((nskip+1)**2)

    for kx in range(nskip+1):
        indx = (i-1)*nskip + kx
        for ky in range(nskip+1):
            indy = (j-1)*nskip + ky
            x_val = grid_x[indy,indx]
            y_val = grid_y[indy,indx]

            # shape functions over the grid points

            iind = ky*(nskip+1) + kx

            N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    modemat_local_u = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_u,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_u,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_u,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_u
    ])
    
    modemat_local_v = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_v,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_v,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_v,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_v
    ])

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global M GFEM matrix')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            M_u[np.ix_(lltogl, lltogl)] = M_u[np.ix_(lltogl, lltogl)] + M_local_u
            M_v[np.ix_(lltogl, lltogl)] = M_v[np.ix_(lltogl, lltogl)] + M_local_v

    # Convert lilmatrix to csr matrix
    M_u = M_u.tocsc()
    M_v = M_v.tocsc()

    dof_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
    dof_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

    print('M constructed')

    
    print('Constructing global L GFEM matrix')
    L_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
    L_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            L_local_u = np.zeros((dof_elem, num_snaps))
            L_local_v = np.zeros((dof_elem, num_snaps))

            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            indx_cell = i * nskip
            indy_cell = j * nskip

            Q_local_u = Q_grid_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]
            Q_local_v = Q_grid_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]

            Q_local_u_vec = np.zeros(((nskip+1)**2, num_snaps))
            Q_local_v_vec = np.zeros(((nskip+1)**2, num_snaps))

            for kx in range(nskip+1):
                for ky in range(nskip+1):
                    iind = ky*(nskip+1) + kx
                    
                    Q_local_u_vec[iind, :] = Q_local_u[kx, ky, :]
                    Q_local_v_vec[iind, :] = Q_local_v[kx, ky, :]
            # print(Q_local_u_vec.T.shape, Q_local_v_vec.T.shape)
            # print(modemat_local_u_wt.T.shape, modemat_local_v_wt.T.shape)
            # print((modemat_local_u_wt.T @ Q_local_u_vec).shape, (modemat_local_v_wt.T @ Q_local_v_vec).shape)
            # print(L_local_u.shape, L_local_v.shape)
            L_local_u = modemat_local_u_wt.T @ Q_local_u_vec
            L_local_v = modemat_local_v_wt.T @ Q_local_v_vec

            L_u[lltogl,:] = L_u[lltogl,:] + L_local_u
            L_v[lltogl,:] = L_v[lltogl,:] + L_local_v

    print('Done constructing global matrices')

    
    print('Prefactorizing M')
    # Pre-factorize the matrices for efficiency
    solve_M_u = factorized(M_u)
    solve_M_v = factorized(M_v)
    print('Done prefactorizing M')
    print('Solving for dof')
    for id in tqdm(range(num_snaps)):
        dof_u[:, id] = solve_M_u(L_u[:, id])
        dof_v[:, id] = solve_M_v(L_v[:, id])

    print('Done solving for dof')


    config = dls_Config(data=data, patch_size=patch_size, num_modes=num_modes, modemat_local_u=modemat_local_u, modemat_local_v=modemat_local_v)

    return dof_u, dof_v, config

def gfem_2d_long(data_path: str, field_name: str, latent_file: str, patch_size: int, num_modes: int, batch_size: int = 2500):
    with h5py.File(data_path, 'r') as f:
        num_snaps = f[field_name].shape[0]
        nx = f[field_name].shape[1]
        ny = f[field_name].shape[2]
        num_vars = f[field_name].shape[3]
        mode_data = f[field_name][0,:,:,:].transpose(2,0,1) - f['mean'][:].transpose(2,0,1)
        print('shape of mode data: ', mode_data.shape)
        print('number of snapshots: ', num_snaps)
        print('number of batches: ', num_snaps // batch_size)
        print('nx: ', nx)
        print('ny: ', ny)
        print('num_vars: ', num_vars)


    grid_x = np.linspace(1, nx, nx)
    grid_y = np.linspace(1, ny, ny)
    [grid_x, grid_y] = np.meshgrid(grid_x, grid_y)
    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element


    # Compute local modes
    local_modes_u, eigVal = Modal_decomp_2D(mode_data[0], patch_size)
    local_modes_v, eigVal = Modal_decomp_2D(mode_data[1], patch_size)
    # print(local_modes.shape)
    mode_grid_u = local_modes_u[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')
    mode_grid_v = local_modes_v[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')

    # Mode grid components for the four quadrants
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt-1, nskip_sample + 1))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt-1, nskip_sample + 1))

    modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

    modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

    for i in range(num_modes):
        modes_grid_1_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

        modes_grid_1_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

    modes_vec_1_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_u = np.zeros((mid_pt ** 2, num_modes))

    modes_vec_1_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_v = np.zeros((mid_pt ** 2, num_modes))

    for i in range(num_modes):
        modes_vec_1_comp_u[:, i] = modes_grid_1_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_u[:, i] = modes_grid_2_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_u[:, i] = modes_grid_3_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_u[:, i] = modes_grid_4_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')

        modes_vec_1_comp_v[:, i] = modes_grid_1_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_v[:, i] = modes_grid_2_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_v[:, i] = modes_grid_3_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_v[:, i] = modes_grid_4_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
    
    i = 1
    j = 1

    M_local_u = np.zeros((dof_elem, dof_elem))
    M_local_v = np.zeros((dof_elem, dof_elem))
    
    # x, y locations of the GFEM element nodes
    x1 = grid_x[i*nskip,       (j-1)*nskip]
    x2 = grid_x[i*nskip,       j*nskip]
    x3 = grid_x[(i-1)*nskip,   j*nskip]
    x4 = grid_x[(i-1)*nskip,   (j-1)*nskip]

    y1 = grid_y[i*nskip,       (j-1)*nskip]
    y2 = grid_y[i*nskip,       j*nskip]
    y3 = grid_y[(i-1)*nskip,   j*nskip]
    y4 = grid_y[(i-1)*nskip,   (j-1)*nskip]


    # Combining x, y nodal coordinates into vector form
    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((nskip+1)**2)
    N2 = np.zeros((nskip+1)**2)
    N3 = np.zeros((nskip+1)**2)
    N4 = np.zeros((nskip+1)**2)

    for kx in range(nskip+1):
        indx = (i-1)*nskip + kx
        for ky in range(nskip+1):
            indy = (j-1)*nskip + ky
            x_val = grid_x[indy,indx]
            y_val = grid_y[indy,indx]

            # shape functions over the grid points

            iind = ky*(nskip+1) + kx

            N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    modemat_local_u = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_u,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_u,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_u,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_u
    ])
    
    modemat_local_v = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_v,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_v,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_v,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_v
    ])

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global M GFEM matrix')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            M_u[np.ix_(lltogl, lltogl)] = M_u[np.ix_(lltogl, lltogl)] + M_local_u
            M_v[np.ix_(lltogl, lltogl)] = M_v[np.ix_(lltogl, lltogl)] + M_local_v



    print('M constructed')

    print('Prefactorizing M')
    # Convert lilmatrix to csr matrix
    M_u = M_u.tocsc()
    M_v = M_v.tocsc()
    # Pre-factorize the matrices for efficiency
    solve_M_u = factorized(M_u)
    solve_M_v = factorized(M_v)
    print('Done prefactorizing M')

    # create h5 dataset for dof of shape (num_gfem_nodes*dof_node, num_snaps)
    dof_file = h5py.File(latent_file, 'w')
    dof_file.create_dataset('dof_u', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')
    dof_file.create_dataset('dof_v', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')

    dof_u = np.zeros((num_gfem_nodes * dof_node, batch_size))
    dof_v = np.zeros((num_gfem_nodes * dof_node, batch_size))


    print('Looping through snapshots, solving for dofs')
    with h5py.File(data_path, 'r') as f:
        for i in tqdm(range(num_snaps // batch_size)):
            snap_start = i * batch_size
            snap_end = (i + 1) * batch_size
            u_mean = f['mean'][:, :, 0]
            v_mean = f['mean'][:, :, 1]
            Q_grid_u = f[field_name][snap_start:snap_end,:,:,0]
            Q_grid_v = f[field_name][snap_start:snap_end,:,:,1]
            Q_grid_u = Q_grid_u.transpose(1,2,0) - u_mean[:,:, np.newaxis]
            Q_grid_v = Q_grid_v.transpose(1,2,0) - v_mean[:,:, np.newaxis]
    
            L_u = np.zeros((num_gfem_nodes * dof_node, batch_size))
            L_v = np.zeros((num_gfem_nodes * dof_node, batch_size))

            for i in range(nx_g-1):
                for j in range(ny_g-1):
                    L_local_u = np.zeros((dof_elem, batch_size))
                    L_local_v = np.zeros((dof_elem, batch_size))

                    lltogl = np.zeros(dof_elem, dtype=int)
                    for lindx in range(4):
                        indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                        indx_dof_end = indx_dof_start + dof_node

                        lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

                    indx_cell = i * nskip
                    indy_cell = j * nskip

                    Q_local_u = Q_grid_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]
                    Q_local_v = Q_grid_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]

                    Q_local_u_vec = np.zeros(((nskip+1)**2, batch_size))
                    Q_local_v_vec = np.zeros(((nskip+1)**2, batch_size))

                    for kx in range(nskip+1):
                        for ky in range(nskip+1):
                            iind = ky*(nskip+1) + kx
                            
                            Q_local_u_vec[iind, :] = Q_local_u[kx, ky, :]
                            Q_local_v_vec[iind, :] = Q_local_v[kx, ky, :]

                    L_local_u = modemat_local_u_wt.T @ Q_local_u_vec
                    L_local_v = modemat_local_v_wt.T @ Q_local_v_vec

                    L_u[lltogl,:] = L_u[lltogl, :] + L_local_u
                    L_v[lltogl,:] = L_v[lltogl, :] + L_local_v
            
            dof_u = solve_M_u(L_u)
            dof_v = solve_M_v(L_v)

            dof_file['dof_u'][snap_start:snap_end] = dof_u.T
            dof_file['dof_v'][snap_start:snap_end] = dof_v.T
            

    print('Done solving for dof')


    config = dls_long_Config(data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v)

    return config


class dls_long_Config:
    def __init__(self, data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v):
        with h5py.File(data_path, 'r') as f:
            self.num_snaps = f[field_name].shape[0]
            self.nx = f[field_name].shape[1]
            self.ny = f[field_name].shape[2]
            self.num_vars = f[field_name].shape[3]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.num_gfem_nodes = self.nx_g * self.ny_g
        self.dof_node = num_modes + 1
        self.dof_elem = 4 * self.dof_node
        self.modemat_local_u = modemat_local_u
        self.modemat_local_v = modemat_local_v
        self.compression_ratio = self.num_vars*self.num_snaps*self.nx*self.ny / (self.num_vars*self.num_snaps*self.dof_node + self.num_vars * num_modes * self.patch_size**2 )


def gfem_recon_long(dof_path, config):
    with h5py.File(dof_path, 'r') as f:
        dof_u = f['dof_u'][:]
        dof_v = f['dof_v'][:]

        IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

        nskip = config.nskip
        dof_node = config.dof_node # DOFs/node
        dof_elem = config.dof_elem # DOFs/element

        Q_rec_u = np.zeros((config.nx_t, config.ny_t, dof_u.shape[-1]))
        Q_rec_v = np.zeros((config.nx_t, config.ny_t, dof_v.shape[-1]))

        for i in range(config.nx_g-1):
            for j in range(config.ny_g-1):
                lltogl = np.zeros(dof_elem, dtype=int)
                for lindx in range(4):
                    indx_dof_start = ((i+IJK[lindx, 0])*config.ny_g + (j+IJK[lindx, 1]))*dof_node
                    indx_dof_end = indx_dof_start + dof_node

                    lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)
                # print(lltogl)
                for id in range(dof_u.shape[-1]):
                    Q_rec_local_u_vec = config.modemat_local_u @ dof_u[lltogl, id]
                    Q_rec_local_v_vec = config.modemat_local_v @ dof_v[lltogl, id]
                    
                    Q_rec_local_u = Q_rec_local_u_vec.reshape((nskip+1, nskip+1), order='F')
                    Q_rec_local_v = Q_rec_local_v_vec.reshape((nskip+1, nskip+1), order='F')

                    Q_rec_u[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_u
                    Q_rec_v[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_v
        Q_rec = np.zeros((2, config.nx_t, config.ny_t, dof_u.shape[-1]))
        Q_rec[0, :, :, :] = Q_rec_u
        Q_rec[1, :, :, :] = Q_rec_v
    return Q_rec