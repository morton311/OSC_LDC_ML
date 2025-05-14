import numpy as np
import time
import h5py
from tqdm import tqdm
from lib.dls import *

def TKE_long(data_path: str, field_name: str, axis_order: list, snap_start: int = 0, snap_end: int = None) -> np.ndarray:
    """
    Calculate the turbulent kinetic energy (TKE) from the velocity data.
    The TKE is calculated as the sum of the squares of the velocity components divided by 2.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file containing the velocity data.
    field_name : str
        Name of the field in the HDF5 file that contains the velocity data.
    axis_order : list
        Axis indices for the velocity components in the HDF5 file.
        Order should be (num_variables, nx, ny, snapshots)
    """

    # Open the HDF5 file
    with h5py.File(data_path, 'r+') as f:
        if snap_start == 0 and snap_end is None:
            num_snapshots = f[field_name].shape[axis_order[3]]
        else:
            num_snapshots = snap_end - snap_start
        TKE = np.zeros((num_snapshots, 1))

        for i in tqdm(range(num_snapshots), desc="Loading data"):
            # Load the velocity data for the current snapshot
            index = [slice(None)] * len(axis_order)
            index[axis_order[3]] = i
            vel = f[field_name][tuple(index)] - f['mean'][:]
            # Calculate TKE for the current snapshot
            TKE[i] = 0.5 * np.sum((vel)**2) 
            
    return TKE

def latent_eval(data_path: str, latent_path: str, field_name: str, axis_order: list, config, snap_start: int = 0, snap_end: int = None) -> np.ndarray:
    """
    Evaluate the latent variables from the velocity data.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file containing the velocity data.
    latent_path : str
        Path to the HDF5 file containing the latent variables.
    field_name : str
        Name of the field in the HDF5 file that contains the velocity data.
    axis_order : list
        Axis indices for the velocity components in the HDF5 file.
        Order should be (num_variables, nx, ny, snapshots)
    snap_start : int
        Starting snapshot index.
    snap_end : int
        Ending snapshot index.
    """
    
    # Open the HDF5 files
    with h5py.File(data_path, 'r+') as f:
        with h5py.File(latent_path, 'r+') as f_latent:
            if snap_start == 0 and snap_end is None:
                num_snapshots = f[field_name].shape[axis_order[3]]
            else:
                num_snapshots = snap_end - snap_start

            l2_err = np.zeros((num_snapshots, 1))
            latent_RMS = np.zeros((2, config.nx_t, config.ny_t))
            latent_TKE = np.zeros((num_snapshots, 1))
            true_TKE = np.zeros((num_snapshots, 1))

            for i in tqdm(range(num_snapshots), desc="Loading data"):
                # Load the velocity data for the current snapshot
                index = [slice(None)] * len(axis_order)
                index[axis_order[3]] = i
                vel = f[field_name][tuple(index)] - f['mean'][:]
                vel = vel.transpose((np.array(axis_order[:-1]) - 1))
                vel = vel[:,:config.nx_t,:config.ny_t]

                dof_u = f_latent['dof_u'][i]
                dof_v = f_latent['dof_v'][i]

                vel_rec = gfem_recon(dof_u, dof_v, config)
                vel_rec = vel_rec.squeeze()

                latent_RMS += vel_rec**2 / num_snapshots

                latent_TKE[i] = 0.5 * np.sum((vel_rec)**2)
                true_TKE[i] = 0.5 * np.sum((vel)**2)

                # calculate err
                l2_err[i] = np.linalg.norm(vel[:,:config.nx_t,:config.ny_t] - vel_rec) / np.linalg.norm(vel[:,:config.nx_t,:config.ny_t])
                
            latent_RMS = np.sqrt(latent_RMS)
            
    return l2_err, latent_RMS, latent_TKE, true_TKE

