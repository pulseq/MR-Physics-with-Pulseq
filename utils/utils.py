import os
import torch
import numpy as np
import urllib.request
import pypulseq as pp
from scipy import ndimage

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm

try:
    import MRzeroCore as mr0
    mr0_available = True
except ImportError:
    mr0_available = False

try:
    import mapvbvd
    mapvbvd_available = True
except ImportError:
    mapvbvd_available = False

import sigpy as sp
from sigpy.mri.sim import birdcage_maps

# Read Siemens raw data in twix format
def read_raw_data(filename):
    if not mapvbvd_available:
        raise RuntimeError('read_raw_data called, but mapvbvd is not available. Run: !pip install pymapvbvd')

    # Read twix file
    twix_obj = mapvbvd.mapVBVD(filename)
    
    # If twix obj contains multiple scans, pick last one
    if isinstance(twix_obj, list):
        twix_obj = twix_obj[-1]
    
    # Load unsorted data: Do not remove oversampling and do not reflect lines according the the REV label
    twix_obj.image.removeOS = False
    twix_obj.image.disableReflect = True
    kdata = twix_obj.image.unsorted() # Shape: [N_adc, N_coils, N_meas]
    
    # Load in phasecor data and sort it into kdata by memory position (i.e. acquisition number)
    if hasattr(twix_obj, 'phasecor'):
        twix_obj.phasecor.removeOS = False
        twix_obj.phasecor.disableReflect = True
        kdata_phasecor = twix_obj.phasecor.unsorted() # Shape: [N_adc, N_coils, N_meas]
        
        inds = np.argsort(np.concatenate((twix_obj.image.memPos, twix_obj.phasecor.memPos)))
        
        kdata = np.concatenate((kdata, kdata_phasecor), axis=-1)[:,:,inds]
    
    # Reorder to: [N_coils, N_meas, N_adc]
    kdata = kdata.transpose(1,2,0)
    
    return kdata

# Simulate a sequence using a 2D brain phantom (200x200x8 mm)
# seq can be either a filename, or a sequence object
# sim_size: When provided, will resize the simulation matrix to this size (i.e. to speed up the simulations)
# noise_level: When set to value other than zero, will add complex-valued gaussian noise to the simulated signal
# n_coils: When set to value greater than 1, will simulate multiple coils with a birdcage pattern
def simulate_2d(seq, sim_size=None, noise_level=0, n_coils=1, dB0=0, B0_scale=1, B0_polynomial=None):
    if not mr0_available:
        raise RuntimeError('simulate_2d called, but MRzeroCore is not available. Run: !pip install mrzerocore')

    # Download .mat file for phantom if necessary
    sim_url = 'https://github.com/mzaiss/MRTwin_pulseq/raw/mr0-core/data/numerical_brain_cropped.mat'
    sim_filename = os.path.basename(sim_url)
    if not os.path.exists(sim_filename):
        print(f'Downloading {sim_url}...')
        urllib.request.urlretrieve(sim_url, sim_filename)
    
    # If seq is not a str, assume it is a sequence object and write to a temporary sequence file
    if not isinstance(seq, str):
        seq.write('tmp.seq')
        seq_filename = 'tmp.seq'
        adc_samples = int(seq.adc_library.data[1][0])
    else:
        seq_filename = seq
        tmp_seq = pp.Sequence()
        tmp_seq.read(seq_filename)
        adc_samples = int(tmp_seq.adc_library.data[1][0])
    
    # Create phantom from .mat file
    obj_p = mr0.VoxelGridPhantom.load_mat(sim_filename)
    if sim_size is not None:
        obj_p = obj_p.interpolate(sim_size[0], sim_size[1], 1)

    # Insert coil sensitivities generate by sigpy if multicoil data is requested
    if n_coils > 1:
        csm = birdcage_maps([n_coils, obj_p.PD.shape[0], obj_p.PD.shape[1]])
        obj_p.coil_sens = torch.from_numpy(csm[...,None])

    # Manipulate loaded data
    obj_p.B0 *= B0_scale
    obj_p.B0 += dB0
    
    if B0_polynomial is not None:
        x,y = torch.meshgrid(torch.linspace(-1,1,obj_p.PD.shape[0]),torch.linspace(-1,1,obj_p.PD.shape[1]))
        
        obj_p.B0 = B0_polynomial[0]
        if len(B0_polynomial) > 1:
            obj_p.B0 += x * B0_polynomial[1]
        if len(B0_polynomial) > 2:
            obj_p.B0 += y * B0_polynomial[2]
        if len(B0_polynomial) > 3:
            obj_p.B0 += x*x * B0_polynomial[3]
        if len(B0_polynomial) > 4:
            obj_p.B0 += y*y * B0_polynomial[4]
        if len(B0_polynomial) > 5:
            obj_p.B0 += x*y * B0_polynomial[5]
        
        obj_p.B0 = obj_p.B0[:,:,None]
    
    obj_p.D *= 0
    
    # Convert Phantom into simulation data
    obj_p = obj_p.build()

    # MR zero simulation
    seq0 = mr0.Sequence.import_file(seq_filename)
    
    # Remove temporary sequence file
    if not isinstance(seq, str):
        os.unlink('tmp.seq')
    
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-5)
    signal = mr0.execute_graph(graph, seq0, obj_p)
    kdata = signal.reshape(-1, adc_samples, n_coils).numpy().transpose(2,0,1) # Reshape to [N_coils, N_meas, N_adc]

    # Add noise to the simulated data
    if noise_level > 0:
        kdata += noise_level * (np.random.randn(*kdata.shape) + 1j * np.random.randn(*kdata.shape))
    
    return kdata


# Centered FFT and IFFT functions. 1D/2D/3D operate on the last dimensions of the matrix. ND operates on the specified axes.
def fft_1d(x):
    return fft_nd(x, (-1,))

def ifft_1d(x):
    return ifft_nd(x, (-1,))

def fft_2d(x):
    return fft_nd(x, (-2,-1))

def ifft_2d(x):
    return ifft_nd(x, (-2,-1))

def fft_3d(x):
    return fft_nd(x, (-3,-2,-1))

def ifft_3d(x):
    return ifft_nd(x, (-3,-2,-1))

def fft_nd(x, axes):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)

def ifft_nd(x, axes):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)



# Function to sort kspace data according to labels provided in the sequence
# Returns data in shape: [N_coils, (Nseg), (Nset), (Nrep), (Nphs), (Neco), (Nslc), (Nlin), Nx]
def sort_data_labels(kdata, seq, shape=None):
    n_coils = kdata.shape[0]
    adc_samples = kdata.shape[2]

    supported_labels = ['SEG', 'SET', 'REP', 'PHS', 'ECO', 'SLC', 'LIN', 'AVG']

    # Get label evolutions from sequence
    labels = seq.evaluate_labels(evolution='adc')

    # Reverse lines
    if 'REV' in labels:
        rev = labels['REV'] != 0
        kdata[:, rev, :] = kdata[:, rev, ::-1]

    # Find out which labels are used in the sequence and calculate kspace shape
    index = []
    label_shape = []
    used_labels = []
    for lab in supported_labels:
        if lab in labels:
            index.append(labels[lab])
            label_shape.append(labels[lab].max() + 1)
            used_labels.append(lab)
    
    label_shape += [adc_samples]
    
    if shape is None:
        shape = label_shape
        print(f'Automatically detected matrix size: {used_labels + ["ADC"]} {shape}')
    elif len(shape) != len(label_shape):
        raise ValueError('Provided shape does not have the same number of dimensions as the number of labels')

    # Assigned measured data to kspace
    kspace_matrix = np.zeros((n_coils,) + tuple(shape), dtype=np.complex64)
    kspace_matrix[(slice(None),) + tuple(index) + (slice(None),)] = kdata

    # Do averaging
    if 'AVG' in labels:
        kspace_matrix = kspace_matrix.mean(axis=-2)

    return kspace_matrix

# Find unique values in x that differ by at least 'tol'
def unique_isclose(x, tol=1e-4):
    sorted_x = np.sort(x)
    
    current_value = sorted_x[0]
    unique = [current_value]
    for value in sorted_x[1:]:
        if value - current_value < tol:
            continue
        
        current_value = value
        unique.append(value)
    
    return np.array(unique)


# Returns the slice position for each ADC event
def get_adc_slice_positions(seq):
    slice_pos = (0,0,0)
    curr_dur = 0
    
    gw_pp = seq.get_gradients()
    
    slice_positions = []
    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)
        
        if block.rf is not None:
            rf = block.rf
            t = rf.delay + pp.calc_rf_center(rf)[0]
            
            gx = gw_pp[0](curr_dur + t) if gw_pp[0] else 0
            gy = gw_pp[1](curr_dur + t) if gw_pp[1] else 0
            gz = gw_pp[2](curr_dur + t) if gw_pp[2] else 0
             
            if not hasattr(rf, "use") or block.rf.use in [
                "excitation",
                "undefined",
            ]:
                slice_pos = (rf.freq_offset / gx if gx != 0 else 0,
                             rf.freq_offset / gy if gy != 0 else 0,
                             rf.freq_offset / gz if gz != 0 else 0)
                
            elif block.rf.use == "refocusing":
                slice_pos_new = (rf.freq_offset / gx if gx != 0 else 0,
                                 rf.freq_offset / gy if gy != 0 else 0,
                                 rf.freq_offset / gz if gz != 0 else 0)
                
                # If refocusing is slice-selective, update slice_pos, else keep slice_pos from excitation
                if slice_pos_new != (0,0,0):
                    slice_pos = slice_pos_new
        
        if block.adc is not None:  # ADC
            slice_positions.append(slice_pos)
    
        curr_dur += seq.block_durations[block_counter]
    
    return slice_positions

# Function to sort kspace data according to the kspace trajectory as calculated
# by pypulseq
# Returns data in shape: [N_coils, (N_reps), (Nz), Ny, Nx]
def sort_data_implicit(kdata, seq, shape=None):
    n_coils = kdata.shape[0]
    adc_samples = kdata.shape[2]
    
    # Calculate k-space trajectory
    ktraj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    assert kdata.shape[1] * kdata.shape[2] == ktraj_adc.shape[1]
    
    fov = list(seq.get_definition('FOV'))

    if fov:
        delta_kx = 1 / fov[0]
        delta_ky = 1 / fov[1]
        delta_kz = 1 / fov[2]
    else:
        # Determine unique x,y,z k-space locations
        x_coords = unique_isclose(np.unique(np.round(ktraj_adc[0], 5)))
        y_coords = unique_isclose(np.unique(np.round(ktraj_adc[1], 5)))
        z_coords = unique_isclose(np.unique(np.round(ktraj_adc[2], 5)))
        
        # Determine delta-k
        delta_kx = np.diff(x_coords).min()
        delta_ky = np.diff(y_coords).min()
        if z_coords.shape[0] > 1:
            delta_kz = np.diff(z_coords).min()
        else:
            delta_kz = 1
        
        print(f'Automatically detected FOV: [{round(1/delta_kz,2)}, {round(1/delta_ky,2)}, {round(1/delta_kx,2)}]')
       
    # Calculate x,y,z indices, assuming k-space is centered around k0
    
    # Shift trajectory. Ideally we wouldn't need this, but otherwise sampling at
    # e.g -1.5, -0.5, 0.5, 1.5 is prone to rounding errors
    shift_x = abs(ktraj_adc[0]).min()
    shift_y = abs(ktraj_adc[1]).min()
    shift_z = abs(ktraj_adc[2]).min()
    
    if abs(shift_x) / delta_kx > 1 or abs(shift_y) / delta_ky > 1 or abs(shift_z) / delta_kz > 1:
        print(f'Warning: Large shift used in data sorting. Is trajectory not centered on k0? (Shift = [{shift_x / delta_kx:.2f}, {shift_y / delta_kz:.2f}, {shift_z / delta_kz:.2f}])')

    x_index = np.round((ktraj_adc[0] - shift_x) / delta_kx).astype(int)
    y_index = np.round((ktraj_adc[1] - shift_y) / delta_ky).astype(int)
    
    # Detect multi-slice sequence (only in Z direction)
    slice_positions = np.array(get_adc_slice_positions(seq))[:,2]
    unique_slice_positions = unique_isclose(slice_positions)
    
    if len(unique_slice_positions) > 1:
        print(f'Detected multi-slice sequence! ({len(unique_slice_positions)} slices)')
        z_index = np.searchsorted(unique_slice_positions, slice_positions)
        z_index = z_index[:,None].repeat(adc_samples,1).flatten()
    else:
        z_index = np.round((ktraj_adc[2] - shift_z) / delta_kz).astype(int)
    
    if (abs((ktraj_adc[0] - shift_x) / delta_kx - x_index) > 0.5).any():
        print('Warning: Some (or all) X coordinates do not align to Cartesian grid')
    if (abs((ktraj_adc[1] - shift_y) / delta_ky - y_index) > 0.5).any():
        print('Warning: Some (or all) Y coordinates do not align to Cartesian grid')
    if len(unique_slice_positions) <= 1 and (abs((ktraj_adc[2] - shift_z) / delta_kz - z_index) > 0.5).any():
        print('Warning: Some (or all) Z coordinates do not align to Cartesian grid')

    # Get kspace size
    if shape:
        nz,ny,nx = shape
    else:
        nx = max(-x_index.min()*2, x_index.max()*2 + 1)
        ny = max(-y_index.min()*2, y_index.max()*2 + 1)
        
        if len(unique_slice_positions) <= 1:
            nz = max(-z_index.min()*2, z_index.max()*2 + 1)
        else:
            nz = z_index.max() + 1
        print(f'Automatically detected matrix size: ({nz}, {ny}, {nx})')
        
        if nz > 1024 or ny > 1024 or nx > 1024:
            raise RuntimeError('Large matrix size detected, stopping to prevent memory issues!')

    x_index += nx//2
    y_index += ny//2
    
    if len(unique_slice_positions) <= 1:
        z_index += nz//2

    # Determine repetition number of duplicated k-space locations
    linear_index = x_index + y_index*nx + z_index*nx*ny
        
    inds = np.argsort(linear_index, kind='stable')
    r = np.arange(len(inds))
    r -= np.maximum.accumulate(r * (np.diff(np.concatenate(([0], linear_index[inds]))) != 0))
    repeat_index = np.zeros(linear_index.shape, dtype=int)
    repeat_index[inds] = r
    
    # Create kspace matrix and assign measured data
    if repeat_index.max() > 0:
        if nz > 1:
            kspace_matrix = np.zeros((n_coils, repeat_index.max() + 1, nz, ny, nx), dtype=np.complex64)
            kspace_matrix[:, repeat_index, z_index, y_index, x_index] = kdata.reshape(n_coils, -1)
        else:
            kspace_matrix = np.zeros((n_coils, repeat_index.max() + 1, ny, nx), dtype=np.complex64)
            kspace_matrix[:, repeat_index, y_index, x_index] = kdata.reshape(n_coils, -1)
    else:
        if nz > 1:
            kspace_matrix = np.zeros((n_coils, nz, ny, nx), dtype=np.complex64)
            kspace_matrix[:, z_index, y_index, x_index] = kdata.reshape(n_coils, -1)
        else:
            kspace_matrix = np.zeros((n_coils, ny, nx), dtype=np.complex64)
            kspace_matrix[:, y_index, x_index] = kdata.reshape(n_coils, -1)

    return kspace_matrix


# Sum of squares coil combination, assumes x is a matrix of size [N_coils, ...]
def combine_coils(x):
    return np.sqrt((abs(x)**2).sum(axis=0))

def recon_cartesian_2d(kdata, seq, shape=None, use_labels=None):
    if use_labels is None:
        # Detect if sequence used labels
        use_labels = seq.label_set_library.data != {} or seq.label_inc_library.data != {}
        if use_labels:
            print('Detected labels in the sequence!')
        else:
            print('Did not detect labels in the sequence, using kspace calculation for sorting!')
    
    if use_labels:
        kspace = sort_data_labels(kdata, seq, shape=shape)
    else:
        kspace = sort_data_implicit(kdata, seq, shape=shape)
    im = ifft_2d(kspace)
    if im.shape[0] > 1:
        sos = combine_coils(im)
    else:
        sos = im[0]
    
    return sos

def recon_cartesian_3d(kdata, seq, shape=None, use_labels=None):
    if use_labels is None:
        # Detect if sequence used labels
        use_labels = seq.label_set_library.data != {} or seq.label_inc_library.data != {}
        if use_labels:
            print('Detected labels in the sequence!')
        else:
            print('Did not detect labels in the sequence, using kspace calculation for sorting!')
    
    if use_labels:
        kspace = sort_data_labels(kdata, seq, shape=shape)
    else:
        kspace = sort_data_implicit(kdata, seq, shape=shape)
    im = ifft_3d(kspace)
    
    if im.shape[0] > 1:
        sos = combine_coils(im)
    else:
        sos = im[0]
    
    return sos


def reconstruct(kdata, seq, shape=None, use_labels=None, cartesian=None, is_3d=None, lambda_l2=0.01, lambda_tv=0, trajectory_delay=0):
    # Autodetect of cartesian or 3D requires k-space trajectory
    if cartesian is None or is_3d is None:
        ktraj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    if cartesian is None:
        # Automatically detect axis-aligned Cartesian sampling
        # Assume a Cartesian sequence has no more than 512 unique sampling locations in X/Y/Z
        x_coords = unique_isclose(np.unique(np.round(ktraj_adc[0], 5)))
        y_coords = unique_isclose(np.unique(np.round(ktraj_adc[1], 5)))
        z_coords = unique_isclose(np.unique(np.round(ktraj_adc[2], 5)))
        
        if (x_coords.shape[0] <= 512
            and y_coords.shape[0] <= 512
            and z_coords.shape[0] <= 512):
            
            x_steps = np.diff(x_coords)
            y_steps = np.diff(y_coords)
            z_steps = np.diff(z_coords)
            
            if ((x_steps.shape[0] <= 1 or (abs((x_steps / x_steps.min()) % 1) < 1e-3).all())
                and (y_steps.shape[0] <= 1 or (abs((y_steps / y_steps.min()) % 1) < 1e-3).all())
                and (z_steps.shape[0] <= 1 or (abs((z_steps / z_steps.min()) % 1) < 1e-3).all())):
                cartesian = True
                print('Automatically detected Cartesian sequence')
            else:
                cartesian = False
                print('Automatically detected non-Cartesian sequence (failed integer coordinate check)')
        else:
            cartesian = False
            print('Automatically detected non-Cartesian sequence (>512 unique locations)')    
    
    if is_3d is None:
        # Automatically detect 3D sequence: Are all Z k-space locations the same?
        is_3d = not np.isclose(ktraj_adc[2], ktraj_adc[2][0], atol=1e-4).all()
        
        if is_3d:
            print('Automatically detected 3D sequence')
        else:
            print('Automatically detected 2D sequence')
    
    if is_3d:
        if cartesian:
            return recon_cartesian_3d(kdata, seq, shape=shape, use_labels=use_labels)
        else:
            return recon_nufft_3d(kdata, seq, shape, lambda_l2=lambda_l2, lambda_tv=lambda_tv, trajectory_delay=trajectory_delay)
    else:
        if cartesian:
            return recon_cartesian_2d(kdata, seq, shape=shape, use_labels=use_labels)
        else:
            return recon_nufft_2d(kdata, seq, shape, lambda_l2=lambda_l2, lambda_tv=lambda_tv, trajectory_delay=trajectory_delay)
    

def recon_nufft_2d(kdata, seq, shape=None, lambda_l2=0.01, lambda_tv=0, max_iter=None, trajectory_delay=0, profile_range=None):
    if max_iter is None:
        if lambda_tv == 0:
            max_iter = 10
        else:
            max_iter = 100

    fov = seq.get_definition('FOV')
    delta_k = 1/np.array(fov)

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=trajectory_delay)

    coords = k_traj_adc  / delta_k[:,None]
    coords = coords[[1,0,2]] # Transpose X and Y coordinates

    if shape is None:
        shape = (int(np.ceil(abs(coords[0]).max()))*2, int(np.ceil(abs(coords[1]).max()))*2)
        print(f'Automatically detected matrix size: {shape}')
        
        if any(x>1024 for x in shape):
            raise RuntimeError('Large matrix size detected, stopping to prevent memory issues!')

    # Subsample acquisitions if requested
    if profile_range is not None:
        coords = coords.reshape(3, *kdata.shape[1:])

        coords = coords[:,profile_range[0]:profile_range[1]]
        kdata = kdata[:,profile_range[0]:profile_range[1]]
        
        coords = coords.reshape(3, -1)
        
    kdata_shape = kdata.shape
    kdata = kdata.reshape(kdata.shape[0], -1)

    # Detect multi-slice sequence (only in Z direction)
    slice_positions = np.array(get_adc_slice_positions(seq))[:,2]
    unique_slice_positions = unique_isclose(slice_positions)
    
    if len(unique_slice_positions) > 1:
        print(f'Detected multi-slice sequence! ({len(unique_slice_positions)} slices)')
        
        recs = []
        for slice_pos in unique_slice_positions:
            mask = np.isclose(slice_positions, slice_pos, atol=1e-4)
        
            coords_subset = coords.reshape(3, *kdata_shape[1:])[:,mask].reshape(3,-1)
            kdata_subset = kdata.reshape(kdata_shape)[:,mask].reshape(kdata.shape[0],-1)
        
            if lambda_tv != 0:
                app = sp.mri.app.TotalVariationRecon(kdata_subset, np.ones((kdata.shape[0],) + shape + (1,)), lamda=lambda_tv*abs(kdata.mean()), coord=coords_subset.T, max_iter=max_iter)
            else:
                nufft = sp.linop.NUFFT((kdata.shape[0],) + shape + (1,), coords_subset.T, 2)
                app = sp.app.LinearLeastSquares(nufft, kdata_subset, lamda=lambda_l2*abs(kdata.mean()), max_iter=max_iter)
            
            rec = app.run()
            
            if lambda_tv == 0 and rec.shape[0] > 1:
                if rec.shape[0] > 1:
                    rec = combine_coils(rec)
                else:
                    rec = rec[0]
            
            recs.append(rec)
        
        rec = np.stack(recs, axis=0)
        
    else:
        if lambda_tv != 0:
            app = sp.mri.app.TotalVariationRecon(kdata, np.ones((kdata.shape[0],) + shape + (1,)), lamda=lambda_tv*abs(kdata.mean()), coord=coords.T, max_iter=max_iter)
        else:
            nufft = sp.linop.NUFFT((kdata.shape[0],) + shape + (1,), coords.T, 2)
            app = sp.app.LinearLeastSquares(nufft, kdata, lamda=lambda_l2*abs(kdata.mean()), max_iter=max_iter)
        
        rec = app.run()
        
        if lambda_tv == 0:
            if rec.shape[0] > 1:
                rec = combine_coils(rec)
            else:
                rec = rec[0]


    return rec[...,0]


def recon_nufft_3d(kdata, seq, shape=None, lambda_l2=0.01, lambda_tv=0, max_iter=None, trajectory_delay=0):
    if max_iter is None:
        if lambda_tv == 0:
            max_iter = 10
        else:
            max_iter = 100

    fov = seq.get_definition('FOV')
    delta_k = 1/np.array(fov)

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=trajectory_delay)
    kdata= kdata.reshape(kdata.shape[0], -1)
    
    coords = k_traj_adc  / delta_k[:,None]
    coords = coords[[2,1,0]] # Transpose Z, X and Y coordinates
 
    if shape is None:
        shape = (int(np.ceil(abs(coords[0]).max()))*2, int(np.ceil(abs(coords[1]).max()))*2, int(np.ceil(abs(coords[2]).max()))*2)
        print(f'Automatically detected matrix size: {shape}')
        if any(x>1024 for x in shape):
            raise RuntimeError('Large matrix size detected, stopping to prevent memory issues!')

    if lambda_tv != 0:
        app = sp.mri.app.TotalVariationRecon(kdata, np.ones((kdata.shape[0],) + shape), lamda=lambda_tv*abs(kdata.mean()), coord=coords.T, max_iter=max_iter)
    else:
        nufft = sp.linop.NUFFT((kdata.shape[0],) + shape, coords.T, 2)
        app = sp.app.LinearLeastSquares(nufft, kdata, lamda=lambda_l2*abs(kdata.mean()), max_iter=max_iter)
    
    rec = app.run()
    
    if lambda_tv == 0:
        if rec.shape[0] > 1:
            rec = combine_coils(rec)
        else:
            rec = rec[0]

    return rec



# Animate the k-space trajectory of a seq
# seq = Sequence object
# dt = time per frame (sec)
# plot_window = width of the RF/GX/GY/GZ plots (sec), always centered on the current time point
# time_range = [begin, end], provides begin and end times (in sec) for the animation (e.g. [1,2] to plot the sequence from 1 sec to 2 sec)
# fps = frames per second when rendering the animation
# max_frames = Maximal number of frames to render (overrides end of time_range if shorter)
# show = Show the animation window (not useful in Colab)
# save_filename = Filename to save the animation to (either .mp4 or .gif)
# show_progress = Show progress bar when saving the animation
def animate(seq, dt=1e-3, plot_window=1e-2, time_range=None, fps=30, max_frames=None, show=True, save_filename=None, show_progress=False):
    if time_range is None:
        time_range = [0, seq.duration()[0]]

    delta_kx = 1/seq.get_definition('FOV')[0]
    delta_ky = 1/seq.get_definition('FOV')[1]
    delta_kz = 1/seq.get_definition('FOV')[2]

    ts = np.linspace(time_range[0], time_range[1], int(np.ceil((time_range[1] - time_range[0])/dt))+1)

    fig = plt.figure()
    
    gs = GridSpec(7,2, figure=fig)

    ax_rf = fig.add_subplot(gs[0,:])
    ax_x = fig.add_subplot(gs[1,:])
    ax_y = fig.add_subplot(gs[2,:])
    ax_z = fig.add_subplot(gs[3,:])

    ax_kspace = fig.add_subplot(gs[4:,0])
    ax_kspace2 = fig.add_subplot(gs[4:,1])

    for a in [ax_rf, ax_x, ax_y, ax_z, ax_kspace, ax_kspace2]:
        a.xaxis.set_ticklabels([])
        a.yaxis.set_ticklabels([])
        a.spines['left'].set_position('zero')
        a.spines['bottom'].set_position('zero')
        a.spines['right'].set_color('none')
        a.spines['top'].set_color('none')

    for a in [ax_rf, ax_x, ax_y, ax_z]:
        a.spines['left'].set_position('center')
        a.spines['left'].set_color('none')
        a.yaxis.set_ticks([])


    p_vrf = ax_rf.axvline(0, color='r')
    p_vx = ax_x.axvline(0, color='r')
    p_vy = ax_y.axvline(0, color='r')
    p_vz = ax_z.axvline(0, color='r')

    gw_pp = seq.get_gradients()
    wv = seq.waveforms(append_RF=True)

    ax_rf.plot(wv[3][0].real, wv[3][1].real)
    ax_rf.plot(wv[3][0].real, wv[3][1].imag)

    if gs[0] != None:
        ax_x.plot(wv[0][0], wv[0][1])

    if gs[0] != None:
        ax_y.plot(wv[1][0], wv[1][1])

    if gs[0] != None:
        ax_z.plot(wv[2][0], wv[2][1])

    total_duration = sum(seq.block_durations.values())


    # Copy of calculate_kspace, but with grad_raster_time sampling of every non-zero gradient
    t_excitation, fp_excitation, t_refocusing, _ = seq.rf_times()
    t_adc, _ = seq.adc_times()
    ng = 3
    eps = 1e-8
    gm_pp = []
    tc = []
    for i in range(ng):
        if gw_pp[i] is None:
            gm_pp.append(None)
            continue

        gm_pp.append(gw_pp[i].antiderivative())
        tc.append(gm_pp[i].x)
        # "Sample" ramps for display purposes otherwise piecewise-linear display (plot) fails
        ii = np.flatnonzero((abs(gm_pp[i].c)>0).any(0))

        # Do nothing if there are no ramps
        if ii.shape[0] == 0:
            continue

        starts = np.int64(np.floor((gm_pp[i].x[ii] + eps) / seq.grad_raster_time))
        ends = np.int64(np.ceil((gm_pp[i].x[ii+1] - eps) / seq.grad_raster_time))

        # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
        lengths = ends-starts+1
        inds = np.ones((lengths).sum())
        # Calculate output index where each range will start
        start_inds = np.cumsum(np.concatenate(([0],lengths[:-1])))
        # Create element-wise differences that will cumsum into
        # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
        inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

        tc.append(np.cumsum(inds) * seq.grad_raster_time)
    if tc != []:
        tc = np.concatenate(tc)

    t_acc = 1e-10  # Temporal accuracy
    t_acc_inv = 1 / t_acc
    t_ktraj = t_acc * np.unique(
        np.round(
            t_acc_inv
            * np.array(
                [
                    *tc,
                    0,
                    *np.asarray(t_excitation) - 2 * seq.rf_raster_time,
                    *np.asarray(t_excitation) - seq.rf_raster_time,
                    *t_excitation,
                    *np.asarray(t_refocusing) - seq.rf_raster_time,
                    *t_refocusing,
                    *t_adc,
                    total_duration,
                ]
            )
        )
    )

    t_acc = 1e-10  # Temporal accuracy
    t_acc_inv = 1 / t_acc
    i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
    i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
    i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

    i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
    if len(i_excitation) > 0:
        ii_next_excitation = 0
    else:
        ii_next_excitation = -1
    if len(i_refocusing) > 0:
        ii_next_refocusing = 0
    else:
        ii_next_refocusing = -1

    k_traj = np.zeros((ng, len(t_ktraj)))
    for i in range(ng):
        if gw_pp[i] is None:
            continue

        it = np.where(np.logical_and(
            t_ktraj >= t_acc * round(t_acc_inv * gm_pp[i].x[0]),
            t_ktraj <= t_acc * round(t_acc_inv * gm_pp[i].x[-1]),
        ))[0]
        k_traj[i, it] = gm_pp[i](t_ktraj[it])
        if t_ktraj[it[-1]] < t_ktraj[-1]:
            k_traj[i, it[-1] + 1 :] = k_traj[i, it[-1]]

    # Convert gradient moments to kspace
    dk = -k_traj[:, 0]
    for i in range(len(i_periods) - 1):
        i_period = i_periods[i]
        i_period_end = i_periods[i + 1]
        if ii_next_excitation >= 0 and i_excitation[ii_next_excitation] == i_period:
            if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > t_acc:
                raise Warning(
                    f"abs(t_ktraj[i_period]-t_excitation[ii_next_excitation]) < {t_acc} failed for ii_next_excitation={ii_next_excitation} error={t_ktraj(i_period) - t_excitation(ii_next_excitation)}"
                )
            dk = -k_traj[:, i_period]
            if i_period > 0:
                # Use nans to mark the excitation points since they interrupt the plots
                k_traj[:, i_period - 1] = np.NaN
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_excitation = min(len(i_excitation) - 1, ii_next_excitation + 1)
        elif (
            ii_next_refocusing >= 0 and i_refocusing[ii_next_refocusing] == i_period
        ):
            dk = -2 * k_traj[:, i_period] - dk
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_refocusing = min(len(i_refocusing) - 1, ii_next_refocusing + 1)

        k_traj[:, i_period:i_period_end] = (
            k_traj[:, i_period:i_period_end] + dk[:, None]
        )

    k_traj[:, i_period_end] = k_traj[:, i_period_end] + dk
    k_traj_adc = k_traj[:, i_adc]


    p_kspace = ax_kspace.plot([],[])[0]
    p_adc = ax_kspace.plot([],[], 'r.', markersize=1)[0]
    p_cursor = ax_kspace.plot([],[], 'kx')[0]

    p_kspace2 = ax_kspace2.plot([],[])[0]
    p_adc2 = ax_kspace2.plot([],[], 'r.', markersize=1)[0]
    p_cursor2 = ax_kspace2.plot([],[], 'kx')[0]


    ax_kspace.set_xlim(np.nanmin(k_traj_adc[0]) - delta_kx*10, np.nanmax(k_traj_adc[0]) + delta_kx*10)
    ax_kspace.set_ylim(np.nanmin(k_traj_adc[1]) - delta_ky*10, np.nanmax(k_traj_adc[1]) + delta_ky*10)

    ax_kspace2.set_xlim(np.nanmin(k_traj_adc[0]) - delta_kx*10, np.nanmax(k_traj_adc[0]) + delta_kx*10)
    ax_kspace2.set_ylim(np.nanmin(k_traj_adc[2]) - delta_kz*10, np.nanmax(k_traj_adc[2]) + delta_kz*10)

    frames = len(ts)-1
    if max_frames is not None:
        frames = min(max_frames, frames)

    if show_progress:
        progress_bar = tqdm(total=frames)

    def update(frame):
        if show_progress:
            progress_bar.update(frame + 1 - progress_bar.n)
        t_start,t_end = list(zip(ts[:-1], ts[1:]))[frame]

        ax_rf.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_x.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_y.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_z.set_xlim(t_start - plot_window/2, t_end + plot_window/2)

        t = (t_start + t_end)/2
        p_vrf.set_data(([t,t], [0,1]))
        p_vx.set_data(([t,t], [0,1]))
        p_vy.set_data(([t,t], [0,1]))
        p_vz.set_data(([t,t], [0,1]))

        mask = np.flatnonzero((t_ktraj <= t_end) & (t_ktraj >= time_range[0]))
        mask_adc = np.flatnonzero((t_adc <= t_end) & (t_adc >= time_range[0]))

        p_kspace.set_xdata(k_traj[0,mask])
        p_kspace.set_ydata(k_traj[1,mask])
        p_kspace2.set_xdata(k_traj[0,mask])
        p_kspace2.set_ydata(k_traj[2,mask])

        p_adc.set_xdata(k_traj_adc[0,mask_adc])
        p_adc.set_ydata(k_traj_adc[1,mask_adc])
        p_adc2.set_xdata(k_traj_adc[0,mask_adc])
        p_adc2.set_ydata(k_traj_adc[2,mask_adc])

        c_ind = abs(t_ktraj-t_end).argmin()
        p_cursor.set_xdata([k_traj[0, c_ind]])
        p_cursor.set_ydata([k_traj[1, c_ind]])
        p_cursor2.set_xdata([k_traj[0, c_ind]])
        p_cursor2.set_ydata([k_traj[2, c_ind]])

    ax_rf.text(-0.07,0,'RF',transform=ax_rf.transAxes)
    ax_x.text(-0.07,0,'GX',transform=ax_x.transAxes)
    ax_y.text(-0.07,0,'GY',transform=ax_y.transAxes)
    ax_z.text(-0.07,0,'GZ',transform=ax_z.transAxes)

    ax_kspace.text(0.5,0,'Ky ↑',horizontalalignment='center',verticalalignment='top',transform=ax_kspace.transAxes)
    ax_kspace.text(0.0,0.5,'Kx → ',verticalalignment='center',horizontalalignment='right',transform=ax_kspace.transAxes)

    ax_kspace2.text(0.5,0,'Kz ↑',horizontalalignment='center',verticalalignment='top',transform=ax_kspace2.transAxes)
    ax_kspace2.text(0.0,0.5,'Kx → ',verticalalignment='center',horizontalalignment='right',transform=ax_kspace2.transAxes)

    plt.tight_layout()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=1000/fps)

    if show:
        plt.show()

    if save_filename is not None:
        ani.save(save_filename, fps=fps)

    if not show:
        plt.close()

    return ani


def plot_nd(rec, complex=False, vmin=None, vmax=None):
    rec = np.squeeze(rec)

    # Reduce dimensions to a maximum of 4
    if rec.ndim >= 5:
        rec = rec.reshape(-1, *rec.shape[-3:])

    # Plot anything between 2D, 3D, and 4D matrices
    if rec.ndim <= 2:
        im = rec
    elif rec.ndim == 3:
        if rec.shape[0] > 8:
            # Show up to 8 slices/contrasts
            rec = rec[np.floor(np.linspace(0,rec.shape[0]-1,8)).astype(int)]

        im = np.concatenate(rec, axis=1)
    elif rec.ndim == 4:
        # Show up to 8 contrasts
        if rec.shape[0] > 8:
            rec = rec[np.floor(np.linspace(0,rec.shape[0]-1,8)).astype(int)]
        # Show up to 8 slices
        if rec.shape[1] > 8:
            rec = rec[:,np.floor(np.linspace(0,rec.shape[1]-1,8)).astype(int)]
        
        # Create grid of images
        im = np.concatenate(np.concatenate(rec, axis=2), axis=0)


    plt.figure()
    
    # In case of 1D data, just show a line plot
    if im.ndim == 1:
        if not complex:
            plt.plot(abs(im))
        else:
            plt.plot(im.real)
            plt.plot(im.imag)
        return
    
    if not complex:
        plt.imshow(abs(im), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, origin='lower');
    else:
        if vmin is None:
            vmin = abs(im).min()
        if vmax is None:
            vmax = abs(im).max()
    
        phase_index = (np.angle(im) / np.pi + 1)/2
        colors = plt.get_cmap('hsv')(phase_index)
        intensity = np.clip((abs(im) - vmin) / (vmax - vmin), 0, 1)
        rgb = np.clip(np.stack((intensity * colors[...,0], intensity * colors[...,1], intensity * colors[...,2])),0,1)
    
        plt.imshow(rgb.transpose(1,2,0), origin='lower')
