import tflite_runtime.interpreter as tflite
# import tflite
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from skimage.restoration import inpaint
from plotly.tools import mpl_to_plotly
import plotly.express as px
import mne
from mne.inverse_sparse import mixed_norm, make_stc_from_dipoles
import matplotlib
import plotly.figure_factory as FF
import time
import os
from joblib import Parallel, delayed
import ast


def simulate_source(snr, n_sources, size, n, leadfield, pos, source_shape='flat'):
    ''' This function takes the simulation settings and simulates a pseudo-random sample in brain and sensor space.
    settings keys: ['snr', 'n_sources', 'size']
    '''
    # Check Inputs
    amps = (10, 100)
    if type(snr) == str:
        snr = str2num(snr)
    if type(n_sources) == str:
        n_sources = str2num(n_sources)
        if type(n_sources) == tuple or type(n_sources) == list:
            n_sources = [int(i) for i in n_sources]
        else:
            n_sources = int(n_sources)
    if type(size) == str:
        size = str2num(size)

    print(f'snr={snr}, n_sources={n_sources}, size={size}, n={n}')
    print(f'snr={type(snr)}, n_sources={type(n_sources)}, size={type(size)}, n={type(n)}')

    # Load basic Files
    # pth = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'assets\\modeling'))
    # ## Leadfield
    # with open(pth+'\\leadfield.pkl', 'rb') as f:
    #     leadfield = pkl.load(f)[0]
    # ## Positions
    # with open(pth+'\\pos.pkl', 'rb') as f:
    #     pos = pkl.load(f)[0]
    

    # Generate a source configuration based on settings
    y = np.zeros((n, leadfield.shape[1]))
    x_img = np.zeros((n, 7, 11))
    for s in range(n):
        if type(n_sources) == list or type(n_sources) == tuple:
            srange = np.arange(n_sources[0], n_sources[1]+1)
            n_sources_tmp = np.random.choice(srange)
        else:
            n_sources_tmp = n_sources

        src_centers = np.random.choice(np.arange(0, pos.shape[0]), n_sources_tmp,
                                    replace=False)
        if type(size) == list or type(size) == tuple:
            src_diams = (size[1]-size[0]) * np.random.random_sample(n_sources_tmp) + size[0]
        else:
            src_diams = np.repeat(size, n_sources_tmp)

        src_amps = (amps[1]-amps[0]) * np.random.random_sample(n_sources_tmp) + amps[0]   
        # Smoothing and amplitude assignment
        
        d = {}
        if source_shape == 'gaussian':
            print("hi")
            for i in range(src_centers.shape[0]):
                dists = np.sqrt(np.sum((pos - pos[src_centers[i], :])**2, axis=1))
                y[s, :] += gaussian(dists, 0, src_diams[i]/2) * src_amps[i]

        else:
            for i in range(src_centers.shape[0]):
                dists = np.sqrt(np.sum((pos - pos[src_centers[i], :])**2, axis=1))
                d[i] = np.where(dists<src_diams[i]/2)
                y[s, d[i]] = src_amps[i]
        
    # Noise
    if type(snr) == tuple or type(snr) == list: # pick noise in some range
        db_choice = (snr[1]-snr[0]) * np.random.random_sample(n) + snr[0]
    else:  # pick definite noise
        if n == 1:
            db_choice = [snr]
        else:
            db_choice = np.repeat(snr, n)

    x_noise = source_to_ximg(y, leadfield, n, db_choice)

    print('test')
    x_img = np.stack(Parallel(n_jobs = -1, backend = 'loky')(delayed(vec_to_sevelev_newlayout)(i) for i in x_noise))

    # x_img = np.stack([vec_to_sevelev_newlayout(i) for i in x_noise], axis=0)
        
    return np.squeeze(y), np.squeeze(x_img), db_choice

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def source_to_ximg(y, leadfield, n, db_choice):
    x_noise = np.zeros((n, leadfield.shape[0]))
    

    for s in range(n): # loop simulations
        # Project target
        x = np.sum(y[s, :] * leadfield, axis=1)
        # Add noise
        
        
        relnoise = (10**(db_choice[s] / 10))**-1
        ss = np.sum(x**2)
        sc = len(x)
        rms = np.sqrt(ss/sc)
        noise = np.random.randn(len(x)) * relnoise * rms
        x_noise[s,] = x + noise

        # CAR
        x_noise[s,] = x_noise[s,] - (np.sum(x_noise[s,]) / len(x_noise[s,]))
    return x_noise

def make_fig_objects(y, x_img, tris, pos):
    # Scale vectors
    x_img /= np.max(np.abs(x_img))
    y /= np.max(np.abs(y))

    fig_x = px.imshow(x_img)
    fig_y, _ = brain_plotly(y, tris, pos)

    
    return fig_y, fig_x

def load_model(pth):
    ''' Load optimized tensorflow lite model using only the tf interpreter instead of the whole 500mb framework''' 
    model = tflite.Interpreter(model_path= pth + '/tf_lite_model_optimized.tflite')
    model.allocate_tensors()

    return model

def predict_source(x, leadfield, model):
    ''' Predict source y from EEG input data x with model and calculate the forward projection x_img of y through the leadfield'''
    print('prediction func')
    # Predict
    if len(x.shape) == 2:
        x = np.expand_dims(np.expand_dims(x, axis=2), axis=0)

    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    x = np.array(x, dtype=np.float32)
    input_shape = input_details[0]['shape']
    model.set_tensor(input_details[0]['index'], x)
    model.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    y = np.squeeze(model.get_tensor(output_details[0]['index']))

    # without tensorflow lite:
    # y = np.squeeze(model.predict(x))
 
    # Create forward projection
    # Project target
    x = np.sum(y * leadfield, axis=1)
    # CAR
    x -= np.mean(x)
    x_img = vec_to_sevelev_newlayout(x)

    return y, x_img

def inverse_solution(x, SNR, evoked, fwd, leadfield, method):
    ''' Calculate some inverse solutions as implemented in MNE Python '''

    source = np.zeros((5124, ))
    n_tr = 100
    base_interval = (0, 50)
    sig_interval = (50, 75)
    n_elec = len(x)
    # CAR
    x -= np.mean(x)
    # Scale
    x /= np.max(np.abs(x))
    # Calculate RMS
    rms = np.sqrt(np.mean(x**2))

    snr_erp = np.clip(db_conv(SNR), 1e-10, None)  # convert SNR from dB to relative
    relnoise_st = (1 * np.sqrt(n_tr)) / snr_erp    # calculate single-trial SNR based on desired ERP-SNR given the number of trials (n_tr)
    # evoked = epochs.average()

    print(f'rms = {rms}\nnoise = {SNR}dB\nsnr_erp = {snr_erp}\nrelnoise_st = {relnoise_st}')

    # Noise covariance
    #---------------------------------------------------------------------#
    ''' Create real epoch with real noise, baseline correction and some timeline with active source '''
    # Create empty epochs structure
    
    info = mne.create_info(evoked.ch_names, 1000, ch_types='eeg')
    data = np.random.randn(n_tr, n_elec, sig_interval[1]) * relnoise_st * rms
    # CAR noise section
    for i in range(n_tr):
        for j in range(base_interval[0], base_interval[1]):
            data[i, :, j] -= np.mean(data[i, :, j])
    # noise + signal section
    for i in range(n_tr):
        for j in range(sig_interval[0], sig_interval[1]):
            data[i, :, j] += x
            data[i, :, j] -= np.mean(data[i, :, j])    
    
    epochs = mne.EpochsArray(data, info, events=None, tmin=0.0, event_id=None, reject=None, flat=None, reject_tmin=None, reject_tmax=None, baseline=None, proj=True, on_missing='error', metadata=None, selection=None, verbose=None)

    epochs.set_eeg_reference(ref_channels='average')
    # montage = mne.channels.read_montage(kind='standard_1020', ch_names=epochs.ch_names,
    #                                     transform=False)
    # montage = mne.make_standard_montage('standard_1020')

    epochs.set_montage('standard_1020')
    
    epochs.apply_baseline(baseline=(base_interval[0] / 1000, base_interval[1] / 1000))

    # breakpoint()
    noise_cov = mne.compute_covariance(epochs, tmin=base_interval[0] / 1000, tmax=base_interval[1] / 1000,
                            method='empirical')

    data_cov = mne.compute_covariance(epochs, tmin=sig_interval[0] / 1000, tmax=(sig_interval[1] - 1) / 1000,
                            method='empirical')
    
    evoked = epochs.average()

    for i in range(sig_interval[1]):
        evoked.data[:, i] -= np.mean(evoked.data[:, i])
    
    # evoked_no_ref, _ = mne.set_eeg_reference(evoked, [])
    
    evoked.set_eeg_reference(ref_channels='average', projection=True)
    lambda2 = 1. / SNR**2   # https://mne.tools/0.16/auto_tutorials/plot_mne_dspm_source_localization.html
    if method == 'lcmv':
        
        filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, weight_norm='nai')
        stc = mne.beamformer.apply_lcmv(evoked, filters)
        source = stc.data[:, -1]

    elif method == 'mxne':
        # raise NameError('No name error, but mxne is not properly implemented yet :)')
        inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov,
                                        depth=0.9, fixed=True,
                                        use_cps=True)

        stc_dspm = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=lambda2,
                        method='dSPM')

        # Compute MxNE inverse solution with dipole output
        dipoles, residual = mixed_norm(
            evoked, fwd, noise_cov, 55, loose=0.2, depth=0.9, maxit=3000,
            tol=1e-4, active_set_size=10, debias=True, weights=stc_dspm,
            weights_min=8., n_mxne_iter=10, return_residual=True,
            return_as_dipoles=True)
        
        stc = make_stc_from_dipoles(dipoles, fwd['src'])
        source = stc.data[:, -1]
        # source = dipoles.data[:, -1]

        print(f'mxne source = {source}, dipoles={dipoles}')
        print(f'mxne dipoles = {dipoles}')
        print(f'mxne source = {source}')
        # breakpoint()
    else:
        # minimum-norm-based solutions
        print(f'epochs.info: {epochs.info}')
        inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, fixed=True, verbose=False)
        stc = np.abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2=lambda2, method=method, verbose=False))
        source = stc.data[:, -1]
        # breakpoint()

    x = np.sum(source * leadfield, axis=1)
    x_img = vec_to_sevelev_newlayout(x)
    return source, x_img

def db_conv(db):
    # Converts dB to relative
    return (10**(db / 10))

def str2num(mystr):
    if len(mystr) == 1:
        return float(mystr)
    else:
        arr = []
        
        return ast.literal_eval(mystr)
        

def vec_to_sevelev_newlayout(x):
    ''' convert a vector consisting of 32 electrodes to a 7x11 matrix using 
    inpainting '''
    x = np.squeeze(x)
    w = 11
    h = 7
    elcpos = np.empty((h, w))
    elcpos[:] = np.nan
    elcpos[0, 4] = x[0]
    elcpos[1, 3] = x[1]
    elcpos[1, 2] = x[2]
    elcpos[2, 0] = x[3]
    elcpos[2, 2] = x[4]
    elcpos[2, 4] = x[5]
    elcpos[3, 3] = x[6]
    elcpos[3, 1] = x[7]
    elcpos[4, 0] = x[8]
    elcpos[4, 2] = x[9]
    elcpos[4, 4] = x[10]
    
    elcpos[5, 5] = x[11]
    elcpos[5, 3] = x[12]
    elcpos[5, 2] = x[13]
    elcpos[6, 4] = x[14]
    elcpos[6, 5] = x[15]
    elcpos[6, 6] = x[16]

    elcpos[5, 7] = x[17]
    elcpos[5, 8] = x[18]
    elcpos[4, 10] = x[19]
    elcpos[4, 8] = x[20]
    elcpos[4, 6] = x[21]
    elcpos[3, 5] = x[22]
    elcpos[3, 7] = x[23]
    elcpos[3, 9] = x[24]
    
    elcpos[2, 10] = x[25] # FT10
    elcpos[2, 8] = x[26]
    elcpos[2, 6] = x[27]
    elcpos[1, 7] = x[28]
    elcpos[1, 8] = x[29]
    elcpos[0, 6] = x[30]
    # elcpos[1, 5] = 5 Fz was reference
    # elcpos[6, 2] = 28 PO9 deleted
    # elcpos[6, 8] = 32 PO10 deleted

    mask = np.zeros((elcpos.shape))
    mask[np.isnan(elcpos)] = 1
        
    
    return inpaint.inpaint_biharmonic(elcpos, mask, multichannel=False)

def sevelev_to_vec_newlayout(x):
    # x = np.squeeze(x)
    if len(x.shape) == 2:
        # x_out = np.zeros((1, 31))
        # x = np.expand_dims(x, axis=0)
        x_out = x[[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0], [4, 3, 2, 0, 2, 4, 3, 1, 0, 2, 4, 5, 3, 2, 4, 5, 6, 7, 8, 10, 8, 6, 5, 7, 9, 10, 8, 6, 7, 8, 6]]
    else:
        x_out = np.zeros(shape=(x.shape[0], 31))
        for i in range(x.shape[0]):
            tmp = np.squeeze(x[i, :])
            # tmp = x[i, ]
            # print(tmp.shape)
            x_out[i,:] = tmp[[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0], [4, 3, 2, 0, 2, 4, 3, 1, 0, 2, 4, 5, 3, 2, 4, 5, 6, 7, 8, 10, 8, 6, 5, 7, 9, 10, 8, 6, 7, 8, 6]]
    
    return x_out

def addNoise(x, db):
    if len(x.shape) > 2:
        n_samples = x.shape[0]
    else:
        n_samples = 1
        x = np.expand_dims(x, axis=0)

    x_out = np.zeros_like(x)
    if not isinstance(db, int) and not isinstance(db, float):
        db_choice = (db[1] - db[0]) * np.random.random_sample(n_samples) + db[0]                
    else:
        db_choice = np.repeat(db, n_samples)

    for i in range(n_samples):
        relnoise = (10**(db_choice[i] / 10))**-1
        ss = np.sum(x[i]**2)
        sc = x.shape[1]
        rms = np.sqrt(ss/sc)
        noise = np.random.randn(x.shape[1]) * relnoise * rms
        x_out[i] = x[i] + noise
    
    
    return x_out, db_choice

def brain_plotly(y, tris, pos):
    ''' takes triangulated mesh, list of coordinates and a vector of brain activity and plots a plotly triangulated surface '''
    # pth = "C:\\Users\\Lukas\\Documents\\cd_dash\\assets\\modeling\\"
    ## Positions
    # with open(pth+'pos.pkl', 'rb') as f:
    #     pos = pkl.load(f)[0]

    ## Inverse operator, needed to get the triangle-information in the plotting
    # fname_inv = pth + 'inverse-inv.fif'
    # inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)
    # tris = inverse_operator['src'][0]['use_tris']
    

    # Concatenate tris so that it covers the whole brain
    tmp = tris + int(pos.shape[0]/2)
    new_tris = np.concatenate([tris, tmp], axis=0)
    # Calculate the true value for each triangle (which is the mean of the triangle's vertices)
    colors = []
    for tri in new_tris:
        positions = pos[tri, :]
        indices = []
        for j in positions:
            indices.append(np.where((pos == j).all(axis=1))[0][0])
        colors.append(np.mean(y[indices]))


    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    ## Plot
    fig1 = FF.create_trisurf(x=x, y=y, z=z,
                            simplices=new_tris,
                            title="Simulated brain activity",
                            color_func=colors,
                            aspectratio=dict(x=1, y=1, z=1),
                            )
    fig1['layout']['height'] = 650
    fig1['layout']['width'] = 720
    return fig1, colors


def split_data(x, y, key):
    ''' split input and target data based on key'''
    # shuffle first:
    idx = np.arange(0, y.shape[0])
    np.random.shuffle(idx)
    x = x[idx,]
    y = y[idx,]

    tr_range = np.arange(0, np.floor(x.shape[0]*key[0]), dtype='int16')

    if len(key) == 2:
        val_range = np.arange(np.floor(x.shape[0]*key[0]), np.floor(x.shape[0]*key[1]), dtype='int16')
        tst_range = np.arange(np.floor(x.shape[0]*key[1]), x.shape[0], dtype='int16')

        return x[tr_range], x[val_range], x[tst_range], y[tr_range], y[val_range], y[tst_range]

    elif len(key) == 1:
        val_range = np.arange(np.floor(x.shape[0]*key[0]), x.shape[0], dtype='int16')
        
        return x[tr_range], x[val_range], y[tr_range], y[val_range]

