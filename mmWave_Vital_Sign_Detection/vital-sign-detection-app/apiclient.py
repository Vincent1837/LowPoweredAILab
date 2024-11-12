# -*- coding: utf-8 -*-
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Tuesday, June 21, 2022                                             #
# Description: Infineon FMCW RADAR System for Vital Sign Detection         #
# Copyright 2021. All Rights Reserved.                                     #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
#==========================================================================#

import os
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import *
from sklearn.cluster import DBSCAN
from collections import Counter
import pygame

pygame.mixer.init()
sys.path.append('.')

from utils.configure import Configure
from utils.logger import get_logger
from utils.checks import folder_check

from mmwave_processing.ifxRadarSDK import *
from mmwave_processing.fft_spectrum import *
import mmwave_processing.dsp as dsp
from mmwave_processing.dsp.utils import Window, windowing
from mmwave_processing.dsp.compensation import exponential_smoothing

def on_press(event):
    global exit
    
    print('press key: ', event.key)
    sys.stdout.flush()
    if event.key == 'q':
        plt.close('all')
        exit = True
        
def main():
    global exit
    
    # read FMCW radar configuration
    configs = Configure()
    
    # set radar metric
    metric = {
        'sample_rate_Hz':           configs.sample_rate_Hz,
        'range_resolution_m':       configs.range_resolution_m,
        'max_range_m':              configs.max_range_m,
        'max_speed_m_s':            configs.max_speed_m_s,
        'speed_resolution_m_s':     configs.speed_resolution_m_s,
        'frame_repetition_time_s':  configs.frame_repetition_time_s,
        'center_frequency_Hz':      configs.center_frequency_Hz,
        'rx_mask':                  configs.rx_mask,
        'tx_mask':                  configs.tx_mask,
        'tx_power_level':           configs.tx_power_level,
        'if_gain_dB':               configs.if_gain_dB
    }

    # Initiate a radar device
    device = Device()

    # radar configuration from metric
    cfg = device.metrics_to_config(**metric)

    # set the configuration of the radar device
    device.set_config(**cfg)
    
    # get radar configuration
    config = device.get_config()
    configs.start_frequency_Hz = config['start_frequency_Hz']
    configs.end_frequency_Hz = config['end_frequency_Hz']
    configs.num_chirps_per_frame = config['num_chirps_per_frame']
    configs.num_samples_per_chirp = config['num_samples_per_chirp']
    configs.chirp_repetition_time_s = config['chirp_repetition_time_s']
    configs.mimo_mode = config['mimo_mode']
    configs.frame_rate = 1 / configs.frame_repetition_time_s

    # RADAR performance metrics
    configs.d_res, configs.d_max = dsp.range_resolution_max_distance(configs)
    configs.V_res, configs.V_max, configs.wave_length = dsp.doppler_resolution_max_speed(configs)
    #doppler_res = -2*configs.V_res/configs.wave_length                  # Doppler Shift resolution

    # show and record radar configuration
    folder_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    configs.record_radar_parameters(logger)
   
    # Vital Sign Detection parameters
    numRangeBins = configs.RANGE_FFT
    numDopplerBins = configs.DOPPLER_FFT
    fs_slow = configs.frame_rate                                        # sample rate for vital sign
    Nyquist_slow = fs_slow/2                                            # Nyquist frequency
    T_slow = 1 / fs_slow                                                # sample time for vital sign, slow time
    velocities = np.arange(configs.DOPPLER_FFT/2) * configs.V_res
    vibrate_res = fs_slow / configs.VIBRATION_FFT
    
    # Axis scale for plot
    r_axis = np.arange(0, configs.RANGE_FFT/2, 4)                       # range axis, meter
    r_axis_cm = np.round(r_axis * configs.d_res * 100)                  # range axis, centi-meter
    x_axis = np.arange(0, configs.DOPPLER_FFT, 4)                       # Doppler axis, m/s
    x1 = np.arange(0, configs.DOPPLER_FFT/2, 4)                         # Doppler axis, -V_max <= v <= V_max
    x2 = -x1[::-1]
    x3 = np.array(list(x2)[:-1] + list(x1))
    x_axis_speed = np.round(x3 * configs.V_res * 100)                   # Doppler axis, cm/s
    
    # time constant
    t0 = 128    # 12.8 sec (128 frames)
    t1 = 32     # 3.2 sec (32 frames)
    t_axis = np.arange(0, t0, 8)                                        # slow time axis for vital waveform plot
    t_axis_s = np.round(t_axis * configs.frame_repetition_time_s)
    v_axis = np.arange(0, configs.VIBRATION_FFT, 4)                     # frequency axis for vibration spectrum
    v_axis_hz = np.round(v_axis * vibrate_res * 60)                     # unit: beats / per minute
    
    BINS_PROCESSED = int(configs.RANGE_FFT / 2)                         # range bins
        
    #--------------------------------#
    # Bandpass filter for Heart Rate #
    #--------------------------------#
    
    idx_hr_low = int(configs.HR_LOW // vibrate_res) + 1                 # lower bound of heart rate detection
    idx_hr_high = int(configs.HR_HIGH // vibrate_res)                   # upper bound of heart rate detection
    #print(f'(HR_LOW, HR_HIGH) = ({idx_hr_low}, {idx_hr_high})')        # (41, 102)
    N, Wn = buttord([configs.HR_LOW/Nyquist_slow, configs.HR_HIGH/Nyquist_slow], 
                           [configs.HR_LOW/1.6/Nyquist_slow, configs.HR_HIGH*1.6/Nyquist_slow], 3, configs.attenuation)
    #print(f'Filter order: {N}, Cutoff Frequency: {Wn}')
    sos_hr = butter(N, Wn, btype='bandpass', output='sos')
    
    if configs.plot_hr_filter:
        w, h = sosfreqz(sos_hr)
        plt.subplot(2, 1, 1)
        db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
        plt.semilogx(w/np.pi, db)
        plt.grid(True)
        plt.ylabel('Gain [dB]')
        plt.title('Frequency Response for Heart Rate Filter')

        plt.subplot(2, 1, 2)
        plt.semilogx(w/np.pi, np.angle(h))
        plt.grid(True)
        plt.title('Phase Response')
        plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.ylabel('Phase [rad]')
        plt.xlabel(r'Normalized frequency [x$\pi$ rad / sample] (1 = Nyquist)')
        plt.show()

    #---------------------------------#
    # Bandpass filter for Breath Rate #
    #---------------------------------#
    
    idx_br_low = int(configs.BR_LOW // (vibrate_res)) + 1               # lower bound of breath rate detection
    idx_br_high = int(configs.BR_HIGH // (vibrate_res))                 # upper bound of breath rate detection
    #print(f'(BR_LOW, BR_HIGH) = ({idx_br_low}, {idx_br_high})')        # (6, 25)
    N, Wn = buttord([configs.BR_LOW/Nyquist_slow, configs.BR_HIGH/Nyquist_slow], 
                           [configs.BR_LOW/1.6/Nyquist_slow, configs.BR_HIGH*1.6/Nyquist_slow], 3, configs.attenuation)
    #print(f'Filter order: {N}, Cutoff Frequency: {Wn}')
    sos_br = butter(N, Wn, btype='bandpass', output='sos')
    
    if configs.plot_br_filter:
        w, h = sosfreqz(sos_br)
        plt.subplot(2, 1, 1)
        db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
        plt.semilogx(w/np.pi, db)
        plt.grid(True)
        plt.ylabel('Gain [dB]')
        plt.title('Frequency Response for Breath Rate Filter')

        plt.subplot(2, 1, 2)
        plt.semilogx(w/np.pi, np.angle(h))
        plt.grid(True)
        plt.title('Phase Response')
        plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.ylabel('Phase [rad]')
        plt.xlabel(r'Normalized frequency [x$\pi$ rad / sample] (1 = Nyquist)')
        plt.show()

    # initial condition of bandpass filter, max human number: 1, number of RX antenna: 1
    max_num_human = 1
    zi_hr = np.zeros([max_num_human, configs.NUM_RX, sosfilt_zi(sos_hr).shape[0], sosfilt_zi(sos_hr).shape[1]])
    zi_br = np.zeros([max_num_human, configs.NUM_RX, sosfilt_zi(sos_br).shape[0], sosfilt_zi(sos_br).shape[1]])
    for i in range(max_num_human):
        for j in range(configs.NUM_RX):
            zi_hr[i, j, :, :] = sosfilt_zi(sos_hr)
            zi_br[i, j, :, :] = sosfilt_zi(sos_br)
    zo_hr = np.zeros_like(zi_hr)
    zo_br = np.zeros_like(zi_br)

    # initial condition for vital rate detection    
    breath_rate = np.zeros(configs.NUM_RX)
    heart_rate = np.zeros(configs.NUM_RX)

    # initial condition of range bin selection
    observed_frame = t1
    max_range_bin = np.zeros((configs.num_chirps_per_frame, configs.NUM_RX*configs.NUM_TX, observed_frame))

    # initial condition of phase processing
    if configs.plotRangeSlow or configs.plotRangeUnwrap:
        phase_tmp = np.zeros([t1, configs.NUM_RX, BINS_PROCESSED])
        radar_cube_tmp = np.zeros([t1, configs.NUM_RX, BINS_PROCESSED], dtype=complex)
        phase_extract = np.zeros([t0, configs.NUM_RX, BINS_PROCESSED])
        phase_unwrap = np.zeros([t0, configs.NUM_RX, BINS_PROCESSED])
        phase_unwrap_pre = np.zeros([configs.NUM_RX, BINS_PROCESSED])
        radar_cube_slow = np.zeros([t0, configs.NUM_RX, BINS_PROCESSED], dtype=complex)
    else:
        phase_tmp = np.zeros([t1, configs.NUM_RX])
        radar_cube_tmp = np.zeros([t1, configs.NUM_RX], dtype=complex)
        phase_extract = np.zeros([t0, configs.NUM_RX])
        phase_unwrap = np.zeros([t0, configs.NUM_RX])
        phase_unwrap_pre = np.zeros(configs.NUM_RX)
        radar_cube_slow = np.zeros([t0, configs.NUM_RX], dtype=complex)
    
    phase_diff = np.zeros([t0, configs.NUM_RX])
    phase_denoised = np.zeros([t0, configs.NUM_RX])
    phase_diff_pre = np.zeros(configs.NUM_RX)
    phase_denoised_pre = np.zeros([2, configs.NUM_RX])

    # max human number: 1
    phase_hr = np.zeros([t0, configs.NUM_RX])
    phase_br = np.zeros([t0, configs.NUM_RX])
    phase_hr_filt = np.zeros([t1, configs.NUM_RX])
    phase_br_filt = np.zeros([t1, configs.NUM_RX])
    phase_hr_plt = np.zeros([max_num_human, t0])
    phase_br_plt = np.zeros([max_num_human, t0])
    Breath_Rate = np.zeros(max_num_human)
    Heart_Rate = np.zeros(max_num_human)
    Breath_pre = np.ones(max_num_human)*14          # initialized to 14
    Heart_pre = np.ones(max_num_human)*72           # initialized to 72
    Human_distance = np.zeros(max_num_human)
    alpha = 0.25                                    # exponential smoothing of heart rate and breath rate
    denoise_thre = 0.1                              # impulsive noise removal
    doppler_sig_strength = np.zeros([t1, BINS_PROCESSED, configs.DOPPLER_FFT])
    Vibration_hr = np.zeros([configs.VIBRATION_FFT, configs.NUM_RX], dtype=complex)
    Vibration_br = np.zeros([configs.VIBRATION_FFT, configs.NUM_RX], dtype=complex)
    num_peaks = 8       # Top num_peaks peaks for dbscan clustering
    #dbscan_in = np.zeros([(t0/t1)*configs.NUM_RX*num_peaks, 2])
    #for i in range((t0/t1)*configs.NUM_RX*num_peaks):
    #    dbscan_in[i, 0] = 72
    dbscan_in = np.zeros([int(t0/t1)*num_peaks, 2])
    for i in range(int(t0/t1)*num_peaks):
        dbscan_in[i, 0] = 72
        
    # initial condition of human detection
    avg_sig_strength = np.zeros([t1, BINS_PROCESSED])
    
    # setup plot figure
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.canvas.mpl_connect('key_press_event', on_press)

    # initial condition for Exponential Smoothing
    init = True         # for exponential smoothing of static clutter removal
    pre_val = None      # for exponential smoothing of static clutter removal
    
    num_frame = 0
    t = 0
    ptr = 0
    exit = False
    while not exit:
    #while True:
        #start_count = time.process_time()
        #start_count = time.perf_counter()
        
        #------------------------------------------#
        # Get one frame from the FMCW radar module #
        #------------------------------------------#
        
        dataCube = device.get_next_frame()                  # (num_rx, num_chirps, num_samples) = (1, 32, 128)
        #print('Shape of datacube: ', dataCube.shape)

        #===================================#
        # RADAR Signal Processing per Frame #
        #===================================#
        
        #---------------------#
        # 1. Range Processing #
        #---------------------#
        
        # Remove DC bias from ADC raw data before Range FFT
        for i in range(configs.NUM_RX):
            dataCube[i, :, :] = remove_DC_bias(dataCube[i, :, :])

        dataCube = np.transpose(dataCube, (1, 0, 2))        # (num_chirps, num_rx, num_samples) = (32, 1, 128)
        dataCube = dsp.range_processing(dataCube, n_fft=configs.RANGE_FFT, window_type_1d=Window.BLACKMAN)

        radar_cube = dataCube[:, :, :BINS_PROCESSED]        # (num_chirps, num_rx, range bins) = (32, 1, 64)
        #print('Shape of RADAR Cube: ', radar_cube.shape)
        
        # --- Range Plot --- #
        if configs.Range_plot:
            plt.imshow(np.abs(radar_cube).sum(axis=1).T)
            plt.yticks(r_axis, r_axis_cm)
            plt.ylabel('Range (cm)')
            plt.title('Range Plot')
            plt.pause(0.05)
            plt.clf()
        
        #---------------------------#
        # 2. Static Clutter Removal #
        #---------------------------#
        
        # exponential smoothing for static clutter removal
        radar_cube_smoothed, pre_val = exponential_smoothing(radar_cube, pre_val, init, alpha=configs.alpha)
        init = False

        # static clutter removal: method 1, exponential smoothing
        radar_cube_clutter_removed = radar_cube - radar_cube_smoothed   # (num_chirps, num_rx, range bins) = (32, 1, 64)
        
        # static clutter removal: method 2, naive clutter removal
        #mean = radar_cube.mean(0)
        #radar_cube_clutter_removed = radar_cube - mean

        # --- Range Plot with static clutter removal --- #
        if configs.Range_plot_clutter_removed:
            plt.imshow(np.abs(radar_cube_clutter_removed).sum(axis=1).T)
            plt.yticks(r_axis, r_axis_cm)
            plt.ylabel('Range (cm)')
            plt.title('Range Plot with Static Clutter Removal')
            plt.pause(0.05)
            plt.clf()
        
        #------------------------------------------------------#
        # 3. Human detection                                   #
        #    Detect the Number and Position of Humans          #
        #    CA-CFAR (Cell Averaged Constant False Alarm Rate) #
        #    Doppler Processing                                #
        #------------------------------------------------------#
        
        # --- Doppler Processing --- #
        doppler_cube, _ = dsp.doppler_processing(radar_cube_clutter_removed, num_tx_antennas=configs.NUM_TX, 
                                                 n_fft=configs.DOPPLER_FFT, window_type_2d=Window.HAMMING)
        #print(f"RADAR Cube after Doppler FFT: {doppler_cube.shape}")       # (range bins, Doppler FFT) = (64, 32)
        doppler_sig_strength[num_frame%t1, :, :] = doppler_cube             # (t1, range bins, Doppler FFT) = (t1, 64, 32)
        
        # --- Range-Doppler Plot --- #
        if configs.RangeDopp_plot:
            det_matrix_vis = np.fft.fftshift(doppler_cube, axes=1)
            plt.imshow(det_matrix_vis / det_matrix_vis.max())
            plt.title("Range-Doppler Plot of FMCW Radar")
            plt.xticks(x_axis[1:], x_axis_speed)
            plt.yticks(r_axis, r_axis_cm)
            plt.xlabel('Speed (cm/s)')
            plt.ylabel('Ranges (cm)')
            plt.pause(0.05)
            plt.clf()
        
        # average signal strength per frame, (t1, range bins) = (32, 64)
        avg_sig_strength[num_frame%t1] = np.abs(radar_cube_clutter_removed).sum(axis=1)\
                                                                    .sum(axis=0)/radar_cube_clutter_removed.shape[0]
        
        if num_frame % t1 == 0:
            # average signal strength in the past 3.2 seconds
            avg_sig = avg_sig_strength.sum(axis=0)/avg_sig_strength.shape[0]        # (range bins,) = (64,)
            #print(avg_sig[:10])
            
            # --- cfar in range bin ---
            human_detected = np.apply_along_axis(func1d=dsp.ca, axis=0, arr=np.log2(avg_sig), l_bound=1.5, guard_len=4, noise_len=16)
            
            # --- Max Range-Doppler Signal ---
            avg_doppler_sig = doppler_sig_strength.sum(axis=0)/doppler_sig_strength.shape[0]    # (range bins, Doppler FFT) = (64, 32)
            max_idx = np.unravel_index(np.argmax(avg_doppler_sig, axis=None), avg_doppler_sig.shape)
            #print(f'max index:\t{max_idx}')

            #if human_detected.any() == True:
            if human_detected.any() == True and max_idx[1] == 0:
                present_flag = True
                '''
                # number of humans
                start = []
                end = []
                if (human_detected[4] == True):
                    start.append(4)
                for i in range(4, 42):
                    if (human_detected[i] == False) and (human_detected[i+1] == True):
                        start.append(i+1)
                    elif (human_detected[i] == True) and (human_detected[i+1] == False):
                        end.append(i+1)
                    else:
                        continue
                if (human_detected[42] == True):
                    end.append(42)

                # number and position of humans
                position = []
                start_idx = []
                end_idx = []
                for i in range(len(start)):
                    if end[i] - start[i] >= 3:
                        position.append(np.argmax(avg_sig[start[i]:end[i]]) + start[i])
                        start_idx.append(start[i])
                        end_idx.append(end[i])
                human_num = len(position)
                '''
                position = []
                human_num = 1
                pos = max_idx[0]        # true position of human
                position.append(pos)

            else:
                present_flag = False
                human_num = 0
                position = []

                Breath_Rate = np.zeros(max_num_human)
                Heart_Rate = np.zeros(max_num_human)
                Breath_pre = np.ones(max_num_human)*14
                Heart_pre = np.ones(max_num_human)*72
                Human_distance = np.zeros(max_num_human)
                print(f'Detected Breath Rate:\tN/A')
                print(f'Detected Hreath Rate:\tN/A')
                phase_br_plt[:, :-t1] = phase_br_plt[:, t1:]
                phase_hr_plt[:, :-t1] = phase_hr_plt[:, t1:]
                phase_br_plt[:, -t1:] = np.zeros([max_num_human, t1])
                phase_hr_plt[:, -t1:] = np.zeros([max_num_human, t1])

                # --- vital rate and waveform display ---
                if configs.plotWaveform:
                    plt.plot(phase_br_plt[0, :]*configs.wave_length*1e3/(4*np.pi), label=f'Person {0}, Breath Rate: {Breath_Rate[0]:.2f}')
                    plt.plot(phase_hr_plt[0, :]*configs.wave_length*1e3/(4*np.pi), label=f'                Heart Rate:   {Heart_Rate[0]:.2f}')

                    plt.xlim(0, 12.8)
                    #plt.ylim(-np.pi*2, np.pi*2)
                    plt.ylim(-4, 4)
                    plt.xticks(t_axis, t_axis_s)
                    plt.xlabel('Time (sec)')
                    #plt.ylabel('Phase (rad)')
                    plt.ylabel('Amplitude (mm)')
                    plt.title('Vital Waveform')
                    plt.legend(loc = 'upper right')
                    plt.pause(0.05)
                    plt.clf()

        #-----------------------------------------------------------------------------#
        # 4. Phase Processing for vital sign detection                                #
        #    Phase Extraction, Phase Unwrapping, Bandpass Filtering and Vibration FFT #
        #-----------------------------------------------------------------------------#
        
        #present_flag = True
        #human_num = 1
        #position.append(20)
        if present_flag:
            for k in range(human_num):

                #--------------------------#
                # --- Phase Extraction --- # radarCube.shape = (num_chirps, num_rx, range bins) = (32, 1, 64)
                #--------------------------#
                
                if configs.plotRangeSlow or configs.plotRangeUnwrap:
                    phase_tmp[num_frame%t1, :, :] = np.arctan2(radar_cube[0, :, :].imag, radar_cube[0, :, :].real) # (t1, num_rx, range bins) = (32, 1, 64)
                    #phase_tmp[num_frame%t1, :, :] = np.arctan2(radar_cube[:, :, :].sum(axis=0).imag, radar_cube[:, :, :].sum(axis=0).real)

                    radar_cube_tmp[num_frame%t1, :, :] = radar_cube_clutter_removed[0, :, :]
                    #radar_cube_tmp[num_frame%t1, :, :] = radar_cube_clutter_removed[:, :, :].sum(axis=0)
                    
                else:
                    phase_tmp[num_frame%t1, :] = np.arctan2(radar_cube[0, :, position[k]].imag, radar_cube[0, :, position[k]].real) # (t1, num_rx) = (32, 1)
                    #phase_tmp[num_frame%t1, :] = np.arctan2(radar_cube[:, :, position[k]].sum(axis=0).imag, radar_cube[:, :, position[k]].sum(axis=0).real)
                    
                if num_frame % t1 == 0:

                    if configs.plotRangeSlow or configs.plotRangeUnwrap:
                        phase_extract[:-t1, :, :] = phase_extract[t1:, :, :]        # (t0, num_rx, range bins) = (128, 1, 64)
                        phase_extract[-t1:, :, :] = phase_tmp
                    
                        if configs.plotRangeSlow:
                            radar_cube_slow[:-t1, :, :] = radar_cube_slow[t1:, :, :]    # (t0, num_rx, range bins) = (128, 1, 64)
                            radar_cube_slow[-t1:, :, :] = radar_cube_tmp
                            
                            plt.imshow(np.abs(radar_cube_slow).sum(axis=1))
                            plt.xticks(r_axis, r_axis_cm)
                            plt.yticks(t_axis, t_axis_s)
                            plt.xlabel('Range (cm)')
                            plt.ylabel('Slow-time (sec)')
                            plt.title('Range Slow-time matrix')
                            plt.pause(0.05)
                            plt.clf()
                    else:
                        phase_extract[:-t1, :] = phase_extract[t1:, :]              # (t0, num_rx) = (128, 1)
                        phase_extract[-t1:, :] = phase_tmp

                    #--------------------------#
                    # --- Phase Unwrapping --- #
                    #--------------------------#
                    if configs.plotRangeSlow or configs.plotRangeUnwrap:
                        phase_unwrap[:, :, :] = phase_extract[:, :, :]              # (t0, RX Ant, range bins) = (128, 1, 64)
                        for j in range(configs.NUM_RX):
                            for l in range(BINS_PROCESSED):
                                if (phase_unwrap[0, j, l] - phase_unwrap_pre[j, l]) > np.pi:
                                    phase_unwrap[0, j, l] = phase_unwrap[0, j, l] - 2*np.pi
                                elif (phase_unwrap[0, j, l] - phase_unwrap_pre[j, l]) < -np.pi:
                                    phase_unwrap[0, j, l] = phase_unwrap[0, j, l] + 2*np.pi

                                for i in range(1, t0):
                                    if (phase_unwrap[i, j, l] - phase_unwrap[i-1, j, l]) > np.pi:
                                        phase_unwrap[i, j, l] = phase_unwrap[i, j, l] - 2*np.pi
                                    elif (phase_unwrap[i, j, l] - phase_unwrap[i-1, j, l]) < -np.pi:
                                        phase_unwrap[i, j, l] = phase_unwrap[i, j, l] + 2*np.pi

                                phase_unwrap_pre[j, l] = phase_unwrap[-1, j, l]
                            
                        if configs.plotRangeUnwrap:
                            plt.imshow(phase_unwrap[:,0,:])
                            plt.xticks(r_axis, r_axis_cm)
                            plt.yticks(t_axis, t_axis_s)
                            #plt.xlabel('Range (cm)')
                            plt.ylabel('Slow-time (sec)')
                            plt.title('Range Slow-time matrix')
                            plt.pause(0.05)
                            plt.clf()
                    else:
                        phase_unwrap[:, :] = phase_extract[:, :]                  # (t0, RX Ant) = (128, 1)
                        for j in range(configs.NUM_RX):
                            if (phase_unwrap[0, j] - phase_unwrap_pre[j]) > np.pi:
                                phase_unwrap[0, j] = phase_unwrap[0, j] - 2*np.pi
                            elif (phase_unwrap[0, j] - phase_unwrap_pre[j]) < -np.pi:
                                phase_unwrap[0, j] = phase_unwrap[0, j] + 2*np.pi

                            for i in range(1, t0):
                                if (phase_unwrap[i, j] - phase_unwrap[i-1, j]) > np.pi:
                                    phase_unwrap[i, j] = phase_unwrap[i, j] - 2*np.pi
                                elif (phase_unwrap[i, j] - phase_unwrap[i-1, j]) < -np.pi:
                                    phase_unwrap[i, j] = phase_unwrap[i, j] + 2*np.pi

                            phase_unwrap_pre[j] = phase_unwrap[-1, j]
                            
                        if configs.plotUnwrap:
                            plt.plot(phase_unwrap[:,0])
                            #plt.plot(phase_denoised)
                            plt.ylim(-3.14, 3.14)
                            plt.xticks(t_axis, t_axis_s)
                            #plt.xlabel('Slow-time (sec)')
                            plt.ylabel('Phase (rad)')
                            plt.title('Unwrapped Phase')
                            plt.pause(0.05)
                            plt.clf()

                    #--------------------------#
                    # --- Phase Difference --- #
                    #--------------------------#
                    
                    if configs.plotRangeSlow or configs.plotRangeUnwrap:
                        phase_diff[0, :] = phase_unwrap[0, :, position[k]] - phase_diff_pre[:]
                        phase_diff[1:, :] = phase_unwrap[1:, :, position[k]] - phase_unwrap[:-1, :, position[k]]  # (t0, RX Ant) = (128, 1)
                        phase_diff_pre[:] = phase_unwrap[-1, :, position[k]]
                    else:
                        phase_diff[0, :] = phase_unwrap[0, :] - phase_diff_pre[:]
                        phase_diff[1:, :] = phase_unwrap[1:, :] - phase_unwrap[:-1, :]  # (t0, RX Ant) = (128, 1)
                        phase_diff_pre[:] = phase_unwrap[-1, :]

                    #---------------------------------#
                    # --- Impulsive Noise Removal --- #                               (t0, RX Ant) = (128, 1)
                    #---------------------------------#
                    
                    for j in range(configs.NUM_RX):
                        if (phase_denoised_pre[1, j] - phase_denoised_pre[0, j]) > denoise_thre and (phase_denoised_pre[1, j] - phase_diff[0, j]) > denoise_thre:
                            phase_denoised[0, j] = (phase_diff[0, j] + phase_denoised_pre[0, j]) / 2
                        else:
                            phase_denoised[0, j] = phase_denoised_pre[1, j]

                        if (phase_diff[0, j] - phase_denoised_pre[1, j]) > denoise_thre and (phase_diff[0, j] - phase_diff[1, j]) > denoise_thre:
                            phase_denoised[1, j] = (phase_diff[1, j] + phase_denoised_pre[1, j]) / 2
                        else:
                            phase_denoised[1, j] = phase_diff[0, j]

                        for i in range(1, t0-1):
                            if (phase_diff[i, j] - phase_diff[i+1, j]) > denoise_thre and (phase_diff[i, j] - phase_diff[i-1, j]) > denoise_thre:
                                phase_denoised[i+1, j] = (phase_diff[i+1, j] + phase_denoised[i-1, j]) / 2
                            else:
                                phase_denoised[i+1, j] = phase_diff[i, j]

                    phase_denoised_pre[:, :] = phase_diff[-2:, :]

                    if configs.plotDenoise:
                        plt.plot(phase_denoised[:,0])
                        plt.ylim(-3.14, 3.14)
                        plt.xticks(t_axis, t_axis_s)
                        #plt.xlabel('Slow-time (sec)')
                        plt.ylabel('Phase (rad)')
                        plt.title('Phase De-noised')
                        plt.pause(0.05)
                        plt.clf()

                    #----------------------------#
                    # --- Bandpass Filtering --- #      (t1, num_rx) = (32, 1)
                    #----------------------------#
                    
                    for j in range(configs.NUM_RX):
                        if configs.plotRangeSlow or configs.plotRangeUnwrap:
                            #phase_hr_filt[:, j], zo_hr[k, j, :, :] = sosfilt(sos_hr, phase_unwrap[:t1, j, position[k]], axis=0, zi=zi_hr[k, j, :, :])
                            phase_br_filt[:, j], zo_br[k, j, :, :] = sosfilt(sos_br, phase_unwrap[:t1, j, position[k]], axis=0, zi=zi_br[k, j, :, :])
                        else:
                            #phase_hr_filt[:, j], zo_hr[k, j, :, :] = sosfilt(sos_hr, phase_unwrap[:t1, j], axis=0, zi=zi_hr[k, j, :, :])
                            phase_br_filt[:, j], zo_br[k, j, :, :] = sosfilt(sos_br, phase_unwrap[:t1, j], axis=0, zi=zi_br[k, j, :, :])

                        phase_hr_filt[:, j], zo_hr[k, j, :, :] = sosfilt(sos_hr, phase_denoised[:t1, j], axis=0, zi=zi_hr[k, j, :, :])
                        #phase_br_filt[:, j], zo_br[k, j, :, :] = sosfilt(sos_br, phase_denoised[:t1, j], axis=0, zi=zi_br[k, j, :, :])

                        zi_hr[k, j, :, :] = zo_hr[k, j, :, :]
                        zi_br[k, j, :, :] = zo_br[k, j, :, :]

                    phase_hr[:-t1, :] = phase_hr[t1:, :]    # (t0, num_rx) = (128, 1)
                    phase_hr[-t1:, :] = phase_hr_filt
                    phase_br[:-t1, :] = phase_br[t1:, :]
                    phase_br[-t1:, :] = phase_br_filt
                    
                    phase_hr_plt[k, :-t1] = phase_hr_plt[k, t1:]
                    phase_br_plt[k, :-t1] = phase_br_plt[k, t1:]
                    phase_hr_plt[k, -t1:] = phase_hr_filt[:, 0]
                    phase_br_plt[k, -t1:] = phase_br_filt[:, 0]

                    #------------------------------#
                    # --- Vital sign detection --- #
                    #------------------------------#
                    
                    # zero padding
                    if phase_hr.shape[0] < configs.VIBRATION_FFT:
                        zero_pad = np.zeros([configs.VIBRATION_FFT-phase_hr.shape[0], configs.NUM_RX])
                        phase_hr = np.vstack((phase_hr, zero_pad))
                        phase_br = np.vstack((phase_br, zero_pad))
                    
                    # windowing
                    fft_in_hr = windowing(phase_hr.T, window_type=Window.HAMMING, axis=1)
                    fft_in_br = windowing(phase_br.T, window_type=Window.HAMMING, axis=1)
                    #fft_in_hr = windowing(phase_hr.sum(axis=1)/configs.NUM_RX, window_type=Window.HAMMING, axis=0)
                    #fft_in_br = windowing(phase_br.sum(axis=1)/configs.NUM_RX, window_type=Window.HAMMING, axis=0)
                    
                    # --- Vibration FFT --- #    (VIBRATION_FFT, num_rx) = (512, 1)
                    Vibration_hr[:, :] = np.fft.fft(fft_in_hr.T, n=configs.VIBRATION_FFT, axis=0)
                    Vibration_br[:, :] = np.fft.fft(fft_in_br.T, n=configs.VIBRATION_FFT, axis=0)
                    #Vibration_hr = np.fft.fft(fft_in_hr, n=configs.VIBRATION_FFT, axis=0)
                    #Vibration_br = np.fft.fft(fft_in_br, n=configs.VIBRATION_FFT, axis=0)
                    
                    # average over all Rx Ant
                    HR_spectrum = np.abs(Vibration_hr[:int(configs.VIBRATION_FFT/2), :].sum(axis=1)/configs.NUM_RX)
                    #HR_spectrum = np.abs(Vibration_hr[:int(configs.VIBRATION_FFT/2)])       # (256,)
                    BR_spectrum = np.abs(Vibration_br[:int(configs.VIBRATION_FFT/2), :].sum(axis=1)/configs.NUM_RX)
                    #BR_spectrum = np.abs(Vibration_br[:int(configs.VIBRATION_FFT/2)])       # (256,)
                    #print(f'HR shape:\t{HR_spectrum.shape}')
                    
                    if configs.plotVibrateFFT:
                        
                        plt.plot(HR_spectrum, label='Heart')
                        plt.vlines(configs.VIBRATION_FFT*72/600, 0, 30)
                        plt.xticks(v_axis, v_axis_hz)
                        plt.xlim(idx_hr_low, idx_hr_high+1)
                        plt.ylim(0, 30)
                        plt.xlabel('Rate (beats/min)')
                        plt.ylabel('Amplitude')
                        plt.title('Vibration Spectrum')
                        plt.legend(loc='upper right')
                        '''
                        plt.plot(BR_spectrum, label='Breath')
                        plt.xticks(v_axis, v_axis_hz)
                        plt.xlim(idx_br_low, idx_br_high+1)
                        plt.ylim(0, 30)
                        plt.xlabel('Rate (beats/min)')
                        plt.ylabel('Amplitude')
                        plt.title('Vibration Spectrum')
                        plt.legend(loc='upper right')
                        '''
                        plt.pause(0.05)
                        plt.clf()

                    #---------------------------------------#    
                    #     Vital Rate Detection (DBSCAN)     #    
                    #---------------------------------------#
                    
                    #--- Breath rate detection ---#
                    br_max = np.argmax(BR_spectrum[idx_br_low:idx_br_high+1])     # 6 ~ 25
                    br_idx = (br_max + idx_br_low)
                    breath_rate = br_idx * vibrate_res * 60
                    Breath_Rate[k] = breath_rate
                    #print(f'BR:\t{breath_rate}')

                    # Breath rate detection, find top 3 peaks
                    num_br_peak = 3
                    breath_rate = []
                    for i in range(num_br_peak):
                        br_max = np.argmax(BR_spectrum[idx_br_low:idx_br_high+1])     # 6 ~ 25
                        br_idx = (br_max + idx_br_low)
                        breath_rate.append(br_idx * vibrate_res * 60)
                    
                        # remove peak, right
                        pr = 1
                        while True:
                            if br_idx+pr+1 >= 25:
                                break

                            if BR_spectrum[br_idx+pr] >= BR_spectrum[br_idx+pr+1]:
                                pr += 1
                            else:
                                break

                        # remove peak, left
                        pl = 1
                        while True:
                            if br_idx-pl-1 <= 5:
                                break
                                
                            if BR_spectrum[br_idx-pl] >= BR_spectrum[br_idx-pl-1]:
                                pl += 1
                            else:
                                break

                        # remove peak
                        BR_spectrum[br_idx-pl:br_idx+pr+1] = 0
                        #print(f'BR Spec: {BR_spectrum[idx_br_low:idx_br_high]}')

                    Breath_Rate[k] = breath_rate[0]
                    #print(f'BR peaks:\t{breath_rate}')
                    #_ = input('Press ENTER key to continue')
                    
                    #--- Heart rate detection ---#
                    # find top N peaks of heart rate spectrum
                    peaks = []
                    for j in range(configs.NUM_RX):
                        for i in range(num_peaks):
                            hr_max = np.argmax(HR_spectrum[idx_hr_low:idx_hr_high+1])        # 41 ~ 102
                            #hr_max = np.argmax(HR_spectrum[idx_hr_low:idx_hr_high, j])     # 41 ~ 102
                            
                            if hr_max == 0:
                                break
                                
                            hr_idx = hr_max + idx_hr_low
                            
                            # remove peak, right
                            pr = 1
                            while True:
                                if HR_spectrum[hr_idx+pr] >= HR_spectrum[hr_idx+pr+1]:
                                #if HR_spectrum[hr_idx+pr, j] >= HR_spectrum[hr_idx+pr+1, j]:
                                    pr += 1
                                else:
                                    break

                            # remove peak, left
                            pl = 1
                            while True:
                                if HR_spectrum[hr_idx-pl] >= HR_spectrum[hr_idx-pl-1]:
                                #if HR_spectrum[hr_idx-pl, j] >= HR_spectrum[hr_idx-pl-1, j]:
                                    pl += 1
                                else:
                                    break
                            
                            # remove peak
                            HR_spectrum[hr_idx-pl:hr_idx+pr+1] = 0
                            #HR_spectrum[hr_idx-pl:hr_idx+pr+1, j] = 0
                            #print(f'HR Spec: {HR_spectrum[idx_hr_low:idx_hr_high+1, j]}')
                            #print(f'HR Spec: {HR_spectrum[idx_hr_low:idx_hr_high+1]}')
                            
                            # harmonic of breath rate?
                            heart_rate = hr_idx * vibrate_res * 60
                            for p in range(len(breath_rate)):
                                if heart_rate % breath_rate[p] <= 1 or (breath_rate[p] - (heart_rate % breath_rate[p])) <= 1:
                                    #print('Harmonic of breath rate!')
                                    continue
                                else:
                                    peaks.append(heart_rate)

                    # put found peaks into the input circular buffer of dbscan
                    for j in range(len(peaks)):
                        #dbscan_in[(ptr+j)%(int((t0/t1)*configs.NUM_RX*num_peaks)), 0] = peaks[j]
                        dbscan_in[(ptr+j)%(int((t0/t1)*num_peaks)), 0] = peaks[j]
                    ptr =  ptr + len(peaks)
                    #if ptr > (int(t1*configs.NUM_RX*num_peaks)):
                    #    ptr = ptr % (int(t1*configs.NUM_RX*num_peaks))
                    if ptr > (int((t0/t1)*num_peaks)):
                        ptr = ptr % (int((t0/t1)*num_peaks))
                    #print(f'Pointer:\t{ptr}')    
                    
                    if configs.plotDBSCAN:
                        plt.vlines(72, -10, 10)
                        plt.scatter(dbscan_in[:,0], dbscan_in[:,1])
                        plt.pause(0.05)
                        plt.clf()
                    
                    #-------------------#
                    # dbscan clustering #
                    #-------------------#
                    
                    #print(f'dbscan_in:\t{dbscan_in}')
                    labels = DBSCAN(eps=2.0, min_samples=4, metric='euclidean').fit(dbscan_in).labels_
                    #print(f'labels:\t{labels}')
                    label_count = Counter(labels)
                    #print(f'label count:\t{label_count}')
                    
                    if len(label_count.most_common()) == 1:
                        pred_label = label_count.most_common()[0][0]
                    else:
                        if label_count.most_common()[0][0] == -1:
                            pred_label = label_count.most_common()[1][0]
                        else:
                            pred_label = label_count.most_common()[0][0]
                    #print(f'predict label:\t{pred_label}')
                    
                    pred_hr = dbscan_in[labels == pred_label]
                    #print(f'predict HR:\t{pred_hr}')
                    Heart_Rate[k] = pred_hr[:,0].mean()
                    #print(f'Heart rate:\t{Heart_Rate[k]}')
                    
                    # exponential smoothing with smoothing factor, 0 < alpha < 1
                    Breath_Rate[k] = alpha * Breath_Rate[k] + (1 - alpha) * Breath_pre[k]
                    Heart_Rate[k] = alpha * Heart_Rate[k] + (1 - alpha) * Heart_pre[k]
                    Breath_pre[k] = Breath_Rate[k]
                    Heart_pre[k] = Heart_Rate[k]
                    Human_distance[k] = position[k]*configs.d_res

                    print('='*10)
                    print(f'Person {k+1}')
                    print('='*10)
                    print(f'Distance:\t{Human_distance[k]:.2f} (m)')
                    print(f'Breath Rate:\t{Breath_Rate[k]:.2f}')
                    print(f'Heart Rate:\t{Heart_Rate[k]:.2f}')
                    print('-'*30)
                    print()
                    
                    # make a beep
                    sound = pygame.mixer.Sound('bell.oga')
                    sound.play()

            # --- vital rate and waveform display ---
            if configs.plotWaveform and (num_frame % t1) == 0:
                for k in range(human_num):
                    plt.plot(phase_br_plt[k, :]*configs.wave_length*1e3/(4*np.pi), label=f'Person {k+1}, Breath Rate: {Breath_Rate[k]:.2f}')
                    plt.plot(phase_hr_plt[k, :]*configs.wave_length*1e3/(4*np.pi), label=f'                Heart Rate:   {Heart_Rate[k]:.2f}')

                plt.xlim(0, 12.8)
                #plt.ylim(-np.pi*2, np.pi*2)
                plt.ylim(-4, 4)
                plt.xticks(t_axis, t_axis_s)
                plt.xlabel('Time (sec)')
                #plt.ylabel('Phase (rad)')
                plt.ylabel('Amplitude (mm)')
                plt.title('Vital Waveform')
                plt.legend(loc = 'upper right')
                plt.pause(0.05)
                plt.clf()

        num_frame = num_frame + 1
        if num_frame % t0 == 0:
            num_frame = 0
        
        #end_count = time.process_time()
        #end_count = time.perf_counter()
        #print(f"The time used to process one frame:\t{end_count-start_count}")
        
    print("Disconnecting...")
    # Specifically kill the main process to avoid hanging
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    main()
