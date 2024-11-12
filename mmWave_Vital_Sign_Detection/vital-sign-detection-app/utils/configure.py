# -*- coding: utf-8 -*-
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Tuesday, June 21, 2022                                             #
# Description: Reading FMCW RADAR System Configuration                     #
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

import sys
from distutils.util import strtobool

class Configure:
    def __init__(self, config_file='fmcw_radar.cfg'):
        config = self.config_file_to_dict(config_file)
    
        #----------------- Radar Parameters -----------------#
        # light speed in m/s
        parameter = 'LIGHT_SPEED'
        if parameter in config:
            self.LIGHT_SPEED = float(config[parameter])
        else:
            self.LIGHT_SPEED = 299792458.0
        
        #----------------- Desired Radar Metric -----------------#
        # Range Resolution, meter
        parameter = 'range_resolution_m'
        if parameter in config:
            self.range_resolution_m = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # Maximum range, meter
        parameter = 'max_range_m'
        if parameter in config:
            self.max_range_m = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
          
        # Maximum speed, m/s
        parameter = 'max_speed_m_s'
        if parameter in config:
            self.max_speed_m_s = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # Speed resolution, m/s
        parameter = 'speed_resolution_m_s'
        if parameter in config:
            self.speed_resolution_m_s = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))

        #----------------- Radar Configuration -----------------#
        # number of beams of DBF
        parameter = 'num_beams'
        if parameter in config:
            self.num_beams = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
        
        # FOV, angle ranges from -max_angle to max_angle degrees
        parameter = 'max_angle_degrees'
        if parameter in config:
            self.max_angle_degrees = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
        
        # sample rate, MHz
        parameter = 'sample_rate_Hz'
        if parameter in config:
            self.sample_rate_Hz = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))

        # RX Antenna Mask, for example: activate RX2 and RX3, rx_mask = 110
        parameter = 'rx_mask'
        if parameter in config:
            self.rx_mask = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # TX Antenna Mask, 1 for activate TX1
        parameter = 'tx_mask'
        if parameter in config:
            self.tx_mask = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # IF Gain, dB
        parameter = 'if_gain_dB'
        if parameter in config:
            self.if_gain_dB = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # TX power level: 0 ~ 31
        parameter = 'tx_power_level'
        if parameter in config:
            self.tx_power_level = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # Frame Period
        parameter = 'frame_repetition_time_s'
        if parameter in config:
            self.frame_repetition_time_s = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # Radar Frequency Band, Hz
        parameter = 'center_frequency_Hz'
        if parameter in config:
            self.center_frequency_Hz = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        #----------------- Antenna Configuration -----------------#
        # number of transmit antenna
        parameter = 'NUM_TX'
        if parameter in config:
            self.NUM_TX = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
        
        # number of receive antenna
        parameter = 'NUM_RX'
        if parameter in config:
            self.NUM_RX = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        #----------------- Get Radar Configuration Parameters -----------------#
        self.start_frequency_Hz = None
        self.end_frequency_Hz = None
        self.num_chirps_per_frame = None
        self.num_samples_per_chirp = None
        self.chirp_repetition_time_s = None
        self.mimo_mode = None
        self.frame_rate = None
        self.d_res = None
        self.d_max = None
        self.V_res = None
        self.V_max = None
        self.wave_length = None
            
        #----------------- FFT Parameters -----------------#
        # number  of range fft
        parameter = 'RANGE_FFT'
        if parameter in config:
            self.RANGE_FFT = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # number of Doppler fft
        parameter = 'DOPPLER_FFT'
        if parameter in config:
            self.DOPPLER_FFT = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
        
        # number of vibration fft
        parameter = 'VIBRATION_FFT'
        if parameter in config:
            self.VIBRATION_FFT = int(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        #----------------- DSP Parameters -----------------#
        # exponential smoothing factor, 0 < alpha < 1
        parameter = 'alpha'
        if parameter in config:
            self.alpha = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # bandpass filter for heart rate in Hz
        parameter = 'HR_LOW'
        if parameter in config:
            self.HR_LOW = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        parameter = 'HR_HIGH'
        if parameter in config:
            self.HR_HIGH = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # bandpass filter for breath rate in Hz
        parameter = 'BR_LOW'
        if parameter in config:
            self.BR_LOW = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        parameter = 'BR_HIGH'
        if parameter in config:
            self.BR_HIGH = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
            
        # minimum attenuation in the stopband (dB)
        parameter = 'attenuation'
        if parameter in config:
            self.attenuation = float(config[parameter])
        else:
            print('Warning: parameter {} was not set proporly.'.format(parameter))
        
        #----------------- Directories and Files -----------------#
        # log directory
        parameter = 'log_dir'
        if parameter in config:
            self.log_dir = config[parameter]
        else:
            self.log_dir = './'
        
        #----------------- Plot Options -----------------#
        parameter = 'plotRangeAzimuth'
        if parameter in config:
            try:
                self.plotRangeAzimuth = bool(strtobool(config[parameter]))
            except:
                self.plotRangeAzimuth = False
        else:
            self.plotRangeAzimuth = False
            
        parameter = 'Range_plot'
        if parameter in config:
            try:
                self.Range_plot = bool(strtobool(config[parameter]))
            except:
                self.Range_plot = False
        else:
            self.Range_plot = False
            
        parameter = 'Range_plot_clutter_removed'
        if parameter in config:
            try:
                self.Range_plot_clutter_removed = bool(strtobool(config[parameter]))
            except:
                self.Range_plot_clutter_removed = False
        else:
            self.Range_plot_clutter_removed = False
            
        parameter = 'RangeDopp_plot'
        if parameter in config:
            try:
                self.RangeDopp_plot = bool(strtobool(config[parameter]))
            except:
                self.RangeDopp_plot = False
        else:
            self.RangeDopp_plot = False
            
        parameter = 'MakeMovie'
        if parameter in config:
            try:
                self.MakeMovie = bool(strtobool(config[parameter]))
            except:
                self.MakeMovie = False
        else:
            self.MakeMovie = False
            
        parameter = 'Spectrogram'
        if parameter in config:
            try:
                self.Spectrogram = bool(strtobool(config[parameter]))
            except:
                self.Spectrogram = False
        else:
            self.Spectrogram = False
            
        parameter = 'show_doppler_freq'
        if parameter in config:
            try:
                self.show_doppler_freq = bool(strtobool(config[parameter]))
            except:
                self.show_doppler_freq = False
        else:
            self.show_doppler_freq = False
            
        parameter = 'logGabor'
        if parameter in config:
            try:
                self.logGabor = bool(strtobool(config[parameter]))
            except:
                self.logGabor = False
        else:
            self.logGabor = False
            
        parameter = 'plot_hr_filter'
        if parameter in config:
            try:
                self.plot_hr_filter = bool(strtobool(config[parameter]))
            except:
                self.plot_hr_filter = False
        else:
            self.plot_hr_filter = False
            
        parameter = 'plot_br_filter'
        if parameter in config:
            try:
                self.plot_br_filter = bool(strtobool(config[parameter]))
            except:
                self.plot_br_filter = False
        else:
            self.plot_br_filter = False
            
        parameter = 'plotRangeSlow'
        if parameter in config:
            try:
                self.plotRangeSlow = bool(strtobool(config[parameter]))
            except:
                self.plotRangeSlow = False
        else:
            self.plotRangeSlow = False
            
        parameter = 'plotRangeUnwrap'
        if parameter in config:
            try:
                self.plotRangeUnwrap = bool(strtobool(config[parameter]))
            except:
                self.plotRangeUnwrap = False
        else:
            self.plotRangeUnwrap = False
            
        parameter = 'plotUnwrap'
        if parameter in config:
            try:
                self.plotUnwrap = bool(strtobool(config[parameter]))
            except:
                self.plotUnwrap = False
        else:
            self.plotUnwrap = False
        
        parameter = 'plotDenoise'
        if parameter in config:
            try:
                self.plotDenoise = bool(strtobool(config[parameter]))
            except:
                self.plotDenoise = False
        else:
            self.plotDenoise = False
        
        parameter = 'plotWaveform'
        if parameter in config:
            try:
                self.plotWaveform = bool(strtobool(config[parameter]))
            except:
                self.plotWaveform = False
        else:
            self.plotWaveform = False
            
        parameter = 'plotVibrateFFT'
        if parameter in config:
            try:
                self.plotVibrateFFT = bool(strtobool(config[parameter]))
            except:
                self.plotVibrateFFT = False
        else:
            self.plotVibrateFFT = False
            
        parameter = 'plotDBSCAN'
        if parameter in config:
            try:
                self.plotDBSCAN = bool(strtobool(config[parameter]))
            except:
                self.plotDBSCAN = False
        else:
            self.plotDBSCAN = False
            
    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and line[0] == '#':
                    continue
                if '=' in line:
                    pair = line.strip().split('=', 1)
                    key = pair[0]
                    value = pair[1]
                    try:
                        if key in config:
                            print('Warning: duplicated parameter name found: {}, updated.'.format(key))
                        if value == None:
                            print('Warning: parameter {} was not set proporly.'.format(key))
                        elif value[0] == '[' and value[-1] == ']':
                            value_list = list(value[1:-1].split(','))
                            config[key] = value_list
                        else:
                            config[key] = value
                    except Exception:
                        print('configuration parsing error, please check correctness of the config file.')
                        exit(1)

        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    def show_data_summary(self, logger):
        logger.info('=' * 12 + ' FMCW RADAR SYSTEM CONFIGURATION ' + '=' * 12)
        logger.info('    Radar Parameters:')
        logger.info('      Light speed (m/s):                    {}'.format(self.LIGHT_SPEED))
        logger.info('  ' + '-' * 52)
        logger.info('    Desired Radar Metric:')
        logger.info('      Range resolution (m):                 {}'.format(self.range_resolution_m))
        logger.info('      Maximum range (m):                    {}'.format(self.max_range_m))
        logger.info('      Maximum speed (m/s):                  {}'.format(self.max_speed_m_s))
        logger.info('      Speed resolution (m/s):               {}'.format(self.speed_resolution_m_s))
        logger.info('  ' + '-' * 52)
        logger.info('    Radar Configuration:')
        logger.info('      Number of beams of DBF:               {}'.format(self.num_beams))
        logger.info('      Radar FOV (degree):                   {}'.format(self.max_angle_degrees*2))
        logger.info('      Sample rate (Hz):                     {}'.format(self.sample_rate_Hz))
        logger.info('      RX Antenna Mask:                      {}'.format(self.rx_mask))
        logger.info('      TX Antenna Mask:                      {}'.format(self.tx_mask))
        logger.info('      IF Gain (dB):                         {}'.format(self.if_gain_dB))
        logger.info('      TX Power Level:                       {}'.format(self.tx_power_level))
        logger.info('      Frame Period (s):                     {}'.format(self.frame_repetition_time_s))
        logger.info('      Frame Rate (frames/s):                {}'.format(self.frame_rate))
        logger.info('      Radar Frequency Band (Hz):            {}'.format(self.center_frequency_Hz))
        logger.info('      Wave Length (m):                      {}'.format(self.wave_length))
        logger.info('  ' + '-' * 52)
        logger.info('    Antenna Configuration:')
        logger.info('      Number of transmit antenna:           {}'.format(self.NUM_TX))
        logger.info('      Number of receive antenna:            {}'.format(self.NUM_RX))
        logger.info('  ' + '-' * 52)
        logger.info('    Get Radar Configuration Parameters:')
        logger.info('      Ramp start frequency (Hz):            {}'.format(self.start_frequency_Hz))
        logger.info('      Ramp end frequency (Hz):              {}'.format(self.end_frequency_Hz))
        logger.info('      Number of Chirps per Frame:           {}'.format(self.num_chirps_per_frame))
        logger.info('      Number of samples per Chirp:          {}'.format(self.num_samples_per_chirp))
        logger.info('      Chirp repetition time (s):            {}'.format(self.chirp_repetition_time_s))
        logger.info('      MIMO Mode:                            {}'.format(self.mimo_mode))
        logger.info('  ' + '-' * 52)
        logger.info('    FFT Parameters:')
        logger.info('      Number of range fft:                  {}'.format(self.RANGE_FFT))
        logger.info('      Number of Doppler fft:                {}'.format(self.DOPPLER_FFT))
        logger.info('      Number of vibration fft:              {}'.format(self.VIBRATION_FFT))
        logger.info('  ' + '-' * 52)
        logger.info('    DSP Parameters:')
        logger.info('      Smoothing factor:                     {}'.format(self.alpha))
        logger.info('      Low cut-off freq. for heart rate (Hz):{}'.format(self.HR_LOW))
        logger.info('      High cut-off freq. for heart rate(Hz):{}'.format(self.HR_HIGH))
        logger.info('      Low cut-off freq. for breath rate(Hz):{}'.format(self.BR_LOW))
        logger.info('      High cutoff freq. for breath rate(Hz):{}'.format(self.BR_HIGH))
        logger.info('      Min. attenuation in the stopband(dB): {}'.format(self.attenuation))
        logger.info('  ' + '-' * 52)
        logger.info('    Directories and Files:')
        logger.info('      Log directory:                        {}'.format(self.log_dir))
        logger.info('  ' + '-' * 52)
        logger.info('    Plot Options:')
        logger.info('      Range Azimuth Plot:                   {}'.format(self.plotRangeAzimuth))
        logger.info('      Range Plot:                           {}'.format(self.Range_plot))
        logger.info('      Range Plot with Static clutter removal:{}'.format(self.Range_plot_clutter_removed))
        logger.info('      Range-Doppler Plot:                   {}'.format(self.RangeDopp_plot))
        logger.info('      Make Movie:                           {}'.format(self.MakeMovie))
        logger.info('      uDoppler Spectrogram:                 {}'.format(self.Spectrogram))
        logger.info('      Doppler Shift in Hz:                  {}'.format(self.show_doppler_freq))
        logger.info('      Log Gabor filter:                     {}'.format(self.logGabor))
        logger.info('      plot Heart rate filter:               {}'.format(self.plot_hr_filter))
        logger.info('      plot Breath rate filter:              {}'.format(self.plot_br_filter))
        logger.info('      plotRangeSlow:                        {}'.format(self.plotRangeSlow))
        logger.info('      plotRangeUnwrap:                      {}'.format(self.plotRangeUnwrap))
        logger.info('      plotUnwrap:                           {}'.format(self.plotUnwrap))
        logger.info('      plot Phase De-noised:                 {}'.format(self.plotDenoise))
        logger.info('      plotWaveform:                         {}'.format(self.plotWaveform))
        logger.info('      plotVibrateFFT:                       {}'.format(self.plotVibrateFFT))
        logger.info('      plotDBSCAN:                           {}'.format(self.plotDBSCAN))
        logger.info('=' * 15 + ' CONFIGURATION SUMMARY END ' + '=' * 15)
        sys.stdout.flush()

    def record_radar_parameters(self, logger):
        logger.info('=' * 19 + ' REAL FMCW RADAR SYSTEM PARAMETERS ' + '=' * 19)
        logger.info('    Real Radar Parameters:')
        logger.info('      Range resolution (m):                 {}'.format(self.d_res))
        logger.info('      Maximum distance (m):                 {}'.format(self.d_max))
        logger.info('      Speed resolution (m/s):               {}'.format(self.V_res))
        logger.info('      Maximum speed (m/s):                  {}'.format(self.V_max))
        logger.info('=' * 23 + ' SHOW PARAMETERS END ' + '=' * 23)
        sys.stdout.flush()
