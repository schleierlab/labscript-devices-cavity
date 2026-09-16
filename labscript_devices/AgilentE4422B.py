#####################################################################
#                                                                   #
# /AgilentE4422B.py                                                 #
#                                                                   #
#                                                                   #
#####################################################################
from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker

from labscript import IntermediateDevice, config, LabscriptError, StaticAnalogQuantity, AnalogOut, DigitalOut, set_passed_properties

import numpy as np
import labscript_utils.h5_lock, h5py
from ctypes import *
import struct

max_deviations = [
    {'low': 0.1,        'high': 249.999,    'dev': 10.0},
    {'low': 249.999,    'high': 500.0,      'dev': 5.0},
    {'low': 500.0,      'high': 1000.0,     'dev': 10.0},
    {'low': 1000.0,     'high': 2000.0,     'dev': 20.0},
    {'low': 2000.0,     'high': 4000.0,     'dev': 40.0}
]

@labscript_device
class AgilentE4422B(IntermediateDevice):
    description = 'Agilent RF Signal Generator'
    allowed_children = [StaticAnalogQuantity]
    generation = 2

    @set_passed_properties(property_names = {
        "connection_table_properties": ["com_port"]}
        )
    def __init__(self,name,parent_device,com_port='COM1',FMSignal=None,RFOnOff=None,freq_limits = None,freq_conv_class = None,freq_conv_params = {},amp_limits=None,amp_conv_class = None,amp_conv_params = {},phase_limits=None,phase_conv_class = None,phase_conv_params = {}):
        #self.clock_type = parent_device.clock_type # Don't see that this is needed anymore

        IntermediateDevice.__init__(self,name,parent_device)

        self.BLACS_connection = com_port
        self.sweep_dt = 2e-6

        self.RFSettings = {
            'freq': 0,
            'amp': 0,
            'phase': 0,
            'freq_dev':0 ##EE
        }

        self.sweep_params = {
#            'type': None,
            'low': 0,
            'high': 1,
            'duration': 0,
#            'falltime': 0,
            'sample_rate': self.sweep_dt
        }
        self.sweep = False
        self.ext_in = False


        self.frequency = StaticAnalogQuantity(self.name+'_freq',self,'freq',freq_limits,freq_conv_class,freq_conv_params)
        self.amplitude = StaticAnalogQuantity(self.name+'_amp',self,'amp',amp_limits,amp_conv_class,amp_conv_params)
        self.phase = StaticAnalogQuantity(self.name+'_phase',self,'phase',phase_limits,phase_conv_class,phase_conv_params)

        self.FMSignal = {}
        self.RFOnOff = {}

        if FMSignal:
            if 'device' in FMSignal and 'connection' in FMSignal:
                self.FMSignal = AnalogOut(self.name+'_FMSignal', FMSignal['device'], FMSignal['connection'])
            else:
                raise LabscriptError('You must specify the "device" and "connection" for the analog output FMSignal of '+self.name)
        else:
            raise LabscriptError('Expected analog output for "FMSignal" control')

        if RFOnOff:
            if 'device' in RFOnOff and 'connection' in RFOnOff:
                self.RFOnOff = DigitalOut(self.name+'_RFOnOff', RFOnOff['device'], RFOnOff['connection'])
            else:
                raise LabscriptError('You must specify the "device" and "connection" for the digital output RFOnOff of '+self.name)
        else:
            raise LabscriptError('Expected digital output for "RFOnOff" control')

    def reset(self, t):
        self.FMSignal.constant(t,-1)

        return t

    def synthesizeRF(self, t, freq, amp, ph, freqUnits=None, ampUnits=None, phaseUnits=None):
        self.RFSettings['freq'] = freq
        self.RFSettings['amp'] = amp
        self.RFSettings['phase'] = ph

        self.sweep = False
        self.ext_in = False

        self.FMSignal.constant(t,0)
        self.RFOnOff.go_high(t)

        return t

    def synthesizeRF_ext(self, t, freq, amp, ph, freq_dev, freqUnits=None, ampUnits=None, phaseUnits=None):
        self.RFSettings['freq'] = freq
        self.RFSettings['amp'] = amp
        self.RFSettings['phase'] = ph
        self.RFSettings['freq_dev'] = freq_dev ##EE

        self.sweep = False
        self.ext_in = True

        self.FMSignal.constant(t,0) #when plugged into the feedforward, needs to be set to 0.017 to account for some offset
        self.RFOnOff.go_high(t)

        return t

    def sweepRF(self, t, duration, low_freq, high_freq, sample_rate, amp):
        center_freq = np.mean([low_freq,high_freq])
        for dev_freq_range in max_deviations:
            if center_freq > dev_freq_range['low'] and center_freq <= dev_freq_range['high']:
                deviation = high_freq - center_freq
                if deviation > dev_freq_range['dev']:
                    raise LabscriptError('Frequency deviation is too high. For a center frequency of '+str(center_freq)+' MHz, the maximum deviation is '+str(dev_freq_range['dev'])+' MHz ('+str(deviation)+' MHz requested)')

        self.sweep_params['duration'] = duration
        self.sweep_params['low'] = low_freq
        self.sweep_params['high'] = high_freq
        self.sweep_params['sample_rate'] = sample_rate

        self.RFSettings['freq'] = center_freq
        self.RFSettings['amp'] = amp
        self.RFSettings['phase'] = 0

        self.sweep = True

        self.RFOnOff.go_high(t)
        self.FMSignal.ramp(t,duration=duration,initial=-1.03,final=1.08,samplerate=sample_rate)
        #To account for the gain of the feedforward box: initial=-0.085 and +final=+1

        self.RFOnOff.go_low(t+duration)

        return t+duration


    def generate_code(self, hdf5_file):

        IntermediateDevice.generate_code(self, hdf5_file)

        if len(self.child_devices) != 3:
            raise LabscriptError("Wrong number of child_devices. This device expects exactly 3 children")

        grp = hdf5_file.create_group('/devices/'+self.name)


        profile_dtypes = [('freq',float),
                          ('phase',float),
                          ('amp',float),
                          ('ext_in',bool), #EE
                          ('freq_dev',float)] #EE


        profile_table = np.zeros(1, dtype=profile_dtypes)

        profile_table['freq'] = self.RFSettings['freq']
        profile_table['amp'] = self.RFSettings['amp']
        profile_table['phase'] = self.RFSettings['phase']
        profile_table['ext_in'] = 0 ##EE
        profile_table['freq_dev'] = 0 #EE

        ## Emily edit
        if self.ext_in:
            profile_table['ext_in'] = 1
            profile_table['freq_dev'] = self.RFSettings['freq_dev']

        grp.create_dataset('RF_DATA',compression=config.compression,data=profile_table)


        if self.sweep:
            sweep_dtypes = [('sweep_low',float),
                            ('sweep_high',float),
                            ('sweep_duration',float),
                            ('sweep_samplerate',float)]

            sweep_table = np.empty(1, dtype=sweep_dtypes)

#            sweep_table['sweep_type'] = sweep_types[output.sweep_params['type']]
            sweep_table['sweep_low'] = self.sweep_params['low']
            sweep_table['sweep_high'] = self.sweep_params['high']
            sweep_table['sweep_duration'] = self.sweep_params['duration']
#            sweep_table['sweep_falltime'] = output.sweep_params['falltime']
            sweep_table['sweep_samplerate'] = self.sweep_params['sample_rate']

            grp.create_dataset('SWEEP_DATA',compression=config.compression,data=sweep_table)

import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

@BLACS_tab
class AgilentE4422BTab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
        self.base_units =    {'freq':'MHz',         'amp':'dBm',   'phase':'Degrees'}
        self.base_min =      {'freq':0.1,           'amp':-136.0,  'phase':0}
        self.base_max =      {'freq':4000.,         'amp':25.0,    'phase':360}
        self.base_step =     {'freq':1.0,           'amp':1.0,     'phase':1}
        self.base_decimals = {'freq':4,             'amp':4,       'phase':3} # TODO: find out what the phase precision is!

        # Create DDS Output objects
        RF_prop = {}
        RF_prop['channel 0'] = {}
        for subchnl in ['freq', 'amp', 'phase']:
            RF_prop['channel 0'][subchnl] = {'base_unit':self.base_units[subchnl],
                                                'min':self.base_min[subchnl],
                                                'max':self.base_max[subchnl],
                                                'step':self.base_step[subchnl],
                                                'decimals':self.base_decimals[subchnl]
                                                }

        # Create the output objects
        self.create_dds_outputs(RF_prop)

        # Create widgets for output objects
        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("RF Output",dds_widgets))

        # Store the COM port to be used
        self.com_port = str(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)

        # Create and set the primary worker
        self.create_worker("main_worker", AgilentE4422BWorker, {'com_port':self.com_port})
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(False) # !!!
        self.supports_smart_programming(False) # !!!

@BLACS_worker
class AgilentE4422BWorker(Worker):

    def init(self):

        global h5py; import labscript_utils.h5_lock, h5py
        global serial; import serial

        self.COMPort = self.com_port
        self.GPIBAddress = 19
        self.baudrate = 19200

        self.final_values = {}

        self.smart_cache = {'RF_DATA': None,
                            'SWEEP_DATA': None}

        ser = serial.Serial(self.COMPort, baudrate=self.baudrate, timeout=1)
        if(ser.isOpen() == False):
            ser.open()

        ser.write(':SYST:COMM:GPIB:ADDR?\r\n'.encode())
        GPIBAddr = ser.readline()
        ser.close()

        if GPIBAddr == '':
            raise LabscriptError("Device is not connected")
        elif int(GPIBAddr) != self.GPIBAddress:
            raise LabscriptError("Expected device at GPIB address "+str(self.GPIBAddress)+f" (actual address is {int(GPIBAddr)})")

        ser.open()
        ser.write(':OUTP?\r\n'.encode())
        output_state = ser.readline()

        if int(output_state) != 1:
            ser.write(':OUTP ON\r\n'.encode())

        ser.close()

    def check_remote_values(self):

        results = {'channel 0': {}}
        self.final_values = {}

        ser = serial.Serial(self.COMPort, baudrate=self.baudrate, timeout=1)
        if(ser.isOpen() == False):
            ser.open()

        ser.write(':FREQ?\r\n'.encode())
        freq = float(ser.readline()) / 1.0e6
        ser.write(':POW?\r\n'.encode())
        amp = float(ser.readline())

        results['channel 0']['freq'] = freq
        results['channel 0']['amp'] = amp
        results['channel 0']['phase'] = 0

        ser.close()

        return results


    def program_manual(self,front_panel_values):

        ser = serial.Serial(self.COMPort, baudrate=self.baudrate, timeout=1)
        if(ser.isOpen() == False):
            ser.open()

        values = front_panel_values['channel 0']

        ser.write((':POW '+str(values['amp'])+' dBm; :FREQ '+str(values['freq'])+' MHz\r\n').encode())

        ser.close()

        # Now that a manual update has been done, we'd better invalidate the saved RF_DATA:
        self.smart_cache['RF_DATA'] = None

        return self.check_remote_values()


    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):

        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values

        # Store the final values to for use during transition_to_manual:
        self.final_values = {'channel 0' : {}}
        RF_data = None
        sweep_data = None

        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['/devices/'+device_name]
            # If there are values to set the unbuffered outputs to, set them now:
            if 'RF_DATA' in group:
                RF_data = group['RF_DATA'][:][0]

            if 'SWEEP_DATA' in group:
                sweep_data = group['SWEEP_DATA'][:][0]

        serial_line = ""
        if RF_data is not None:
            data = RF_data
            if fresh or data != self.smart_cache['RF_DATA']:
                self.logger.debug('Static data has changed, reprogramming.')
                self.smart_cache['RF_DATA'] = data

            self.final_values['channel 0']['freq'] = data['freq']
            self.final_values['channel 0']['amp'] = data['amp']
            self.final_values['channel 0']['phase'] = data['phase']

            serial_line += ':POW '+str(data['amp'])+' dBm; :FREQ '+str(data['freq'])+' MHz'

            ##EE
            if data['ext_in'] == 1:
                freq_dev = data['freq_dev'] #MHz/Volt
                serial_line += '; :FM:DEV '+str(freq_dev)+' MHz; :FM:SOUR EXT1; :FM:STAT ON'


        if sweep_data is not None:
            data = sweep_data
            if fresh or data != self.smart_cache['SWEEP_DATA']:
                self.logger.debug('Static data has changed, reprogramming.')
                self.smart_cache['SWEEP_DATA'] = data

            freq_deviation = (data['sweep_high'] - data['sweep_low']) * 0.5
            serial_line += '; :FM:DEV '+str(freq_deviation)+' MHz; :FM:SOUR EXT1; :FM:STAT ON'

        serial_line += '\r\n'

        ser = serial.Serial(self.COMPort, baudrate=self.baudrate, timeout=1)
        if(ser.isOpen() == False):
            ser.open()
        ser.write(serial_line.encode())
        ser.close()

        return self.final_values

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def transition_to_manual(self,abort = False):

        if abort:
            # If we're aborting the run, then we need to reset the Static DDSs to their initial values.
            # We also need to invalidate the smart programming cache.
            values = self.initial_values['channel 0']
            self.smart_cache['RF_DATA'] = None
            self.smart_cache['SWEEP_DATA'] = None
        else:
            # If we're not aborting the run, then we need to set the Static DDSs to their final values.
            values = self.final_values['channel 0']

        ser = serial.Serial(self.COMPort, baudrate=self.baudrate, timeout=1)
        if(ser.isOpen() == False):
            ser.open()
        ser.write((':POW '+str(values['amp'])+' dBm; :FREQ '+str(values['freq'])+' MHz\r\n').encode())
        ser.close()

        return True

    def shutdown(self):
        return
