#####################################################################
#                                                                   #
#  Spectrum.py                                                      #
#                                                                   #
#####################################################################

from labscript import IntermediateDevice, Device, LabscriptError, DigitalOut
from labscript_devices import BLACS_tab
from blacs.tab_base_classes import Worker
from blacs.device_base_class import DeviceTab
import labscript_utils.h5_lock

import h5py

import os
import numpy as np
import dill as pickle
import marshal
import types
import math
import time
import time as ptime
import ast
import warnings
from scipy.signal import chirp, sawtooth
import scipy.special
import traceback

import gc
from .spcm import pyspcm as sp
from .spcm import spcm_errors as se
from .spcm import spcm_tools as st
from .spcm.spcm_modulation_compensation import *
from .spcm.numba_chirp import *

import ctypes
import inspect
import ntpath
# decorate spectrum functions for error detection
se.decorate_functions(sp, se.error_decorator)

###### Sequence data classes ##################################################


class pulse:
    def __init__(
        self,
        start_freq,
        end_freq,
        ramp_time,
        phase,
        amplitude,
        ramp_type,
        painting_function=None,
        painting_freq=False,
        painting_list=False,
    ):
        self.start = start_freq
        self.end = end_freq
        self.ramp_time = ramp_time  # In seconds
        self.phase = phase
        self.amp = amplitude
        self.ramp_type = ramp_type  # String. Can be linear, quadratic, None
        self.painting_function = painting_function
        self.is_painted = False
        if not self.painting_function is None:
            self.is_painted = True
        self.painting_freq = painting_freq
        self.painting_list = painting_list

    def __str__(self):
        # Random number to never use smart programming
        if self.is_painted:
            s = f"Painted sweep using function {self.painting_function} painting frequency: {self.painting_freq}, painting_list: {self.painting_list}"
            try:
                s = s + f"{self.painting_function.extra_params}"
            except Exception as e:
                print(e)
            return s
        s = f"Ramp from {self.start} to {self.end} in t = {self.ramp_time} with amp = {self.amp}, "
        s = s + f"phase = {self.phase}"
        return s


class waveform:
    def __init__(
        self,
        time,
        duration,
        port,
        loops=1,
        is_periodic=False,
        pulses=[],
        delta_start=0,
        delta_end=0,
        modulation_frequencies=[0],
        modulation_amplitudes=[0],
        modulation_phases=[0],
        mask=None,
        remove_self_interaction=False,
    ):
        self.time = time  # chunks
        self.duration = duration  # chunks
        # if self.duration == 0:
        #     print("duration is zero")
        #     traceback.print_stack()
        self.port = port  # int
        self.loops = loops  # int
        self.delta_start = delta_start  # samples
        self.delta_end = delta_end  # samples

        self.is_periodic = is_periodic  # bool
        self.modulation_frequencies = modulation_frequencies
        self.modulation_amplitudes = modulation_amplitudes
        self.modulation_phases = modulation_phases
        self.has_mask = True
        self.remove_self_interaction = remove_self_interaction
        if mask == None:
            mask = lambda x: np.ones_like(x)
            self.has_mask = False
        self.mask = mask
        # Make new copies of pulses.  Why do we need to do this??
        self.pulses = [
            pulse(
                start_freq=p.start,
                end_freq=p.end,
                ramp_time=p.ramp_time,
                phase=p.phase,
                amplitude=p.amp,
                ramp_type=p.ramp_type,
                painting_function=p.painting_function,
            )
            for p in pulses
        ]

        self.sample_start = 0
        self.sample_end = duration

    def add_pulse(
        self,
        start_freq,
        end_freq,
        ramp_time,
        phase,
        amplitude,
        ramp_type,
        painting_function=None,
        painting_freq=False,
        painting_list=False,
    ):
        self.pulses.append(
            pulse(
                start_freq,
                end_freq,
                ramp_time,
                phase,
                amplitude,
                ramp_type,
                painting_function=painting_function,
                painting_freq=painting_freq,
                painting_list=painting_list,
            )
        )

    def __str__(self):
        s = (
            "{("
            + ",".join([str(i) for i in self.pulses])
            + f") on ch. {self.port} at t = {self.time} in {self.duration} time with {self.loops} loops"
            + f", mod. freq. {self.modulation_frequencies}, mod. amp. {self.modulation_amplitudes} and mod. phases {self.modulation_phases}"
            + f".  self.has_mask = {self.has_mask * np.random.rand(1)}"
            + "}"
        )
        return s

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))


class waveform_group:
    def __init__(self, time, duration, waveforms, loops=1):
        self.time = time
        self.duration = duration
        self.waveforms = waveforms
        self.loops = loops

    def add_waveform(self, waveform):
        self.waveforms.append(waveform)

    def __str__(self):
        s = f"Time:\t {self.time}\n"
        s += "Duration:\t %d\n" % self.duration
        s += "Waveforms:\t %s\n" % [str(i) for i in self.waveforms]
        s += "Loops:\t %d\n" % self.loops
        return s


class channel_settings:
    def __init__(self, name, power, port):
        self.power = power
        self.name = name
        self.port = port


class sample_data:
    def __init__(self, channels, mode, clock_freq):
        self.waveform_groups = []
        self.mode = mode
        self.clock_freq = clock_freq
        self.channels = channels


class sequence_instr:
    def __init__(self, step, next_step, segment, loops):
        self.step = step
        self.segment = segment
        self.loops = loops
        self.next_step = next_step


##### Labscript classes #######################################################
class Spectrum(IntermediateDevice):
    def __init__(self, name, parent_device, card_address, trigger, triggerDur=5e-6, worker=None):
        self.BLACS_connection = card_address
        Device.__init__(self, name, parent_device, connection=self.BLACS_connection, worker=worker)

        self.set_mode("Off")  # Initialize data structure
        self.samplesPerChunk = 32

        self.triggerDur = triggerDur
        self.raw_waveforms = []

        self.mode_dict = {
            b"single": sp.SPC_REP_STD_SINGLE,
            b"multi": sp.SPC_REP_STD_MULTI,
            b"gate": sp.SPC_REP_STD_GATE,
            b"single_restart": sp.SPC_REP_STD_SINGLERESTART,
            b"sequence": sp.SPC_REP_STD_SEQUENCE,
            b"fifo_single": sp.SPC_REP_FIFO_SINGLE,
            b"fifo_multi": sp.SPC_REP_FIFO_MULTI,
            b"fifo_gate": sp.SPC_REP_FIFO_GATE,
        }

        if trigger:
            if "device" in trigger and "connection" in trigger:
                self.triggerDO = DigitalOut(
                    self.name + "_Trigger", trigger["device"], trigger["connection"]
                )
            else:
                raise LabscriptError(
                    'You must specify the "device" and "connection" for the trigger input to the Spectrum card'
                )
        else:
            raise LabscriptError("No trigger specified for device " + self.name)

    def set_mode(
        self,
        mode_name,
        channels=[],
        clock_freq=625,
        use_ext_clock=False,
        ext_clock_freq=10,
        export_data=False,
        export_path="",
    ):
        """
        Initializes channel_data structure that will be filled by single_freq,
        comb, sweep...
        """
        self.use_ext_clock = use_ext_clock
        self.ext_clock_freq = sp.MEGA(ext_clock_freq)
        self.export_data = export_data
        self.export_path = export_path

        self.num_chs = len(channels)

        if self.num_chs > 0:
            self.duration_min_c = int(12 / self.num_chs)
        else:
            self.duration_min_c = np.inf

        if self.num_chs > 0:
            self.max_duration = 2048e6 / (sp.MEGA(clock_freq) * len(channels))

        if self.num_chs == 3:
            raise LabscriptError(
                "Spectrum card cannot have 3 channels. Please remove a channel or add a dummy."
            )

        channel_objects = []
        enabled_ports = []
        for channel in channels:
            channel_objects.append(
                channel_settings(channel["name"], channel["power"], channel["port"])
            )
            enabled_ports.append(channel["port"])

        self.sample_data = sample_data(
            channels=channel_objects, mode=mode_name, clock_freq=sp.MEGA(clock_freq)
        )
        self.enabled_ports = enabled_ports

    def reset(self, t):
        return t

    # Functions that are simplifications on sweep_comb()
    def single_freq(
        self,
        t,
        duration,
        freq,
        amplitude,
        phase,
        ch,
        loops=1,
        modulation_frequencies=[0],
        modulation_amplitudes=[0],
        modulation_phases=[0],
        ramp_type="static",
        mask=None,
    ):
        if duration == 0:
            return t
        t = self.sweep_comb(
            t=t,
            duration=duration,
            start_freqs=[freq],
            end_freqs=[freq],
            amplitudes=[amplitude],
            phases=[phase],
            ch=ch,
            ramp_type=ramp_type,
            loops=loops,
            modulation_frequencies=modulation_frequencies,
            modulation_amplitudes=modulation_amplitudes,
            modulation_phases=modulation_phases,
            mask=mask,
        )
        return t

    def sweep(
        self,
        t,
        duration,
        start_freq,
        end_freq,
        amplitude,
        phase,
        ch,
        ramp_type,
        loops=1,
        modulation_frequencies=[0],
        modulation_amplitudes=[0],
        modulation_phases=[0],
        mask=None,
    ):
        t = self.sweep_comb(
            t,
            duration,
            [start_freq],
            [end_freq],
            [amplitude],
            [phase],
            ch,
            ramp_type,
            loops,
            modulation_frequencies=modulation_frequencies,
            modulation_amplitudes=modulation_amplitudes,
            modulation_phases=modulation_phases,
            mask=mask,
        )
        return t

    def comb(
        self,
        t,
        duration,
        freqs,
        amplitudes,
        phases,
        ch,
        ramp_type="static",
        loops=1,
        modulation_frequencies=[0],
        modulation_amplitudes=[0],
        modulation_phases=[0],
        mask=None,
        detuning=0,
        detuning_shift=0,
        remove_self_interaction=False,
    ):
        t = self.sweep_comb(
            t=t,
            duration=duration,
            start_freqs=freqs,
            end_freqs=freqs,
            amplitudes=amplitudes,
            phases=phases,
            ch=ch,
            ramp_type=ramp_type,
            loops=loops,
            modulation_frequencies=modulation_frequencies,
            modulation_amplitudes=modulation_amplitudes,
            modulation_phases=modulation_phases,
            mask=mask,
            detuning=detuning,
            detuning_shift=detuning_shift,
            remove_self_interaction=remove_self_interaction,
        )
        return t


    def check_waveform_parameters(self, t, duration, ch, loops, modulation_amplitudes):
        if self.sample_data.mode == b"Off":
            raise LabscriptError(
                "Card has not been properly initialized. Please call set_mode() first."
            )
        if not ch in self.enabled_ports:
            raise LabscriptError(
                "Waveform instruction on disabled channel. Please enable channel {} in set_mode() first.".format(
                    ch
                )
            )
        if t < 0:
            raise LabscriptError("Time t cannot be negative")
        if duration <= 0:
            raise LabscriptError("Duration must be positive")
        if duration > 100e-3:
            warnings.warn(
                "Duration of waveform is very long. You may run out of memory on the experiment control computer or on the Spectrum card.\n"
            )
        if duration > self.max_duration:
            raise LabscriptError(
                "Waveform duration exceeds card memory ({0} s)".format(
                    self.max_duration
                )
            )
        if loops > 2**20 - 1:
            raise LabscriptError(
                "Too many loops requested. Number of loops must be less than 2^20 = 1,048,576"
            )
        if len(modulation_amplitudes) > 0:
            if (
                np.max(np.abs(modulation_amplitudes)) > 1
                or np.round(np.sum(np.abs(modulation_amplitudes)), decimals=4) > 1
            ):
                raise LabscriptError(
                    "Magintude of total modulation amplitude must be less than 1."
                )
        return True

    def sweep_comb(
        self,
        t,
        duration,
        start_freqs,
        end_freqs,
        amplitudes,
        phases,
        ch,
        ramp_type,
        loops=1,
        modulation_frequencies=[0],
        modulation_amplitudes=[0],
        modulation_phases=[0],
        mask=None,
        remove_self_interaction=False,
        detuning=0,
        detuning_shift=0
    ):
        """
        Fundamental function that allows a user to initialize a waveform.
        """

        #print("Waveform characteristics", start_freqs, end_freqs, amplitudes, ch, loops, modulation_frequencies, modulation_amplitudes, modulation_phases)

        # Check for common problems in a waveform
        self.check_waveform_parameters(t, duration, ch, loops, modulation_amplitudes)

        # Convert from time in seconds to time in sample chunks (1 sample chunk = 32 samples)
        t_start_c, delta_start = st.time_s_to_c(
            t, self.sample_data.clock_freq, extend=False
        )
        # extend duration to fill chunks, not truncate
        t_end_c, delta_end = st.time_s_to_c(
            duration, self.sample_data.clock_freq, extend=True
        )
        duration_c = t_end_c

        #print(delta_start, delta_end)

        if loops > 1:
            # TODO: should this be here or in stop()? here is not mode dependent, but stop() can put it under sequence condition
            if duration_c < self.duration_min_c:
                duration_min_ns = (
                    1e9
                    * self.samplesPerChunk
                    * self.duration_min_c
                    / self.sample_data.clock_freq
                )  # nanoseconds
                raise LabscriptError(
                    "Periodic waveforms must be longer than {} ns".format(
                        duration_min_ns
                    )
                )

            if (delta_start != 0) or (delta_end != 0):
                print(
                    "Warning: periodic waveform does not fill an integer number of chunks. "
                    "Waveform will be extended, not zero padded. "
                    "Loops will be corrected to preserve loops*duration."
                )
                print(f"delta_start: {delta_start}, delta_end: {delta_end}")
                delta_s = delta_start + delta_end  # samples
                duration_s = st.time_s_to_sa(
                    duration, self.sample_data.clock_freq
                )  # samples
                loops = int(np.round(loops / (1 + (delta_s / duration_s))))

                delta_start = 0
                delta_end = 0

        #print("Waveform characteristics", ch, loops, modulation_frequencies, modulation_amplitudes, modulation_phases)

        wvf = waveform(
            t_start_c,
            duration_c,
            ch,
            loops,
            is_periodic=(loops > 1),
            delta_start=delta_start,
            delta_end=delta_end,
            modulation_frequencies=modulation_frequencies,
            modulation_amplitudes=modulation_amplitudes,
            modulation_phases=modulation_phases,
            mask=mask,
            remove_self_interaction=remove_self_interaction,
        )

        assert len(start_freqs) == len(
            end_freqs
        ), "Start and End frequencies must be same length"
        assert len(phases) == len(
            amplitudes
        ), "Phase and Amplitude must have same length"
        assert len(phases) == len(
            start_freqs
        ), "Phase and Frequencies must have same length"


        for i in range(len(start_freqs)):
            if (np.min(amplitudes[i]) < 0) or (np.max(amplitudes[i]) > 1):
                raise LabscriptError(
                    "Amplitude["
                    + str(i)
                    + "] = "
                    + str(amplitudes[i])
                    + " is outside the allowed range [0,1]"
                )

            if remove_self_interaction:

                ts = np.arange(
                    0,
                    st.time_c_to_s(wvf.duration, self.sample_data.clock_freq),
                    1 / self.sample_data.clock_freq,
                )

                if wvf.has_mask:
                    # if we have a mask, then we ignore the waveform modulation parameters
                    modulation_waveform = wvf.mask(ts)
                    wvf.modulation_frequencies = [0]
                    wvf.modulation_amplitudes = [0]
                    wvf.modulation_phases = [0]
                else:
                    modulation_waveform = np.sum(
                        [
                            amp * np.cos(2 * np.pi * (freq * ts) + phase * np.pi / 180)
                            for freq, amp, phase in zip(
                                wvf.modulation_frequencies,
                                wvf.modulation_amplitudes,
                                wvf.modulation_phases,
                            )
                        ],
                        axis=0,
                    )
                
                # Merrick modification 15th July 2024 to check why +/- onmsite affects edge states
                if test:=False:
                    fast_switch_wvf = np.cos(2*np.pi*400e3*ts) # assume 200us bloch period
                    zero_crossings = np.where(np.diff(np.sign(fast_switch_wvf)))[0]
                else:
                    # zero_crossings = np.where(np.diff(np.sign(modulation_waveform)))[0]

                    # 2026/02/24 change to fix issue where if mod waveform has actual 0s
                    zero_crossings = np.where(np.diff(np.signbit(modulation_waveform)))[0]

                detuning_jumps = np.array(
                    [0, *zero_crossings, len(modulation_waveform) - 1]
                )
                if len(zero_crossings) > 0:
                    if zero_crossings[0] == 0:
                        detuning_jumps = detuning_jumps[1:]

                # print("Removing self interactions")


                # mod_freq = modulation_frequencies[0]
                # mod_period = 1 / mod_freq
                # mod_halfperiod = mod_period/2
                # # Changed to ceil as floor was causing errors
                # nsegments = int(np.ceil(duration / mod_period))

                # dsign = 1
                for n, j in enumerate(detuning_jumps[:-1]):
                    # Question: is this the best way of getting the sign? Could be broken if it's a zero
                    # at j+1, for example.
                    dsign = np.sign(modulation_waveform[j + 1])

                    # dsign *= -1

                    wvf.add_pulse(
                        start_freqs[i] + dsign * detuning + detuning_shift,
                        end_freqs[i] + dsign * detuning + detuning_shift,
                        ts[detuning_jumps[n + 1]] - ts[detuning_jumps[n]],
                        phases[i],
                        amplitudes[i],
                        ramp_type,
                        painting_function=None,
                    )
                    # wvf.add_pulse(
                    #     start_freqs[i]-detuning, end_freqs[i]-detuning,
                    #     mod_halfperiod, phases[i], amplitudes[i], ramp_type,
                    #     painting_function=None
                    # )
                
            else:

                wvf.add_pulse(
                    start_freqs[i] + detuning + detuning_shift,
                    end_freqs[i] + detuning + detuning_shift,
                    duration,
                    phases[i],
                    amplitudes[i],
                    ramp_type,
                    painting_function=None,
                )

        self.raw_waveforms.append(wvf)
        return t + (loops * duration)

    def painted_sweep(
        self,
        t,
        duration,
        amplitude,
        painting_function,
        ch,
        phase=0,
        painting_freq=False,
        painting_list=False,
        loops=1,
    ):
        """
        Basic function for generating a painted potential.

        Parameters
        ----------
        t: float (s)
            The time at which the painting begins
        duration: float (s)
            The length of the painting
        amplitude: float
            Amplitude of waveform, must be between 0, 1.
        painting_function: function (float -> float)
            Function generating frequency (Hz) as a function of t(s)
        painting_freq: bool
            Is this function specifiying frequency(time)?  Default is phase(time)
        loops: int
            Number of times to play this sequence.


        Returns
        --------
        t_final: float (s)
            The time at which the sequence ends (including loops)
        """
        # Check timing parameters
        self.check_waveform_parameters(t, duration, ch, loops, modulation_amplitudes=[])
        t_start_c, delta_start = st.time_s_to_c(
            t, self.sample_data.clock_freq, extend=False
        )
        # extend duration to fill chunks, not truncate
        t_end_c, delta_end = st.time_s_to_c(
            duration, self.sample_data.clock_freq, extend=True
        )
        # This is bad and should be moved to its own function.  However, in the interests of testing we won't make that change for now.
        duration_c = t_end_c
        if loops > 1:
            # TODO: should this be here on in stop()? here is not mode dependent, but stop() can put it under sequence condition
            if duration_c < self.duration_min_c:
                duration_min_ns = (
                    1e9
                    * self.samplesPerChunk
                    * self.duration_min_c
                    / self.sample_data.clock_freq
                )  # nanoseconds
                raise LabscriptError(
                    "Periodic waveforms must be longer than {} ns".format(
                        duration_min_ns
                    )
                )

            if (delta_start != 0) or (delta_end != 0):
                print(
                    "Warning: periodic waveform does not fill an integer number of chunks. Painting waveform will be extended, not zero padded. Loops will be corrected to preserve loops*duration."
                )
                print(
                    "Be careful!  Painting has not been tested robustly and there could be phase jumps etc."
                )
                print(f"delta_start: {delta_start}, delta_end: {delta_end}")
                delta_s = delta_start + delta_end  # samples
                duration_s = st.time_s_to_sa(
                    duration, self.sample_data.clock_freq
                )  # samples
                loops = int(np.round(loops / (1 + (delta_s / duration_s))))

                delta_start = 0
                delta_end = 0
        assert (amplitude >= 0) and (amplitude <= 1), "Amplitude must be in [0, 1]"
        wvf = waveform(
            t_start_c,
            duration_c,
            ch,
            loops,
            is_periodic=(loops > 1),
            delta_start=delta_start,
            delta_end=delta_end,
        )

        wvf.add_pulse(
            start_freq=0,
            end_freq=0,
            ramp_time=duration,
            phase=phase,
            amplitude=amplitude,
            ramp_type="None",
            painting_function=painting_function,
            painting_freq=painting_freq,
            painting_list=painting_list,
        )

        self.raw_waveforms.append(wvf)

        return t + (loops * duration)

    def generate_code(self, hdf5_file):
        """
        Loads profile table containing data into h5 file using a hierarchical
        data structure.
        """
        print("Starting generation")
        ss = time.time()
        device = hdf5_file.create_group("/devices/" + self.name)

        # Store device settings
        str_dt = h5py.special_dtype(vlen=str)
        settings_dtypes = np.dtype(
            [
                ("mode", "S10"),
                ("clock_freq", np.float),
                ("use_ext_clock", np.int),
                ("ext_clock_freq", np.float),
                ("export_data", np.bool),
                ("export_path", str_dt),
            ]
        )
        settings_table = np.array((0, 0, 0, 0, 0, 0), dtype=settings_dtypes)
        settings_table["mode"] = self.sample_data.mode
        settings_table["clock_freq"] = self.sample_data.clock_freq
        settings_table["use_ext_clock"] = self.use_ext_clock
        settings_table["export_data"] = self.export_data
        settings_table["export_path"] = self.export_path
        settings_table["ext_clock_freq"] = self.ext_clock_freq
        device.create_dataset("device_settings", data=settings_table)

        # Store channel settings
        channel_dtypes = [("power", np.float), ("name", "S10"), ("port", int)]
        channel_table = np.zeros(len(self.sample_data.channels), dtype=channel_dtypes)
        for i, channel in enumerate(self.sample_data.channels):
            channel_table[i]["power"] = channel.power
            channel_table[i]["name"] = channel.name
            channel_table[i]["port"] = channel.port
        device.create_dataset("channel_settings", data=channel_table)

        # Store waveform groups
        g = device.create_group("waveform_groups")
        dill_function_type = h5py.special_dtype(vlen=str)

        for i, group in enumerate(self.sample_data.waveform_groups):
            g_start = time.time()
            group_folder = g.create_group("group " + str(i))
            settings_dtypes = [
                ("time", np.int),
                ("duration", np.int),
                ("loops", np.int),
            ]
            settings_table = np.array((0, 0, 0), dtype=settings_dtypes)
            settings_table["time"] = group.time
            settings_table["duration"] = group.duration
            settings_table["loops"] = group.loops
            group_folder.create_dataset("group_settings", data=settings_table)

            if group.duration == 0:
                raise LabscriptError(
                    "Something went wrong in preparing waveform data. Group duration is 0"
                )

            # Store waveforms
            for wvf in group.waveforms:
                name = (
                    "Waveform: ch = "
                    + str(wvf.port)
                    + ", t = "
                    + str(wvf.time)
                    + ", dur = "
                    + str(wvf.duration)
                )
                if (
                    name in group_folder
                ):  # If waveform already exists, add to already created group
                    grp = group_folder[name]
                else:
                    grp = group_folder.create_group(name)
                    profile_dtypes = [
                        ("time", int),
                        ("duration", int),
                        ("loops", int),
                        ("port", int),
                        ("sample_start", int),
                        ("sample_end", int),
                        ("delta_start", int),
                        ("delta_end", int),
                        (
                            "modulation_frequencies",
                            float,
                            len(wvf.modulation_frequencies),
                        ),
                        (
                            "modulation_amplitudes",
                            float,
                            len(wvf.modulation_amplitudes),
                        ),
                        ("modulation_phases", float, len(wvf.modulation_phases)),
                        ("mask", dill_function_type),
                        ("has_mask", bool),
                        ("remove_self_interaction", bool),
                    ]
                    profile_table = np.zeros(1, dtype=profile_dtypes)
                    profile_table["time"] = wvf.time
                    profile_table["duration"] = wvf.duration
                    profile_table["loops"] = wvf.loops
                    profile_table["port"] = wvf.port
                    profile_table["sample_start"] = wvf.sample_start
                    profile_table["sample_end"] = wvf.sample_end
                    profile_table["delta_start"] = wvf.delta_start
                    profile_table["delta_end"] = wvf.delta_end
                    profile_table["modulation_frequencies"] = wvf.modulation_frequencies
                    profile_table["modulation_amplitudes"] = wvf.modulation_amplitudes
                    profile_table["modulation_phases"] = wvf.modulation_phases
                    profile_table["mask"] = str(pickle.dumps(wvf.mask, recurse=True))
                    profile_table["has_mask"] = wvf.has_mask
                    profile_table[
                        "remove_self_interaction"
                    ] = wvf.remove_self_interaction
                    grp.create_dataset("waveform_settings", data=profile_table)

                if wvf.duration == 0:
                    raise LabscriptError(
                        "Something went wrong in preparing waveform data. Waveform duration is 0."
                    )

                # Store pulses
                profile_dtypes = [
                    ("start_freq", np.float),
                    ("end_freq", np.float),
                    ("ramp_time", np.float),
                    ("phase", np.float),
                    ("amp", np.float),
                    ("ramp_type", "S10"),
                    ("painting_function", dill_function_type), 
                    ("painting_freq", np.bool),
                    ("painting_list", np.bool),
                    ("pulse_str", "S20"),
                ]
                profile_table = np.zeros(len(wvf.pulses), dtype=profile_dtypes)

                if len(wvf.pulses) == 0:
                    raise LabscriptError(
                        "Something went wrong in generating Spectrum card data: waveform does not have any pulse data"
                    )

                for j, pulse in enumerate(wvf.pulses):
                    profile_table["start_freq"][j] = pulse.start
                    profile_table["end_freq"][j] = pulse.end
                    profile_table["ramp_time"][j] = pulse.ramp_time
                    profile_table["phase"][j] = pulse.phase
                    profile_table["amp"][j] = pulse.amp
                    profile_table["ramp_type"][j] = pulse.ramp_type
                    # write function to h5 file
                    pickled_function = repr(
                        pickle.dumps(pulse.painting_function, recurse=True)
                    )
                    # if pulse.is_painted:
                    profile_table["painting_function"] = pickled_function
                    profile_table["painting_freq"][j] = pulse.painting_freq
                    profile_table["painting_list"][j] = pulse.painting_list
                    profile_table["pulse_str"][j] = str(pulse)
                # If waveform already has associated data, add to the existing dataset.
                if "pulse_data" in grp:
                    d = grp["pulse_data"]
                    d.resize((d.shape[0] + profile_table.shape[0]), axis=0)
                    d[-profile_table.shape[0] :] = profile_table
                else:
                    grp.create_dataset(
                        "pulse_data",
                        maxshape=(1000,),
                        data=profile_table,
                        dtype=profile_dtypes,
                        chunks=True,
                    )
            g_end = time.time()
            print(f"Group {i} took {g_end - g_start} s")
        ee = time.time()
        print(f"Time to generate: {ee - ss}")

    def stop(self):
        self.check_channel_collisions(self.raw_waveforms)

        if self.sample_data.mode == b"sequence":
            print("Spectrum card is in sequence mode")

            periodicWvfs = list(
                [k for k in self.raw_waveforms if k.is_periodic == True]
            )
            nonPeriodicWvfs = list(
                [k for k in self.raw_waveforms if k.is_periodic == False]
            )
            # Make nonperiodic groups
            nonPeriodicWvfGroups = self.make_waveform_groups(nonPeriodicWvfs)

            # Zero pad and combine groups that are too short
            nonPeriodicWvfGroups = self.pad_groups_sequence_mode(nonPeriodicWvfGroups)

            # Combine periodic and nonperiodic groups
            self.sample_data.waveform_groups = self.combine_periodic_nonperiodic_groups(
                periodicWvfs, nonPeriodicWvfGroups
            )

            # Sort groups in time order (just in case)
            self.sample_data.waveform_groups = sorted(
                self.sample_data.waveform_groups, key=lambda k: k.time
            )

            # Initial trigger
            self.triggerDO.go_high(0)
            self.triggerDO.go_low(self.triggerDur)

        elif self.sample_data.mode == b"Off":
            print("Spectrum card is OFF")
        elif self.sample_data.mode in self.mode_dict:
            raise LabscriptError(
                "{} mode not implemented".format(self.sample_data.mode)
            )
        else:
            raise LabscriptError("Invalid mode.")
        return

    def make_waveform_groups(self, waveforms):
        """
        Organize an array of waveforms into groups of overlapping waveforms
        """
        # Distinguish edge cases where times are identical
        waveforms = sorted(waveforms, key=lambda k: k.time)

        # List of flags marking start and end times of waveform pieces
        # {t,1} marks the beginning of a piece at time t
        # {t,-1} marks the end of a piece at time t
        flagAddRemoveWvf = []
        for waveform in waveforms:
            flagAddRemoveWvf.append({"t": waveform.time, "flag": 1})
            flagAddRemoveWvf.append(
                {"t": waveform.time + waveform.loops * waveform.duration, "flag": -1}
            )

        flagAddRemoveWvf = sorted(flagAddRemoveWvf, key=lambda k: k["t"])

        # Find the times at which groups of pieces begin and end
        # Groups start when the sum of flags at time t changes from 0 to more than 0
        # Groups end when the sum of flags at time t hits 0
        numOverlaps = 0
        groupStartIndices = []
        groupEndIndices = []
        for i in range(len(flagAddRemoveWvf)):
            nextNumOverlaps = numOverlaps + flagAddRemoveWvf[i]["flag"]

            if (numOverlaps < 0) or (nextNumOverlaps < 0):
                raise LabscriptError(
                    "Something went wrong in make_waveform_groups(): numOverlaps should never be negative"
                )

            if numOverlaps == 0:
                groupStartIndices.append(i)

            if nextNumOverlaps == 0:
                groupEndIndices.append(i)

            numOverlaps = nextNumOverlaps

        if len(groupStartIndices) != len(groupEndIndices):
            raise LabscriptError(
                "Something went wrong in make_waveform_groups(): length of groupStartIndices should be equal to length of groupEndIndices"
            )

        groups = []
        totalWvfs = 0
        for i in range(len(groupStartIndices)):
            t0 = flagAddRemoveWvf[groupStartIndices[i]]["t"]
            t1 = flagAddRemoveWvf[groupEndIndices[i]]["t"]

            wvfsInGroup = list(
                [
                    k
                    for k in waveforms
                    if (k.time >= t0) and (k.time + k.loops * k.duration <= t1)
                ]
            )
            totalWvfs += len(wvfsInGroup)
            groups.append(waveform_group(t0, t1 - t0, wvfsInGroup))

        if totalWvfs != len(waveforms):
            raise LabscriptError(
                "Something went wrong in make_waveform_groups(): totalWvfs after grouping should be equal to total number of waveforms"
            )
            return

        return groups

    def pad_groups_sequence_mode(self, groups):
        """
        Given a list of groups, this function deterines which groups are shorter
        than the minimum segment size for sequence mode and either zero pads
        them or combines them with a later group
        """

        n_groups = len(groups)
        groups_padded = []
        skip_group = False

        duration_min_sa = self.duration_min_c * self.samplesPerChunk
        duration_min_ns = (
            1e9
            * self.samplesPerChunk
            * self.duration_min_c
            / self.sample_data.clock_freq
        )  # nanoseconds

        for idx, group in enumerate(groups):
            if skip_group:
                skip_group = False
                continue

            if group.duration < self.duration_min_c:
                print("Group duration is too short...")
                if idx < n_groups - 1:
                    next_group = groups[idx + 1]
                    delta_time = next_group.time - (group.time + group.duration)
                else:
                    next_group = None
                    delta_time = np.inf

                if delta_time < self.duration_min_c:  # combine the groups
                    print("... it will be combined with the following group.")

                    combined_waveforms = []
                    combined_waveforms.extend(group.waveforms)
                    combined_waveforms.extend(next_group.waveforms)
                    combined_waveforms = sorted(
                        combined_waveforms, key=lambda k: k.time
                    )

                    start = group.time
                    duration = next_group.time + next_group.duration - start
                    combined_group = waveform_group(start, duration, combined_waveforms)

                    groups_padded.append(combined_group)

                    # must skip the next group since it has just been combined with the current group
                    skip_group = True

                else:  # extend the group
                    print(
                        "... it will be zero padded to {} samples ( = {} ns). Subsequent segments (including any loops of current segment) may occur at the wrong time.".format(
                            duration_min_sa, duration_min_ns
                        )
                    )
                    group.duration = self.duration_min_c
                    groups_padded.append(group)

            else:
                groups_padded.append(group)

        # repeat recursively to make sure all groups are long enough
        if len(groups_padded) < len(groups):
            groups_padded = self.pad_groups_sequence_mode(groups_padded)

        return groups_padded

    def check_channel_collisions(self, waveforms):
        """
        Check for channel collisions since a single channel can't do multiple
        things at once
        """
        for ch in self.sample_data.channels:
            port = ch.port
            wvfsPerPort = list([k for k in waveforms if k.port == port])

            groupsPerPort = self.make_waveform_groups(wvfsPerPort)
            for group in groupsPerPort:
                waveforms = group.waveforms
                n_waveforms = len(waveforms)

                if n_waveforms > 1:
                    waveforms = sorted(
                        waveforms,
                        key=lambda k: k.time * self.samplesPerChunk + k.delta_start,
                    )

                    for idx in range(n_waveforms - 1):
                        wvf_0 = waveforms[idx]
                        wvf_1 = waveforms[idx + 1]

                        t_0 = (
                            wvf_0.time + wvf_0.duration
                        ) * self.samplesPerChunk - wvf_0.delta_end
                        t_1 = wvf_1.time * self.samplesPerChunk + wvf_1.delta_start
                        if t_0 > t_1:
                            print(group)
                            raise LabscriptError(
                                "Port collision: you've instructed port {} to play two waveforms at once.".format(
                                    port
                                )
                            )

    def combine_periodic_nonperiodic_groups(self, periodicWvfs, nonPeriodicWvfGroups):
        """
        If part of a periodic waveform overlaps with a nonperiodic group, then
        add this section of the waveform to the group. Then add the rest of the
        periodic waveform as a looping group.
        """

        result_groups = []

        # Add a dummy group so we can automatically take care of the final gap
        nonPeriodicWvfGroups.append(None)

        for i, group in enumerate(nonPeriodicWvfGroups):
            # Handle gap *before* this group (and the final gap)
            if len(nonPeriodicWvfGroups) == 1:  # No groups (only the dummy)
                t_start = float("-inf")
                t_end = float("inf")

            elif i == 0:  # First gap
                t_start = float("-inf")
                t_end = group.time

            elif group == None:  # Final gap
                prev_group = nonPeriodicWvfGroups[i - 1]
                t_start = prev_group.time + prev_group.duration
                t_end = float("inf")

            else:  # Middle gaps
                prev_group = nonPeriodicWvfGroups[i - 1]
                t_start = prev_group.time + prev_group.duration
                t_end = group.time

            if t_start != t_end:  # There is actually a gap between groups
                newWvfs = self.split_periodic_waveforms(periodicWvfs, t_start, t_end)

                if len(newWvfs) > 0:
                    newGroups = []
                    for wvf in newWvfs:
                        if type(wvf) == list:
                            newGrp = waveform_group(
                                wvf[0].time, wvf[0].duration, wvf, loops=wvf[0].loops
                            )
                        else:
                            newGrp = waveform_group(
                                wvf.time, wvf.duration, [wvf], loops=wvf.loops
                            )
                            wvf.loops = 1
                            wvf.is_periodic = False

                        newGroups.append(newGrp)

                    result_groups.extend(newGroups)

            # Handle the group itself
            if group != None:
                t_start = group.time
                t_end = group.time + group.duration

                newWvfs = self.split_periodic_waveforms(periodicWvfs, t_start, t_end)

                # Add the pieces of the split waveform to the group
                for wvf in newWvfs:
                    group.add_waveform(wvf)

                result_groups.append(group)

        return result_groups

    def split_periodic_waveforms(self, waveforms, t_start, t_end):
        """
        Extracts a section of a periodic waveform between t = (t_start,t_end)
        Result comes in (at most) 3 parts: a partial waveform starting at
        t_start, a number of full loops in the middle, and a partial waveform
        ending at t_end
        """

        result_waveforms = []
        overlappedWvfs = list(
            [
                k
                for k in waveforms
                if (k.time <= t_end) and (k.time + k.loops * k.duration >= t_start)
            ]
        )


        overlapGroups = self.make_waveform_groups(overlappedWvfs)
        for ogroup in overlapGroups:
            if len(ogroup.waveforms) > 1:
                unique_wvf_triplets = {
                    (i.time, i.loops, i.duration) for i in ogroup.waveforms
                }
                
                if len(unique_wvf_triplets) == 1:
                    result_waveforms.append(ogroup.waveforms)
                else:
                    for i in ogroup.waveforms:
                        print('This is just a separator')
                        print(i)
                    # TODO: can we accomodate this?
                    raise LabscriptError(
                        "Cannot deal with overlapped periodic waveforms. Please remove the overlapped waveforms."
                    )
            else:
                wvf = ogroup.waveforms[0]

                t0 = wvf.time
                t1 = wvf.time + wvf.loops * wvf.duration
                dur = wvf.duration

                t_start_p = max(t0, t_start)
                t_end_p = min(t1, t_end)

                # Start of the next loop immediately following t_start_p
                t_n = int(t0 + math.ceil(float(t_start_p - t0) / float(dur)) * dur)
                n_full_loops = int(math.floor(float(t_end_p - t_n) / float(dur)))


                if (
                    t_n > t_end_p
                ):  # Partial loop is completely inside the desired region
                    # Partial loop
                    waveform_partial = waveform(
                        t_start_p,
                        t_end_p - t_start_p,
                        wvf.port,
                        loops=1,
                        is_periodic=False,
                        pulses=wvf.pulses,
                    )
                    waveform_partial.sample_start = t_start_p - (t_n - dur)
                    waveform_partial.sample_end = (
                        waveform_partial.sample_start + t_end_p - t_start_p
                    )
                    result_waveforms.append(waveform_partial)

                else:
                    # Full middle loops
                    if n_full_loops > 0:
                        waveform_full_loops = waveform(
                            t_n,
                            dur,
                            wvf.port,
                            loops=n_full_loops,
                            is_periodic=(n_full_loops > 1),
                            pulses=wvf.pulses,
                        )
                        result_waveforms.append(waveform_full_loops)

                    # Partial start loop
                    if t_start_p != t_n:
                        dt_start = t_start_p - (t_n - dur)
                        waveform_partial_start = waveform(
                            t_start_p,
                            dur - dt_start,
                            wvf.port,
                            loops=1,
                            is_periodic=False,
                            pulses=wvf.pulses,
                        )
                        waveform_partial_start.sample_start = dt_start
                        waveform_partial_start.sample_end = dur
                        result_waveforms.append(waveform_partial_start)

                    # Partial end loop
                    if t_end_p != (t_n + n_full_loops * dur):
                        dt_end = t_end_p - (t_n + n_full_loops * dur)
                        waveform_partial_end = waveform(
                            t_n + (n_full_loops * dur),
                            dt_end,
                            wvf.port,
                            loops=1,
                            is_periodic=False,
                            pulses=wvf.pulses,
                        )
                        waveform_partial_end.sample_start = 0
                        waveform_partial_end.sample_end = dt_end
                        result_waveforms.append(waveform_partial_end)

        return result_waveforms


@BLACS_tab
class SpectrumTab(DeviceTab):
    def initialise_GUI(self):
        # GUI value parameters
        self.base_units = {"freq": "MHz", "Power": "dBm", "phase": "Degrees"}
        self.base_min = {"freq": 0.001, "Power": -11.9, "phase": 0}
        self.base_max = {"freq": 4000.0, "Power": 13.5, "phase": 360}
        self.base_step = {"freq": 1.0, "Power": 1.0, "phase": 1}
        self.base_decimals = {"freq": 4, "Power": 4, "phase": 3}

        # Create DDS Output objects
        RF_prop = {}
        RF_prop["channel 0"] = {}
        for subchnl in ["freq", "Power", "phase"]:
            RF_prop["channel 0"][subchnl] = {
                "base_unit": self.base_units[subchnl],
                "min": self.base_min[subchnl],
                "max": self.base_max[subchnl],
                "step": self.base_step[subchnl],
                "decimals": self.base_decimals[subchnl],
            }

        # Create the output objects
        self.create_dds_outputs(RF_prop)
        # Create widgets for output objects
        dds_widgets, ao_widgets, do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("RF Output", dds_widgets))

        # Create and set the primary worker
        self.instance = (
            self.settings["connection_table"]
            .find_by_name(self.device_name)
            .BLACS_connection
        )
        self.create_worker("main_worker", SpectrumWorker, {"instance": self.instance})
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(False)
        self.supports_smart_programming(False)


class SpectrumWorker(Worker):
    def init(self):
        self.final_values = {"channel 0": {}}

        card_address = self.instance

        if type(card_address) == str:
            card_address = str.encode(card_address)

        self.card = sp.spcm_hOpen(ctypes.create_string_buffer(card_address))
        if self.card == None:
            raise LabscriptError("Device is not connected.")

        max_channels = sp.int32(0)
        sp.spcm_dwGetParam_i32(
            self.card, sp.SPC_MIINST_CHPERMODULE, sp.byref(max_channels)
        )
        self.max_channels = max_channels.value

        self.samplesPerChunk = 32
        self.bytesPerSample = 2

        self.previous_settings = None
        self.pulse_dictionary = {}

    def card_settings(self):
        # Close the card if it's already open
        if self.card != 1:
            print("Closing card")
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_STOP)
            sp.spcm_vClose(self.card)

        card_address = self.instance

        if type(card_address) == str:
            card_address = str.encode(card_address)

        print("Opening card")
        self.card = sp.spcm_hOpen(ctypes.create_string_buffer(card_address))
        sp.spcm_dwSetParam_i32(self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_RESET)

        # General settings -- mode specific settings are defined in transition_to_buffered
        if self.use_ext_clock == True:
            # clock mode external PLL
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_CLOCKMODE, sp.SPC_CM_EXTREFCLOCK)
            sp.spcm_dwSetParam_i32(
                self.card, sp.SPC_REFERENCECLOCK, self.ext_clock_freq
            )
        else:
            # clock mode internal PLL
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_CLOCKMODE, sp.SPC_CM_INTPLL)

        # --- PLL lock check: apply clock settings now so a lock failure surfaces here ---
        try:
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_WRITESETUP)
        except Exception as e:
            if "CLOCKNOTLOCKED" in str(e):
                raise LabscriptError(
                    "Spectrum card reference clock failed to lock — check external "
                    "reference signal amplitude/frequency at the REF input."
                )
            raise

        sp.spcm_dwSetParam_i32(self.card, sp.SPC_SAMPLERATE, sp.int32(self.clock_freq))
        sp.spcm_dwSetParam_i32(self.card, sp.SPC_TRIG_EXT0_MODE, sp.SPC_TM_POS) # was EXT1
        sp.spcm_dwSetParam_i32(self.card, sp.SPC_TRIG_ORMASK, sp.SPC_TMASK_EXT0) # was EXT1

        self.mode_dict = {
            b"single": sp.SPC_REP_STD_SINGLE,
            b"multi": sp.SPC_REP_STD_MULTI,
            b"gate": sp.SPC_REP_STD_GATE,
            b"single_restart": sp.SPC_REP_STD_SINGLERESTART,
            b"sequence": sp.SPC_REP_STD_SEQUENCE,
            b"fifo_single": sp.SPC_REP_FIFO_SINGLE,
            b"fifo_multi": sp.SPC_REP_FIFO_MULTI,
            b"fifo_gate": sp.SPC_REP_FIFO_GATE,
        }

        if self.mode in self.mode_dict:
            sp.spcm_dwSetParam_i32(
                self.card, sp.SPC_CARDMODE, self.mode_dict[self.mode]
            )
        else:
            raise LabscriptError("Invalid operating mode.")

    def check_remote_values(self):
        results = {"channel 0": {}}
        results["channel 0"]["freq"] = 0
        results["channel 0"]["Power"] = 0
        results["channel 0"]["phase"] = 0
        return results

    def program_manual(self, front_panel_values):
        return self.check_remote_values()

    def generate_buffer(self):
        """
        Uses class structure to generate the necessary buffer to be sent to the
        Spectrum Card. How the buffer is organized is dependent on the mode
        being used. In single mode, there is a single segment, but with the
        possibility of looping over that segment. In multimode, segments are
        added in a row, with zero padding where necessary. To handle exceptions,
        the function returns False if a buffer was not generated, so as not to
        send this information to the card. Otherwise, the function returns True.
        """

        print("Generating buffer")
        start_time = time.time()
        # Iterate over the channels which are on. Set channel-specific.

        channel_enable_word = 0

        channel_dicts = {
            0: {
                "ch_mask": sp.CHANNEL0,
                "amp_address": sp.SPC_AMP0,
                "enable_address": sp.SPC_ENABLEOUT0,
            },
            1: {
                "ch_mask": sp.CHANNEL1,
                "amp_address": sp.SPC_AMP1,
                "enable_address": sp.SPC_ENABLEOUT1,
            },
            2: {
                "ch_mask": sp.CHANNEL2,
                "amp_address": sp.SPC_AMP2,
                "enable_address": sp.SPC_ENABLEOUT2,
            },
            3: {
                "ch_mask": sp.CHANNEL3,
                "amp_address": sp.SPC_AMP3,
                "enable_address": sp.SPC_ENABLEOUT3,
            },
        }

        for channel in self.channels:
            # Setting amplitude corresponding to chosen power.
            amplitude = int(np.sqrt(0.1) * 10 ** (float(channel.power) / 20.0) * 1000)
            if amplitude < 80:
                raise LabscriptError(
                    "Power below acceptable range. Min power = -11.94 dBm"
                )
            if amplitude > 2500:
                raise LabscriptError(
                    "Power above acceptable range. Max power = 17.96 dBm."
                )

            cd = channel_dicts[channel.port]

            channel_enable_word |= cd["ch_mask"]

            sp.spcm_dwSetParam_i32(self.card, cd["amp_address"], sp.int32(amplitude))
            sp.spcm_dwSetParam_i32(self.card, cd["enable_address"], 1)

        sp.spcm_dwSetParam_i32(
            self.card, sp.SPC_CHENABLE, sp.int32(channel_enable_word)
        )


        #### SEQUENCE MODE ####
        if self.mode == b"sequence":
            # Sort groups in time order (just in case)
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Define shortest group time in chunks:
            self.duration_min_c = int(12 / self.num_chs)

            # Add dummy sequences between waveform groups
            dummy_groups = []
            self.sequence_instrs = []

            # Add main dummy loop segment to beginning of stack
            # If dummy segments are too short, we can't do enough loops to last for many seconds of
            # downtime (we are limited to at most 2^20 loops). So we must loop over longer dummy segments
            # length in segment chunks (1 chunk = 32 samples)
            dummy_loop_dur = 1024
            if dummy_loop_dur < self.duration_min_c:
                raise LabscriptError(
                    "Dummy segment duration must be longer than {} chunks".format(
                        self.duration_min_c
                    )
                )

            dummy_groups.append(waveform_group(float("-inf"), dummy_loop_dur, "dummy"))

            # Add leading dummy groups and generate sequence instructions
            cur_step = 0
            cur_segm = 1  # segm 0 is the dummy loop

            for idx, group in enumerate(self.waveform_groups):
                # t0 = end of previous group, t1 = start of this group
                if idx == 0:
                    t0 = 0
                else:
                    prev_group = self.waveform_groups[idx - 1]
                    t0 = prev_group.time + prev_group.loops * prev_group.duration

                t1 = group.time

                # Play leading dummy segment
                dummy_dur = t1 - t0

                # Only create the dummy group if there is a gap between
                # where the previous group ends and the current group
                if dummy_dur > 0:
                    n_loops = int(math.floor(float(dummy_dur) / float(dummy_loop_dur)))
                    leftover = dummy_dur - dummy_loop_dur * n_loops

                    if n_loops > 0:
                        # Send card to segment 0
                        self.sequence_instrs.append(
                            sequence_instr(cur_step, cur_step + 1, 0, n_loops)
                        )
                        cur_step += 1

                    if leftover > self.duration_min_c:
                        # Send card to the 'leftover' dummy segment
                        self.sequence_instrs.append(
                            sequence_instr(cur_step, cur_step + 1, cur_segm, 1)
                        )
                        cur_step += 1
                        cur_segm += 1
                        dummy_groups.append(
                            waveform_group(
                                t0 + dummy_loop_dur * n_loops, leftover, "dummy"
                            )
                        )

                    else:
                        group.time -= leftover  # Extend group backward
                        group.duration += leftover  # Keep end time the same

                # Play group segment
                self.sequence_instrs.append(
                    sequence_instr(cur_step, cur_step + 1, cur_segm, group.loops)
                )
                cur_step += 1
                cur_segm += 1

            # Loop the sequence back to the zeroth step
            self.sequence_instrs[len(self.sequence_instrs) - 1].next_step = 0

            # Merge and sort groups in time order
            self.waveform_groups.extend(dummy_groups)
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Get rid of the '-inf's that we used earlier
            # And check for zero-duration groups
            for group in self.waveform_groups:
                group.time = max(group.time, 0)

                if group.duration == 0:
                    raise LabscriptError(
                        "Something went wrong in preparing waveform data. Group duration is 0"
                    )

            # Print debugging for checking times and durations:
            # st.check_groups_and_instructions(self.waveform_groups, self.sequence_instructions, self.clock_freq, dummy_loop_dur)

            samples_per_chunk = self.samplesPerChunk
            bytes_per_sample = self.bytesPerSample

            # Split memory into segments
            num_segments = len(self.waveform_groups)
            num_segments = int(2 ** math.ceil(math.log(num_segments, 2)))
            sp.spcm_dwSetParam_i32(
                self.card, sp.SPC_SEQMODE_MAXSEGMENTS, sp.int32(num_segments)
            )
            # start on dummy segment
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_SEQMODE_STARTSTEP, 0)

            # Prepare data structure for saving segments, if desired
            if self.export_data:
                segments = {}
                #            print(f"Beginning to write segments: t = {time.time() - start_time}")
            # Write segments
            if self.precomputed:
                preloaded_data = np.load(
                    self.precomputed_output_name, allow_pickle=True
                ).item()
                print(preloaded_data)

            for seg_idx, group in enumerate(self.waveform_groups):
                buffer_size = int(
                    self.num_chs
                    * int(group.duration)
                    * samples_per_chunk
                    * bytes_per_sample
                )

                if buffer_size < 0:
                    raise LabscriptError(
                        "Buffer size is negative, indicating np.int32 overflow due "
                        "to type inheritance from group.duration"
                    )
                if buffer_size > 2**29:
                    raise LabscriptError(
                        "Buffer size is larger than 2**29, will cause memory error "
                        "when calling ctypes.create_string_buffer"
                    )

                pBuffer = ctypes.create_string_buffer(
                    buffer_size
                )  # should this be self.buffer?

                np_waveform = np.zeros(
                    self.num_chs * group.duration * samples_per_chunk, dtype=sp.int16
                )
                #                print(f"\t Filling buffer {seg_idx}: t = {time.time() - start_time}")
                # Fill buffer

                if group.waveforms != "dummy":
                    verbose = False
                    if self.precomputed:
                        print(f"Loading numpy waveform {seg_idx}")
                        np_waveform = preloaded_data[seg_idx]
                        # print(np_waveform_pc)
                    else:
                        for wvf_index, wvf in enumerate(group.waveforms):
                            t0 = st.time_c_to_s(
                                wvf.sample_start, self.clock_freq
                            ) + st.time_sa_to_s(wvf.delta_start, self.clock_freq)
                            t1 = st.time_c_to_s(
                                wvf.sample_end, self.clock_freq
                            ) - st.time_sa_to_s(wvf.delta_end, self.clock_freq)
                            dur = t1 - t0  # seconds
                            dur = (
                                st.time_c_to_s(wvf.duration, self.clock_freq)
                                - st.time_sa_to_s(wvf.delta_end, self.clock_freq)
                                - st.time_sa_to_s(wvf.delta_start, self.clock_freq)
                            )
                            t = np.arange(0, dur, 1 / self.clock_freq)

                            pulse_data = np.zeros(len(t))
                            pulse_data_temp = np.array([])
                            if verbose:
                                print(wvf_index)
                            if (
                                hash(wvf) in self.pulse_dictionary.keys()
                                and not wvf.has_mask
                            ):
                                if verbose:
                                    print("Found waveform")
                                pulse_data = self.pulse_dictionary[hash(wvf)]
                            else:
                                if wvf.remove_self_interaction:
                                    for pulse in wvf.pulses:
                                        # To be honest, I (Ocean, 2025/09/29) don't really know what method means
                                        # Basically, trying to run a modulated probe pulse in Sequence mode would consistently yield the error
                                        # "UnboundLocalError: local variable 'method' referenced before assignment"
                                        # As such, we copied this if-else block from the wvf.remove_self_interaction == False block
                                        # This made the error go away. 

                                        if pulse.ramp_type != b"static":  # ramping
                                            method = pulse.ramp_type
                                        else:  # static
                                            method = b"linear"
                                        tsegment = np.arange(
                                            0, pulse.ramp_time, 1 / self.clock_freq
                                        )
                                        c = chirp(
                                            tsegment,
                                            f0=pulse.start,
                                            t1=pulse.ramp_time,
                                            f1=pulse.end,
                                            method=method.decode(),
                                            phi=pulse.phase,
                                        )
                                        pulse_data_temp = np.append(pulse_data_temp, c)

                                    if len(pulse_data) > len(pulse_data_temp):
                                        pulse_data[: len(pulse_data_temp)] = (
                                            pulse_data_temp * (2**15 - 1) * pulse.amp
                                        )
                                    else:
                                        pulse_data = (
                                            pulse_data_temp[: len(pulse_data)]
                                            * (2**15 - 1)
                                            * pulse.amp
                                        )

                                else:
                                    for pulse in wvf.pulses:
                                        if pulse.painting_list:
                                            phase_values = (
                                                2
                                                * pi
                                                * np.cumsum(
                                                    pulse.painting_function(t)
                                                    * 1
                                                    / self.clock_freq,
                                                    axis=1,
                                                )
                                            )
                                            output = (
                                                np.sin(phase_values)
                                                / phase_values.shape[0]
                                            )
                                            c = np.sum(output, axis=0)
                                        elif pulse.is_painted:
                                            if pulse.painting_freq:
                                                phase_values = (
                                                    2
                                                    * np.pi
                                                    * np.cumsum(
                                                        pulse.painting_function(t)
                                                        * 1
                                                        / self.clock_freq
                                                    )
                                                )
                                            else:
                                                phase_values = (
                                                    2
                                                    * pi
                                                    / 360
                                                    * pulse.painting_function(t)
                                                )
                                            c = np.sin(phase_values)
                                        else:
                                            if pulse.ramp_type != b"static":  # ramping
                                                f1 = pulse.end
                                                method = pulse.ramp_type
                                            else:  # static
                                                f1 = pulse.start
                                                method = b"linear"

                                            if (
                                                pulse.start == f1
                                                and np.round(
                                                    int(f1 / 1e5) - f1 / 1e5, 5
                                                )
                                                == 0
                                                and dur > 100e-6
                                            ):
                                                print("Numba")
                                                loop_duration = 1 / np.gcd(
                                                    int(self.clock_freq), int(f1)
                                                )
                                                small_t = np.arange(
                                                    0,
                                                    loop_duration,
                                                    1 / self.clock_freq,
                                                )
                                                num_loops = int(dur / loop_duration)
                                                small_chirp_numba = numba_chirp(
                                                    small_t,
                                                    f0=pulse.start,
                                                    t1=loop_duration,
                                                    f1=f1,
                                                    phi=pulse.phase,
                                                )
                                                c2 = np.tile(small_chirp_numba, num_loops)
                                                if len(t) - len(c2) > 0:
                                                    print(
                                                        "You will end up with some problems here with phase loops."
                                                    )
                                                c = np.concatenate(
                                                    [c2, [0] * (len(t) - len(c2))]
                                                )
                                            else:
                                                c = chirp(
                                                    t,
                                                    f0=pulse.start,
                                                    t1=pulse.ramp_time,
                                                    f1=f1,
                                                    method=method.decode(),
                                                    phi=pulse.phase,
                                                )
                                                print(
                                                    f"\t\t\t Generated untiled chirp: t = {time.time() - start_time}"
                                                )

                                        pulse_data += pulse.amp * (2**15 - 1) * c

                                if wvf.remove_self_interaction:
                                    modulation_waveform = np.abs(
                                        np.sum(
                                            [
                                                amp
                                                * np.cos(
                                                    2 * np.pi * (freq * t)
                                                    + phase * np.pi / 180
                                                )
                                                for freq, amp, phase in zip(
                                                    wvf.modulation_frequencies,
                                                    wvf.modulation_amplitudes,
                                                    wvf.modulation_phases,
                                                )
                                            ],
                                            axis=0,
                                        )
                                    )

                                else:
                                    modulation_waveform = np.sum(
                                        [
                                            amp * (1 / 2
                                                * np.cos( 2 * np.pi * (freq * t) + phase  * np.pi/180
                                                )
                                                + 1 / 2
                                            )
                                            for freq, amp, phase in zip(
                                                wvf.modulation_frequencies,
                                                wvf.modulation_amplitudes,
                                                wvf.modulation_phases,
                                            )
                                        ],
                                        axis=0,
                                    )

                                total_modulation = np.max(
                                    modulation_waveform
                                )  # np.sum(wvf.modulation_amplitudes)
                                # Remove negative amplitudes
                                modulation_waveform = modulation_waveform - np.min(
                                    modulation_waveform
                                )  # + 1/2
                                # Scale top to be <= 1
                                modulation_waveform = (
                                    modulation_waveform / np.max(modulation_waveform)
                                    if np.max(modulation_waveform) > 0
                                    else modulation_waveform
                                )
                                # DC Offset if sum(amplitudes) < 1
                                modulation_waveform = (
                                    total_modulation * modulation_waveform
                                    + (1 - total_modulation)
                                )
                                #
                                #
                                # modulation_waveform = modulation_waveform + 1/2
                                modulation_waveform = amplitude_compensator(
                                    modulation_waveform
                                )
                                assert (
                                    np.min(modulation_waveform) >= 0
                                    and np.max(modulation_waveform) <= 1
                                ), "Modulation is too big"
                                assert (
                                    pulse_data.shape == modulation_waveform.shape
                                ), "Modulation waveform has wrong shape"
                                pulse_data = pulse_data * modulation_waveform
                                
                                mask_values = wvf.mask(
                                    t
                                )  # should we only do this if has mask for extra speed?

                                assert (
                                    np.min(mask_values) >= -1
                                ), f"Mask gets too small, {np.min(mask_values), np.max(mask_values)}"
                                assert np.max(mask_values) <= 1, "Mask gets too large"
                                assert (
                                    pulse_data.shape == mask_values.shape
                                ), "Mask must have same shape as waveform"

                                if wvf.remove_self_interaction:
                                    mask_values = np.abs(mask_values)

                                pulse_data = mask_values * pulse_data

                                if np.max(pulse_data) > (2**15 - 1):
                                    raise LabscriptError(
                                        "Maximum value of pulse_data exceeds 2**15-1, will overflow when cast to sp.int16"
                                    )

                                pulse_data = pulse_data.astype(sp.int16)
                                if dur > 200e-6 and not wvf.has_mask:
                                    self.pulse_dictionary[hash(wvf)] = pulse_data

                            # print(f"Pulse dictionary has size: {len(self.pulse_dictionary.keys())}")
                            if len(self.pulse_dictionary.keys()) > 100:
                                self.pulse_dictionary.clear()

                            port_index = sorted(self.enabled_ports).index(wvf.port)

                            begin = (
                                (wvf.time - group.time) * samples_per_chunk
                                + wvf.delta_start
                            ) * int(self.num_chs) + port_index
                            end = begin + (len(pulse_data) * int(self.num_chs))
                            increment = int(self.num_chs)
                            print(
                                f"Generated waveform {seg_idx}: t = {time.time() - start_time}, {len(np_waveform)}"
                            )

                            np_waveform[begin:end:increment] = pulse_data[
                                : len(np_waveform[begin:end:increment])
                            ]
                            if verbose:
                                print("comp: ", np_waveform, len(np_waveform))
                                print("pc: ", np_waveform_pc, len(np_waveform_pc))
                    # if self.precomputed:
                    #     assert all(np_waveform == np_waveform_pc)

                print(
                    f"\t Beginning data transfer {seg_idx}: t = {time.time() - start_time}"
                )
                ctypes.memmove(
                    pBuffer, np_waveform.ctypes.data_as(sp.ptr16), buffer_size
                )

                ##### Write buffer #####
                # Declare segment to change
                sp.spcm_dwSetParam_i32(
                    self.card, sp.SPC_SEQMODE_WRITESEGMENT, sp.int32(seg_idx)
                )

                # Define segment size
                #                print("SPC_SEQMODE_SEGMENTSIZE", group.duration * samples_per_chunk)
                sp.spcm_dwSetParam_i32(
                    self.card,
                    sp.SPC_SEQMODE_SEGMENTSIZE,
                    sp.int32(group.duration * samples_per_chunk),
                )

                # Define buffer
                sp.spcm_dwDefTransfer_i64(
                    self.card,
                    sp.SPCM_BUF_DATA,
                    sp.SPCM_DIR_PCTOCARD,
                    sp.int32(0),
                    pBuffer,
                    sp.uint64(0),
                    sp.uint64(buffer_size),
                )

                # Execute data transfer
                sp.spcm_dwSetParam_i32(
                    self.card,
                    sp.SPC_M2CMD,
                    sp.M2CMD_DATA_STARTDMA | sp.M2CMD_DATA_WAITDMA,
                )

                # Save data if desired
                if self.export_data:
                    segments[seg_idx] = {"buffer": np_waveform}

            if self.export_data:  # save segment to file for analysis
                export_settings = {
                    "n_channels": self.num_chs,
                    "sample_rate": self.clock_freq,
                }

                dataDict = {"segments": segments, "settings": export_settings}

                full_export_path = os.path.join(self.export_path, "segments.npy")
                print("Exporting segments to {}".format(full_export_path))
                np.save(os.path.join(full_export_path), dataDict)

            # Write sequence instructions
            for instr_idx, instr in enumerate(self.sequence_instrs):
                if instr_idx + 1 == len(self.sequence_instrs):
                    lCond = sp.SPCSEQ_END
                else:
                    lCond = sp.SPCSEQ_ENDLOOPALWAYS

                # end state, number of loops, next step index, current segment index
                lVal = sp.uint64(
                    (lCond << 32)
                    | (int(instr.loops) << 32)
                    | (int(instr.next_step) << 16)
                    | (int(instr.segment))
                )
                sp.spcm_dwSetParam_i64(
                    self.card, sp.SPC_SEQMODE_STEPMEM0 + int(instr.step), lVal
                )

        elif self.mode in self.mode_dict:
            raise LabscriptError("{} mode not implemented.")

        else:
            print(self.mode_dict)
            print(self.mode in self.mode_dict)
            raise LabscriptError("Mode not recognized.")
        print(f"Finished transfer to buffered:  t = {time.time() - start_time}")
        return True

    def set_trigger(self):
        if self.mode == b"Off":
            return

        sp.spcm_dwSetParam_i32(
            self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_START | sp.M2CMD_CARD_ENABLETRIGGER
        )

        print("Card started")

    def transfer_buffer(self):
        # Set to None so the garbage collector can get rid of the buffer when we're done
        self.buffer = None
        gc.collect()  # Tell the garbage collector to throw away the buffer data (we get a memory leak if we don't explicitly do this)
        print("Transfer complete")

    def h5group_to_dict(self, group):
        """
        ....
        """
        ans = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.h5group_to_dict(item)
        return ans

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        tstart = ptime.time()
        print(f"Start of transition to buffered: t = 0")

        self.waveform_groups = []
        self.base_name = h5file.split(".")[0]
        self.file_name = ntpath.basename(h5file)
        self.numpy_name = self.file_name.split(".")[0] + f"_spectrum_{device_name}.npy"
        self.groups_numpy_name = self.file_name.split(".")[0] + f"_spectrum_{device_name}_groups.npy"
        self.dict_numpy_name = self.file_name.split(".")[0] + f"_spectrum_{device_name}_dict.npy"


        # Look for precomputed pulse data        
        self.output_folder = "C:\\labscript-suite\\labscript-devices\\labscript_devices\\SpectrumPrecompute\\"
        self.precomputed_output_name = self.output_folder + self.numpy_name
        self.precomputed_groups_output_name = self.output_folder + self.groups_numpy_name
        self.precomputed_dict_name = self.output_folder + self.dict_numpy_name
        print(self.precomputed_output_name)
        self.precomputed = False

        t_before_opening_file = ptime.time() - tstart

        if (os.path.isfile(self.precomputed_output_name) and
            os.path.isfile(self.precomputed_groups_output_name)):
            self.precomputed = True
            print("Found precomputed pulse_data!")

        # Load the precomputed dictionary if available
        if (os.path.isfile(self.precomputed_dict_name)):
            device_dict = np.load(self.precomputed_dict_name, allow_pickle=True).item()
            print("found device dict!")
        else:
            print("No device dict")
            with h5py.File(h5file, "r") as file:
                device = file["/devices/" + device_name]
                device_dict = self.h5group_to_dict(device)
        try:
            np.testing.assert_equal(device_dict, self.previous_settings)
            generate_samples = False
        except AssertionError:
            print("Not equal, generating new samples")
            generate_samples = True


        if generate_samples:
            self.previous_settings = device_dict
        
        # Extract settings from the device dictionary
        settings = device_dict["device_settings"]
        self.mode = settings["mode"]
        self.clock_freq = int(settings["clock_freq"])
        self.use_ext_clock = bool(settings["use_ext_clock"])
        self.ext_clock_freq = int(settings["ext_clock_freq"])
        self.export_data = bool(settings["export_data"])
        self.export_path = str(settings["export_path"])

        ch_settings = device_dict["channel_settings"][:]
        self.channels = []
        enabled_ports = []
        for channel in ch_settings:
            self.channels.append(
                channel_settings(channel["name"], channel["power"], channel["port"])
            )
            enabled_ports.append(channel["port"])

        self.enabled_ports = enabled_ports
        self.num_chs = len(enabled_ports)

        if self.num_chs > self.max_channels:
            raise LabscriptError(
                "Spectrum card supports up to {} channels.".format(
                    self.max_channels
                )
            )


        t_before_waveform_extraction = ptime.time() - tstart

        if self.precomputed: 
            # self.waveform_groups = np.load(
            #     self.precomputed_groups_output_name, allow_pickle=True
            # )
            self.waveform_groups = pickle.load(
                open(self.precomputed_groups_output_name, "rb")
            )
            print("Using precomputed waveforms!")
        else:
            groups_folder = device_dict["waveform_groups"]

            for groupname in list(groups_folder.keys()):
                g = groups_folder[groupname]
                gsettings = g["group_settings"]
                time = gsettings["time"]
                duration = gsettings["duration"]
                loops = gsettings["loops"]
                waveforms = []

                for wavename in list(g.keys()):
                    if wavename != "group_settings":
                        wvf = waveform(0, 0, 0)
                        s = g[wavename]
                        for p in list(s.keys()):
                            if p == "waveform_settings":
                                wvf.time = s["waveform_settings"]["time"][0]
                                wvf.duration = int(
                                    s["waveform_settings"]["duration"][0]
                                )
                                wvf.loops = s["waveform_settings"]["loops"][0]
                                wvf.port = s["waveform_settings"]["port"][0]
                                wvf.sample_start = s["waveform_settings"][
                                    "sample_start"
                                ][0]
                                wvf.sample_end = s["waveform_settings"]["sample_end"][0]
                                wvf.delta_start = s["waveform_settings"]["delta_start"][
                                    0
                                ]
                                wvf.delta_end = s["waveform_settings"]["delta_end"][0]
                                wvf.modulation_frequencies = s["waveform_settings"][
                                    "modulation_frequencies"
                                ].ravel()
                                wvf.modulation_amplitudes = s["waveform_settings"][
                                    "modulation_amplitudes"
                                ].ravel()
                                wvf.modulation_phases = s["waveform_settings"][
                                    "modulation_phases"
                                ].ravel()
                                wvf.has_mask = s["waveform_settings"]["has_mask"][0]
                                wvf.remove_self_interaction = s["waveform_settings"][
                                    "remove_self_interaction"
                                ][0]
                                mask_str = s["waveform_settings"]["mask"][0]
                                mask_pickle = bytes(eval(mask_str))
                                mask = pickle.loads(mask_pickle)
                                wvf.mask = mask
                            if p == "pulse_data":
                                dset = s["pulse_data"]
                                for i in range(dset.shape[0]):
                                    start_freq = dset["start_freq"][i]
                                    end_freq = dset["end_freq"][i]
                                    ramp_time = dset["ramp_time"][i]
                                    phase = dset["phase"][i]
                                    amplitude = dset["amp"][i]
                                    ramp_type = dset["ramp_type"][i]
                                    painting_freq = dset["painting_freq"][i]
                                    painting_list = dset["painting_list"][i]
                                    # Convert painting function string back to function
                                    painting_function_str = dset["painting_function"][i]
                                    painting_function_pickle = bytes(
                                        eval(painting_function_str)
                                    )
                                    painting_function = pickle.loads(
                                        painting_function_pickle
                                    )
                                    wvf.add_pulse(
                                        start_freq,
                                        end_freq,
                                        ramp_time,
                                        phase,
                                        amplitude,
                                        ramp_type,
                                        painting_function=painting_function,
                                        painting_freq=painting_freq,
                                        painting_list=painting_list,
                                    )
                        waveforms.append(wvf)

                self.waveform_groups.append(
                    waveform_group(time, duration, waveforms, loops)
                )

        if len(self.waveform_groups) == 0:
            print(
                "Did not find any sample groups. Either something is wrong, or you haven't instructed the Spectrum card to do anything."
            )
            return self.final_values

        # Sort groups in time order (just in case)
        self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

        t_before_generate_samples = ptime.time() - tstart

        ### Card Settings ###
        ### Check cache ###
        if generate_samples:
            # returns true if no errors are raised
            self.card_settings()

            buf_gen_successful = self.generate_buffer()
            if buf_gen_successful:
                self.transfer_buffer()
                self.set_trigger()
        else:
            print("NO CHANGES")
            sp.spcm_dwSetParam_i32(self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_STOP)
            self.set_trigger()
            ### Send Buffer and Set Trigger ###

        tend = ptime.time() - tstart
        print(f"Before opening H5 file: t = {t_before_opening_file:.4f}")
        print(f"Before waveform extraction: t = {t_before_waveform_extraction:.4f}")
        print(f"Before generate samples: t = {t_before_generate_samples:.4f}")
        print(f"End of transition to buffered: t = {tend:.4f}")

        return self.final_values

    # Other Functions, manual mode is not used for the spectrum instrumentation card

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(abort=True)

    def abort_buffered(self):
        return self.transition_to_manual(abort=True)

    def transition_to_manual(self, abort=False):
        # if len(self.channels) > 2:
        #     self.shutdown()
        #     self.init()
        
        if abort:
            self.shutdown()
            self.init()
        return True

    def shutdown(self):
        sp.spcm_dwSetParam_i32(self.card, sp.SPC_M2CMD, sp.M2CMD_CARD_STOP)
        sp.spcm_vClose(self.card)
