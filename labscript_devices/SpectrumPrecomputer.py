""" Like the spectrum precomputer, but we also save the waveform groups.

The Spectrum Precomputer, but made so you can run multiple at once.
"""

from labscript import IntermediateDevice, Device, LabscriptError, DigitalOut
from labscript_devices import BLACS_tab

import labscript_utils.h5_lock
import os, sys
import traceback
from blacs.remote import Client as BlacsClient
from labscript import LabscriptError
import math
import numpy as np
import dill as pickle

sys.path.append("..")
sys.path.append("C:/Users/QuantumEngineer")
# from Spectrum import sequence_instr, pulse, waveform, waveform_group, channel_settings
from .spcm import pyspcm as sp
from .spcm import spcm_errors as se
from .spcm import spcm_tools as st
from .spcm.spcm_modulation_compensation import *
from .spcm.numba_chirp import *

from scipy.signal import chirp
import ctypes

import os
import time
import pathlib

from labscript_utils import h5_lock
import h5py
import random


blacs_client = BlacsClient(host="171.64.56.36", port=25227)


PRECOMPUTE_FOLDER = "C:\\labscript-suite\\labscript-devices\\labscript_devices\\SpectrumPrecompute\\"

### get files in queue


import pathlib


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

        if self.painting_function == None:
            self.is_painted = False
        else:
            self.is_painted = True
        self.painting_freq = painting_freq
        self.painting_list = painting_list

    def __str__(self):
        # Random number to never use smart programming
        if self.is_painted:
            s = f"Painted sweep using function {self.painting_function} painting frequency: {self.painting_freq}, painting_list: {self.painting_list}, {np.random.random()} random number"
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
            mask =lambda x: np.ones_like(x)
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


import ntpath
import copy

class SpectrumPrecomputer:
    def __init__(self, shot: str, pulse_dictionary):
        self.shot = shot
        self.base_name = self.shot.split(".")[0]
        self.file_name = ntpath.basename(self.shot)
        self.numpy_name = self.file_name.split(".")[0]
        self.output_folder = PRECOMPUTE_FOLDER
        self.previous_settings = None
        self.max_channels = 4
        self.samplesPerChunk = 32
        self.bytesPerSample = 2
        self.pulse_dictionary = pulse_dictionary


    def compute_waveform(self):
        start_time = time.time()
        segments = {}
        self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

        pickle.dump(self.waveform_groups, open(self.output_groups_name, "wb"))
        print(f"Groups output saved to {self.output_groups_name}")

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
                        waveform_group(t0 + dummy_loop_dur * n_loops, leftover, "dummy")
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

        # Prepare data structure for saving segments, if desired
        if self.export_data:
            segments = {}
            #            print(f"Beginning to write segments: t = {time.time() - start_time}")

        # Write segments
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
                if verbose:
                    print(len(group.waveforms))
                for wvf in group.waveforms:
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

                    if hash(wvf) in self.pulse_dictionary.keys() and not wvf.has_mask:
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
                                        np.sin(phase_values) / phase_values.shape[0]
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
                                            2 * pi / 360 * pulse.painting_function(t)
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
                                        and np.round(int(f1 / 1e5) - f1 / 1e5, 5) == 0
                                        and dur > 100e-6
                                    ):
                                        loop_duration = 1 / np.gcd(
                                            int(self.clock_freq), int(f1)
                                        )
                                        small_t = np.arange(
                                            0,
                                            loop_duration,
                                            1 / self.clock_freq,
                                        )
                                        num_loops = int(dur / loop_duration)
                                        # small_chirp = chirp(
                                        #     small_t,
                                        #     pulse.start,
                                        #     loop_duration,
                                        #     f1,
                                        #     phi=pulse.phase,
                                        # )
                                        small_chirp = numba_chirp(
                                            small_t,
                                            f0=pulse.start,
                                            t1=loop_duration,
                                            f1=f1,
                                            phi=pulse.phase,
                                        )
                                        # assert (
                                        #     np.max(np.abs(small_chirp - small_chirp_numba))
                                        #     < 1e-10
                                        # ), "Numba small chirp not equivalent to scipy small chirp"
                                        c2 = np.tile(small_chirp, num_loops)
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

                                pulse_data += pulse.amp * (2**15 - 1) * c

                        if wvf.remove_self_interaction:
                            modulation_waveform = np.abs(
                                np.sum(
                                    [
                                        amp
                                        * np.cos(
                                            2 * np.pi * (freq * t) + phase * np.pi / 180
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
                                    amp
                                    * (
                                        1
                                        / 2
                                        * np.cos(
                                            2 * np.pi * (freq * t) + phase * np.pi / 180
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
                        modulation_waveform = total_modulation * modulation_waveform + (
                            1 - total_modulation
                        )
                        #
                        #
                        # modulation_waveform = modulation_waveform + 1/2
                        modulation_waveform = amplitude_compensator(modulation_waveform)
                        assert (
                            np.min(modulation_waveform) >= 0
                            and np.max(modulation_waveform) <= 1
                        ), "Modulation is too big"
                        assert (
                            pulse_data.shape == modulation_waveform.shape
                        ), "Modulation waveform has wrong shape"
                        if verbose:
                            print("pd: ", pulse_data)
                            print("mw: ", modulation_waveform)
                        pulse_data = pulse_data * modulation_waveform

                        mask_values = wvf.mask(
                            t
                        )  # should we only do this if has mask for extra speed?

                        assert (
                            np.min(mask_values) >= -1
                        ), f"Mask gets too small, {mask_values}"
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
                    # TODO: Also track number of times pulses have been used
                    if len(self.pulse_dictionary.keys()) > 100:
                        self.pulse_dictionary.clear()

                    port_index = sorted(self.enabled_ports).index(wvf.port)

                    begin = (
                        (wvf.time - group.time) * samples_per_chunk + wvf.delta_start
                    ) * int(self.num_chs) + port_index
                    end = begin + (len(pulse_data) * int(self.num_chs))
                    increment = int(self.num_chs)
                    np_waveform[begin:end:increment] = pulse_data[
                        : len(np_waveform[begin:end:increment])
                    ]

                segments[seg_idx] = np_waveform
        np.save(self.output_name, segments, allow_pickle=True)
        print(f"Output saved to {self.output_name}")

        # # Naively, you would save the groups output here. However,
        # # That's a bad idea because we've already mixed in a ton of
        # # dummy data. So don't do the following:
        # pickle.dump(self.waveform_groups, open(self.output_groups_name, "wb"))
        # print(f"Groups output saved to {self.output_groups_name}")

        return self.output_name, self.output_groups_name

    def get_pulse_data(self, wvf):
        if hash(wvf) in self.pulse_dictionary.keys():
            if verbose:
                print("Found waveform")
            pulse_data = self.pulse_dictionary[hash(wvf)]
            return pulse_data
        if wvf.remove_self_interaction:
            pulse_data = self.get_pulse_data_no_self_interaction(wvf)

    def get_pulse_data_no_self_interaction(self, wvf):
        for pulse in wvf.pulses:
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
        return pulse_data
    
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

    def transition_to_buffered(self, device_name):
        self.output_name = self.base_name + f"_spectrum_{device_name}.npy"
        self.numpy_groups_name = self.numpy_name + f"_spectrum_{device_name}_groups.npy"
        self.device_dict_name = (
            self.output_folder + self.numpy_name + f"_spectrum_{device_name}_dict.npy"
        )
        self.numpy_name = self.numpy_name + f"_spectrum_{device_name}.npy"
        self.output_name = self.output_folder + self.numpy_name
        self.output_groups_name = self.output_folder + self.numpy_groups_name

        if os.path.isfile(self.output_name):
            raise FileExistsError(f"A file exists at {self.output_name}")

        if not os.path.isfile(self.shot):
            raise FileExistsError(f"Shot probably already ran {self.shot}")



        self.waveform_groups = []
        with h5py.File(self.shot, "r") as file:
            device = file["/devices/" + device_name]
            try:
                device_dict = self.h5group_to_dict(device)
                print(f"Saving dict! {self.device_dict_name}")
                np.save(self.device_dict_name, device_dict, allow_pickle=True)

                np.testing.assert_equal(device_dict, self.previous_settings)
                self.generate_samples = False
            except AssertionError:
                print("Not equal, generating new samples")
                self.generate_samples = True
            except OSError:
                self.generate_samples = True

            settings = device["device_settings"]
            self.mode = settings["mode"]
            self.clock_freq = int(settings["clock_freq"])
            self.use_ext_clock = bool(settings["use_ext_clock"])
            self.ext_clock_freq = int(settings["ext_clock_freq"])
            self.export_data = bool(settings["export_data"])
            self.export_path = str(settings["export_path"])

            ch_settings = device["channel_settings"][:]
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

            groups_folder = device["waveform_groups"]

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
                                    painting_function_pickle = eval(
                                        painting_function_str
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

        ### Card Settings ###
        ### Check cache ###

        # segment_list = self.generate_buffer()

        return


### for each file in queue
#### compute spectrum
#### save spectrum
all_files = {}
all_groups_files = {}
all_dict_files = {}
import time
import multiprocessing


def generate_npy(shots):
    try:
        shot = shots.get()
        spc = SpectrumPrecomputer(shot)
        spc.transition_to_buffered("SpectrumM4X")
        all_dict_files[shot] = spc.device_dict_name
        all_files[shot], all_groups_files[shot] = spc.compute_waveform()
    except FileNotFoundError:
        print(f"{shot} does not exist anymore")
    except np.core._exceptions._ArrayMemoryError:
        print("Memory error!")
    except Exception as e:
        print("Some other error")


def main_loop():
    all_files = {}
    pulse_dictionary = {}

    while True:
        all_shots = blacs_client.queued_shots()

        # Every check_every_n shots, we check for more shots from BLACS.
        check_every_n = 2
        counter = 0
        completed_shots = []

        start_time = time.time()

        # While there are shots in all_shots
        while all_shots:
            # Choose randomly from the final ten shots
            nextn = random.randrange(
                min(2 * check_every_n, len(all_shots) - 1), len(all_shots)
            )
            shot = all_shots.pop(nextn)
            time.sleep(.3)
            try:
                print(f"\nComputing shot: {shot}\n\n")

                # don't compute if we've already generated a file for it
                if shot not in all_files:
                    spc = SpectrumPrecomputer(shot, pulse_dictionary)
                    # Should throw a FileExistsError if file was already precomputed
                    spc.transition_to_buffered("SpectrumM4X")
                    if spc.generate_samples:
                        all_files[shot], all_groups_files[shot] = spc.compute_waveform()
                    pulse_dictionary = spc.pulse_dictionary
                    all_dict_files[shot] = spc.device_dict_name

                if shot not in completed_shots:
                    completed_shots.append(shot)
            except FileExistsError as e:
                print(e)
                if shot not in completed_shots:
                    completed_shots.append(shot)
            except FileNotFoundError:
                print(f"{shot} does not exist anymore")
            except AttributeError:
                print("Testing origin...")
                traceback.print_exc()
            except OSError as e:
                traceback.print_exc()
                return

            counter += 1
            if counter >= check_every_n:
                print("Checking for new shots")
                all_new_shots = blacs_client.queued_shots()

                # Note: we only use new_shots for this print statements
                new_shots = list(
                    set(all_new_shots) - set(all_shots) - set(completed_shots)
                )
                print("\nNew shots are:", new_shots, "\n")

                # This rids us of shots already run as well as completed shots
                all_shots = list(set(all_new_shots) - set(completed_shots))
                counter = 0

            # spc.transition_to_buffered("TweezerSpectrum")
            # if spc.generate_samples:
            #     all_files[shot] = spc.compute_waveform()
        all_shots = blacs_client.queued_shots()
        time.sleep(2)
        for shot in list(all_files.keys()):
            # If the shot is no longer in BLACS, then it
            # has been run by BLACS or otherwise removed,
            # and we don't need to keep the precomputed
            # data any longer.
            if shot not in all_shots:
                print("Deleting", shot)
                try:
                    # Delete the precomputed file file
                    os.remove(all_files[shot])
                except:
                    print("Error deleting file!")
                    traceback.print_exc()
                try:
                    # Delete the precomputed groups file
                    os.remove(all_groups_files[shot])
                except:
                    print("Error deleting groups file!")
                    traceback.print_exc()
                try:
                    # Delete the precomputed groups file
                    os.remove(all_dict_files[shot])
                except:
                    print("Error deleting dict file!")
                    traceback.print_exc()
                try:
                    del all_files[shot]
                except:
                    print("Error in deletion from dictionary")
                    traceback.print_exc()
                try:
                    del all_groups_files[shot]
                except:
                    print("Error in deletion from groups dictionary")
                    traceback.print_exc()
                try:
                    del all_dict_files[shot]
                except:
                    print("Error in deletion from groups dictionary")
                    traceback.print_exc()

        try:  
            filenames = next(os.walk(PRECOMPUTE_FOLDER))[2]
            halfday = 12 * 60 * 60
            
            print("Checking for old files...")
            for filename in filenames:
                if time.time() > os.stat(PRECOMPUTE_FOLDER+filename).st_mtime + halfday:
                    os.remove(PRECOMPUTE_FOLDER+filename)
                    print(f"Deleted {filename}")
        except:
            print("Error in deleting old files")
            traceback.print_exc()



if __name__ == "__main__":
    main_loop()
### if saved_spectrum is not in queue, delete it
