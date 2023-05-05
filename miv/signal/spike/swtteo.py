__all__ = ["SWTTEODetection"]

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import csv
import functools
import inspect
import logging
import multiprocessing
import os
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pywt
import quantities as pq
import scipy.signal as sps
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher
from miv.statistics.spiketrain_statistics import firing_rates
from miv.typing import SignalType, SpikestampsType, TimestampsType
from miv.visualization.event import plot_spiketrain_raster


@dataclass
class SWTTEODetection(OperatorMixin):
    """SWTTEO spike detection
    This module contains the functions for the spike wavelet transform (SWT) and the spike wavelet transform.
    The algorithm is introduced in [1]_ and the application can be found in [2]_ [3]_.

    Code Example::

        bandpass = ButterBandpass(lowcut=300, highcut=3000)
        lowpass = ButterBandpass(highcut=300, btype='lowpass')
        detection = ThresholdCutoff(cutoff=5.0)
        swtteo = SWTTEODetection()
        spikestamps_intersection = SpikestampsIntersection()

        data >> bandpass >> detection >> spikestamps_intersection
        data >> lowpass >> swtteo >> spikestamps_intersection
        detection >> swtteo

        spiketrains = detection(signal, timestamps, sampling_rate)

    .. [1] Lieb F, Stark HG, Thielemann C. A stationary wavelet transform and a time-frequency based spike detection algorithm for extracellular recorded data. J Neural Eng. 2017 Jun;14(3):036013. doi: 10.1088/1741-2552/aa654b. Epub 2017 Mar 8. PMID: 28272020.
    .. [2] Mayer M, Arrizabalaga O, Lieb F, Ciba M, Ritter S, Thielemann C. Electrophysiological investigation of human embryonic stem cell derived neurospheres using a novel spike detection algorithm. Biosens Bioelectron. 2018 Feb 15;100:462-468. doi: 10.1016/j.bios.2017.09.034. Epub 2017 Sep 19. PMID: 28963963.
    .. [3] https://ars.els-cdn.com/content/image/1-s2.0-S0956566317306437-mmc1.pdf

        Attributes
        ----------
        tag : str
        progress_bar : bool
            Toggle progress bar (default=True)
    """

    spike_width: float = 0.001  # in seconds

    tag: str = "swtteo detection"
    progress_bar: bool = False

    num_proc: int = 1

    hamming_window_size: float = 0.013  # in seconds
    wavelet_decomposition_level: int = 2

    # @wrap_generator_to_generator
    @wrap_cacher(cache_tag="swtteo")
    def __call__(self, signal: SignalType, spikestamps: Spikestamps) -> SpikestampsType:
        """Execute threshold-cutoff method and return spike stamps

        Parameters
        ----------
        signal : Signal

        Returns
        -------
        spiketrain_list : List[SpikestampsType]

        """
        if not inspect.isgenerator(
            signal
        ):  # TODO: Refactor in multiprocessing-enabling decorator
            s = self.apply_swt_teo(signal)
            return self._detection(
                s, spikestamps.get_view(s.get_start_time(), s.get_end_time())
            )
        else:
            collapsed_result = Spikestamps()
            # with multiprocessing.Pool(self.num_proc) as pool:
            #    #for result in pool.map(functools.partial(ThresholdCutoff._detection, self=self), signal):
            #    inputs = list(signal)
            #    print(inputs)
            #    for result in pool.map(self._detection, inputs): # TODO: Something is not correct here. Check memory usage.
            #        collapsed_result.extend(spiketrain)
            for sig in signal:  # TODO: mp
                s = self.apply_swt_teo(sig)
                collapsed_result.extend(
                    self._detection(
                        s, spikestamps.get_view(s.get_start_time(), s.get_end_time())
                    )
                )
            return collapsed_result

    def apply_swt_teo(self, signal):
        # Apply SWT level 1 and 2
        NN = self.wavelet_decomposition_level
        window_size = int(
            self.hamming_window_size * signal.rate
        )  # convert 13ms to window size
        hamming_window = np.hamming(window_size)
        hamming_window = hamming_window / np.sqrt(
            3 * (hamming_window**2).sum() + (hamming_window.sum() ** 2)
        )  # Normalize

        coeffs = pywt.swt(signal.data, "sym5", level=NN, axis=0)

        streams = []
        for idx, (cA, _) in enumerate(coeffs):
            # plt.figure()
            # plt.plot(cA[:,0], label=f'cA {NN-idx}')
            output = self._signal_energy_operator(
                cA, absolute=True
            )  # Teager Energy Operator
            # plt.plot(output[:,0], label='energy')

            # Apply Hamming window
            for ch in range(output.shape[1]):
                output[:, ch] = np.convolve(output[:, ch], hamming_window, mode="same")
            # plt.plot(output[:,0], label='after hammin')
            # plt.legend()
            # plt.show(block=False)

            streams.append(output)

        data = functools.reduce(np.add, streams) / NN
        s = Signal(data=data, rate=signal.rate, timestamps=signal.timestamps)

        return s

    # @staticmethod
    def _detection(self, signal: SignalType, spikestamps: Spikestamps):
        # Spike detection for each channel
        spiketrain_list = []
        num_channels = signal.number_of_channels  # type: ignore
        timestamps = signal.timestamps
        rate = signal.rate

        spike_counts = spikestamps.get_count()
        for channel in tqdm(
            range(num_channels), disable=not self.progress_bar, desc=self.tag
        ):
            array = np.asarray(signal[channel])  # type: ignore
            count = spike_counts[channel]
            if count == 0:
                spiketrain_list.append([])
                continue

            # Spike Detection: get spikestamp
            spike_width = int(self.spike_width * rate)
            peaks, _ = sps.find_peaks(array, distance=spike_width)
            if len(peaks) == 0:
                spiketrain_list.append([])
                continue
            spikestamp = np.sort(timestamps[peaks])
            # energy_spike_indices = array[peaks].argsort()[-count:]
            # spikestamp = np.sort(timestamps[peaks][energy_spike_indices])

            spiketrain_list.append(spikestamp)
        spikestamps = Spikestamps(spiketrain_list)
        return spikestamps

    def __post_init__(self):
        super().__init__()

    # TODO: move to utils, if other functions are added
    def _nonlinear_energy_operator(self, signal: np.ndarray, ll=0, p=0, q=1, s=-1):
        """general form of the nonlinear energy operator

        return x(n-ll)x(n-p) - x(n-q)x(n-s)
        """
        assert (ll + p) == (q + s), "Incorrect parameters. Must be l+p=q+s"

        y = np.zeros_like(signal, dtype=np.float_)

        N = signal.shape[0]  # TODO: Axis could be changed
        n_edge = abs(ll) + abs(p) + abs(q) + abs(s)
        idx = np.arange(n_edge + 1, (N - n_edge - 1))

        # TODO: Axis could be changed
        y[idx, :] = (
            signal[idx - ll, :] * signal[idx - p, :]
            - signal[idx - q, :] * signal[idx - s, :]
        )

        return y

    def _signal_energy_operator(self, signal, absolute=False, variation="teager"):
        """generate different NLEOs based on the same operator

        Parameters
        ----------
        signal: np.ndarray
            input signal
        absolute: bool
            if True, return absolute value of operator
        variation: {'teager', 'agarwal', 'palmu'}
            which type of NLEO?

        Returns
        -------
        output : ndarray
        """

        coeffs = {
            "teager": [0, 0, 1, -1],
            "agarwal": [1, 2, 0, 3],
            "palmu": [1, 2, 0, 3],
        }
        output = self._nonlinear_energy_operator(signal, *coeffs[variation])
        if absolute:
            output.data = np.abs(output.data)
        return output

    def plot_spiketrain(
        self,
        spikestamps,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        """
        Plot spike train in raster
        """
        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # TODO: REFACTOR. Make single plot, and change xlim
        term = 60
        n_terms = int(np.ceil((tf - t0) / term))
        if n_terms == 0:
            # TODO: Warning message
            return None
        for idx in range(n_terms):
            fig, ax = plot_spiketrain_raster(
                spikestamps, idx * term + t0, min((idx + 1) * term + t0, tf)
            )
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"spiketrain_raster_{idx:03d}.png"))
            if not show:
                plt.close("all")
        if show:
            plt.show()
            plt.close("all")
        return ax

    def plot_firing_rate_histogram(self, spikestamps, show=False, save_path=None):
        """Plot firing rate histogram"""
        threshold = 3

        rates = firing_rates(spikestamps)["rates"]
        hist, bins = np.histogram(rates, bins=20)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-3)), np.log10(bins[-1]), len(bins)
        )
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(rates, bins=logbins)
        ax.axvline(
            np.mean(rates),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean {np.mean(rates):.2f} Hz",
        )
        ax.axvline(
            threshold,
            color="g",
            linestyle="dashed",
            linewidth=1,
            label="Quality Threshold",
        )
        ax.set_xscale("log")
        xlim = ax.get_xlim()
        ax.set_xlabel("Firing rate (Hz) (log-scale)")
        ax.set_ylabel("Count")
        ax.set_xlim([min(xlim[0], 1e-1), max(1e2, xlim[1])])
        ax.legend()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "firing_rate_histogram.png"))
            with open(
                os.path.join(f"{save_path}", "firing_rate_histogram.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "firing_rate_hz"])
                data = list(enumerate(rates))
                data.sort(reverse=True, key=lambda x: x[1])
                for ch, rate in data:
                    writer.writerow([ch, rate])
        if show:
            plt.show()

        return ax
