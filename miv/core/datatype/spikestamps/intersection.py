__all__ = ["SpikestampsIntersection"]

from dataclasses import dataclass
from functools import reduce

import numpy as np

from miv.core.datatype.spikestamps.spikestamps import Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher


@dataclass
class SpikestampsIntersection(OperatorMixin):
    """
    Return intersection of multiple spikestamps.
    """

    precision: float = 0.001  # precision in sec

    tag: str = "spikestamps intersection"

    def __post_init__(self):
        super().__init__()
        self.cacher.policy = "OFF"

    @wrap_cacher("spikestamps intersection")
    def __call__(self, *args):
        number_of_channels = None
        for spikestamps in args:
            if number_of_channels is None:
                number_of_channels = spikestamps.number_of_channels
            else:
                assert (
                    number_of_channels == spikestamps.number_of_channels
                ), "All spikestamps must have the same number of channels."

        data = []
        for channel in range(number_of_channels):
            stamps = []
            for spikestamps in args:
                array = np.asarray(spikestamps[channel])
                stamps.append(np.around(array / self.precision).astype(np.int_))
            intersection = reduce(np.intersect1d, stamps)
            data.append(intersection * self.precision)

        return Spikestamps(data)
