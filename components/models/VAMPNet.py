import torch
import torch.nn as nn

import numpy as np
from deeptime.base import Model, Transformer
from typing import Optional
from deeptime.util.torch import map_data

class VAMPNetModel(Transformer, Model):
    r"""
    A VAMPNet model which can be fit to data optimizing for one of the implemented VAMP scores.

    Parameters
    ----------
    lobe : torch.nn.Module
        One of the lobes of the VAMPNet. See also :class:`deeptime.util.torch.MLP`.
    lobe_timelagged : torch.nn.Module, optional, default=None
        The timelagged lobe. Can be left None, in which case the lobes are shared.
    dtype : data type, default=np.float32
        The data type for which operations should be performed. Leads to an appropriate cast within fit and
        transform methods.
    device : device, default=None
        The device for the lobe(s). Can be None which defaults to CPU.

    See Also
    --------
    VAMPNet : The corresponding estimator.
    """

    def __init__(self, lobe: "torch.nn.Module", lobe_timelagged: Optional["torch.nn.Module"] = None,
                 dtype=np.float32, device=None):
        super().__init__()
        self._lobe = lobe
        self._lobe_timelagged = lobe_timelagged if lobe_timelagged is not None else lobe

        if dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()
        self._dtype = dtype
        self._device = device

    @property
    def lobe(self) -> "torch.nn.Module":
        r""" The instantaneous lobe.

        Returns
        -------
        lobe : nn.Module
        """
        return self._lobe

    @property
    def lobe_timelagged(self) -> "torch.nn.Module":
        r""" The timelagged lobe. Might be equal to :attr:`lobe`.

        Returns
        -------
        lobe_timelagged : nn.Module
        """
        return self._lobe_timelagged

    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        if instantaneous:
            self._lobe.eval()
            net = self._lobe
        else:
            self._lobe_timelagged.eval()
            net = self._lobe_timelagged
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(net(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]
