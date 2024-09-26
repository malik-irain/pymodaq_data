import numpy as np
from pymodaq_data import data as data_mod

DWA = data_mod.DataRaw('data', data=[np.arange(10)])
INTEGER = 10


def test_add_int_reversed():
    dwa_add = INTEGER + DWA

    assert dwa_add == DWA + INTEGER


def test_sub_int_reversed():
    dwa_sub = INTEGER - DWA

    assert np.allclose(dwa_sub[0], INTEGER - DWA.data[0])



