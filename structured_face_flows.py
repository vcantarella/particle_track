#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:17:31 2023

@author: vcant
"""

import numpy as np

from flopy.mf6.utils.binarygrid_util import MfGrdFile


def get_structured_faceflows(
    flowja, grb_file=None, ia=None, ja=None, verbose=False
):
    """
    Get the face flows for the flow right face, flow front face, and
    flow lower face from the MODFLOW 6 flowja flows. This method can
    be useful for building face flow arrays for MT3DMS, MT3D-USGS, and
    RT3D. This method only works for a structured MODFLOW 6 model.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    ia : list or ndarray
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray
        CRS column pointers. Only required if grb_file is not provided.
    verbose: bool
        Write information to standard output

    Returns
    -------
    frf : ndarray
        right face flows
    fff : ndarray
        front face flows
    flf : ndarray
        lower face flows

    """
    if grb_file is not None:
        grb = MfGrdFile(grb_file, verbose=verbose)
        if grb.grid_type != "DIS":
            raise ValueError(
                "get_structured_faceflows method "
                "is only for structured DIS grids"
            )
        ia, ja = grb.ia, grb.ja
    else:
        if ia is None or ja is None:
            raise ValueError(
                "ia and ja arrays must be specified if the MODFLOW 6"
                "binary grid file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = np.ravel(flowja)

    # evaluate size of flowja relative to ja
    __check_flowja_size(flowja, ja)

    # create face flow arrays
    shape = (grb.nlay, grb.nrow, grb.ncol)
    frf = np.zeros(shape, dtype=float).ravel()
    fff = np.zeros(shape, dtype=float).ravel()
    flf = np.zeros(shape, dtype=float).ravel()

    # fill flow terms
    vmult = [-1.0, -1.0, -1.0]
    flows = [frf, fff, flf]
    for n in range(grb.nodes):
        i0, i1 = ia[n] + 1, ia[n + 1]
        for j in range(i0, i1):
            jcol = ja[j]
            if jcol > n:
                if jcol == n + 1:
                    ipos = 0
                elif jcol == n + grb.ncol:
                    if grb.nrow > 1:
                        ipos = 1
                    else:
                        ipos = 2
                else:
                    ipos = 2
                flows[ipos][n] = vmult[ipos] * flowja[j]
    return frf.reshape(shape), fff.reshape(shape), flf.reshape(shape)


# internal
def __check_flowja_size(flowja, ja):
    """
    Check the shape of flowja relative to ja.
    """
    if flowja.shape != ja.shape:
        raise ValueError(
            f"size of flowja ({flowja.shape}) not equal to {ja.shape}"
        )