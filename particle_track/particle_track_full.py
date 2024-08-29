#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:35:06 2023

@author: vcant
"""

from functools import partial
from multiprocessing import Pool

import flopy
import numba
import numpy as np
from numba import boolean, float64, int64

from .preprocessing import prepare_arrays


@numba.njit(nogil=True)
def velocity(coords, v0, gv, face_coords):
    """
    Calculates cell velocities at the starting location in the cell.

    Parameters
    ----------
    coords : ndarray
        [x,y,z] of the particle
    v0 : velocity at the lower face coordinate
        [vx0, vy0, vz0]: velocities of left, lower, and bottom faces
    gv : ndarray
        [dvdx, dvdy,dvdz]: linear velocity gradients for each cell
    face_coords : ndarray
        [x0,y0,z0]: coordinates for the left, lower and bottom faces

    Returns
    -------
    v : [vx,vy,vz]
    Current velocities for the particle at x,y,z

    """

    v = gv * (coords - face_coords) + v0

    return v


@numba.vectorize([int64(float64, float64, float64)], nopython=True)
def exit_direction(v1, v2, v):
    """
    Define exit direction (lower or upper) based on current velocity of particle and the velocity at faces (run for each axis: x,y,z)
    Parameters
    ----------
    v1: velocity at lower face
    v2: velocity at upper face
    v: current velocity

    Returns
    index (-1: exit at lower face (negative axis direction),
            0: impossible exit at this axis
            1: exit at upper face (positive axis direction)
    -------

    """
    if (v1 >= 0.0) & (v2 > 0.0):
        r = 1
    elif (v1 < -0.0) & (v2 <= -0.0):
        r = -1
    elif (v1 >= 0.0) & (v2 <= -0.0):
        r = 0
    else:  # (v1 < 0) & (v2 > 0):
        if v > 0.0:
            r = 1
        elif v < -0.0:
            r = -1
        else:
            r = 0
    return r


@numba.vectorize(
    [
        float64(
            int64,
            boolean,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def reach_time(exit_ind, gradient_logic, v1, v2, v, gv, x, left_x, right_x):
    """
    Calculates the time to reach the exit faces at each axis.

    Parameters
    ----------
    exit_ind: index calulated from exit_direction
    gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
    v1: velocity at the lower face
    v2: velocity at the upper face
    v: current particle velocity
    gv: velocity gradient in the cell
    x: current particle coordinates
    left_x: coordinates of lower left corner
    right_x: coordinates of upper right corner

    Returns
    -------
    dt: travel time at the current cell

    """
    if exit_ind == 0:
        tx = np.inf
    elif exit_ind == -1:
        if ~gradient_logic:
            tx = (-1) * (x - left_x) / v
        else:
            tx = (np.log(v1/v)) / gv
    else:  # exit_ind == 1
        if ~gradient_logic:
            tx = (right_x - x) / v
        else:
            tx = (np.log(v2/v)) / gv
    return tx


@numba.vectorize(
    [
        float64(
            int64,
            boolean,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def exit_location(exit_ind, gradient_logic, dt, v1, v2, v, gv, x, left_x, right_x):
    """
    Calculate the coordinates at the exit location in the cell
    Parameters
    ----------
    exit_ind: index calulated from exit_direction
    gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
    dt: calculated travel time at the cell
    v1: velocity at the lower face
    v2: velocity at the upper face
    v: current particle velocity
    gv: velocity gradient in the cell
    x: current particle coordinates
    left_x: coordinates of lower left corner
    right_x: coordinates of upper right corner

    Returns
    -------
    coords: exit coordinate at each axis

    """
    if ~gradient_logic:
        x_new = x + v * dt
    elif np.abs(v) > 1e-20:
        if exit_ind == 1:
            x_new = left_x + (1 / gv) * (v * np.exp(gv * dt) - v1)

        elif exit_ind == -1:
            x_new = right_x + (1 / gv) * (v * np.exp(gv * dt) - v2)
        else:
            x_new = left_x + (1 / gv) * (v * np.exp(gv * dt) - v1)
    else:
        x_new = x

    return x_new


@numba.njit(float64[:](float64[:,:]), nogil=True)
def numba_max_abs(d_array):
    res = []
    for i in range(d_array.shape[0]):
        array = np.max(np.abs(d_array[i, :]))
        res.append(array)

    return np.array(res)


@numba.njit(boolean(int64[:]), nogil=True)
def negative_index(ind_array):
    for i in range(ind_array.shape[0]):
        if ind_array[i] < 0:
            return True
    return False


@numba.njit(boolean(int64[:],int64[:]),nogil=True)
def larger_index(test_array, reference_array):
    """
    Check if the cell index represented by the test_array
     is larger than the reference_array, which is a shape of the grid.
    """
    for i in range(test_array.shape[0]):
        if test_array[i] > reference_array[i]-1:
            return True
    return False


@numba.njit
def trajectory(
    initial_position,
    initial_cell,
    face_velocities,
    gvs,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
):
    """
    Calculates the trajectory of a particle until it leaves the domain or find a termination point.
    Parameters
    ----------
    initial_position
    initial_cell
    face_velocities
    gvs
    xedges
    yedges
    z_lf
    z_uf
    termination

    return
    inds: layer,row,col at each cell it passes
    ts: travel time at each cell
    xes: x positions
    yes: y position
    zes: z positions
    -------

    """
    # initializing:
    x = initial_position[0]
    y = initial_position[1]
    z = initial_position[2]
    layer = initial_cell[0]
    row = initial_cell[1]
    col = initial_cell[2]

    continue_tracking = True
    ts = []
    layers = []
    rows = []
    cols = []
    xes = []
    yes = []
    zes = []
    coords = np.array([x, y, z])

    eps = np.finfo(np.float64).eps

    # While loop until termination:
    while continue_tracking:
        # coordinates at lower and upper faces:
        left_x = xedges[col]
        right_x = xedges[col + 1]
        low_y = yedges[row + 1]
        top_y = yedges[row]
        bt_z = z_lf[layer, row, col]
        up_z = z_uf[layer, row, col]

        coords_0 = np.array([left_x, low_y, bt_z])
        coords_1 = np.array([right_x, top_y, up_z])
        # gradients for the cell
        gvp = gvs[:, layer, row, col]

        # velocities at lower coordinate faces:
        v0 = face_velocities[np.array([0, 2, 4]), layer, row, col]
        # velocities at upper coordinate faces:
        v1 = face_velocities[np.array([1, 3, 5]), layer, row, col]

        # current velocities
        v = velocity(coords, v0, gvp, coords_0)

        # Where is it going:
        velocity_gradient_index = np.abs(v0 - v1) > eps * numba_max_abs(
            np.column_stack((v0, v1))
        )
        exit_direction_index = exit_direction(v0, v1, v)

        # Time to reach each end
        dt_array = reach_time(
            exit_direction_index,
            velocity_gradient_index,
            v0,
            v1,
            v,
            gvp,
            coords,
            coords_0,
            coords_1,
        )

        # actual travel time:
        dt = np.min(dt_array)
        exit_point_loc = np.argmin(dt_array)

        if dt == np.inf:
            break

        exit_point = exit_location(
            exit_direction_index,
            velocity_gradient_index,
            dt,
            v0,
            v1,
            v,
            gvp,
            coords,
            coords_0,
            coords_1,
        )

        if exit_point_loc == 0:
            col = col + exit_direction_index[0]
        if exit_point_loc == 1:
            row = row + (-exit_direction_index[1])
        if exit_point_loc == 2:
            layer = layer + (-exit_direction_index[2])

        ##termination criteria evaluation: whether the particle has reached a termination layer or out of the system
        has_negative_index = negative_index(np.array([layer, row, col]))
        over_index = larger_index(
            np.array([layer, row, col]), np.array(termination.shape)
        )
        term_value = termination[layer, row, col]

        if (term_value == 1) | has_negative_index | over_index:
            continue_tracking = False

            for i, index in enumerate((layer, row, col)):
                if index < 0:
                    index = 0
                if index >= termination.shape[i]:
                    index = termination.shape[i] - 1
                if i == 0:
                    layer = index
                if i == 1:
                    row = index
                if i == 2:
                    col = index

        # Adding to the dataset:
        layers.append(layer)
        rows.append(row)
        cols.append(col)
        ts.append(dt)
        xes.append(exit_point[0])
        yes.append(exit_point[1])
        zes.append(exit_point[2])

        # new loop:
        coords = exit_point

    # ts.append(0.0)
    ts = np.array(ts)
    xes = np.array(xes)
    yes = np.array(yes)
    zes = np.array(zes)
    layers = np.array(layers)
    rows = np.array(rows)
    cols = np.array(cols)
    inds = np.column_stack((layers, rows, cols))
    return (inds, ts, xes, yes, zes)


@numba.jit(nopython=True, parallel=True)
def work_v2(
    particle_centers, face_velocities, gvs, xedges, yedges, z_lf, z_uf, termination
):
    """
    Runs the particle trajectory in a multiprocessing enviroment using plain numba
    """
    # Preallocating lists of the results for thread safety
    inds_list = [
        np.empty((0, 3), dtype=np.int64) for _ in range(particle_centers.shape[0])
    ]
    ts_list = [
        np.empty((0), dtype=np.float64) for _ in range(particle_centers.shape[0])
    ]
    pos_list = [
        np.empty((0, 3), dtype=np.float64) for _ in range(particle_centers.shape[0])
    ]
    # parallel loop of particles
    for i in numba.prange(particle_centers.shape[0]):
        particle_starting_coords = particle_centers[i, :3]
        particle_starting_cell = particle_centers[i, 3:].astype(np.int64)
        inds, ts, xes, yes, zes = trajectory(
            particle_starting_coords,
            particle_starting_cell,
            face_velocities,
            gvs,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
        )
        inds_list[i] = inds
        ts_list[i] = ts
        pos_list[i] = np.column_stack((xes, yes, zes))
    return inds_list, ts_list, pos_list


def work(
    iterator_realization, face_velocities, gvs, xedges, yedges, z_lf, z_uf, termination
):
    """
    Runs the particle trajectory in a multiprocessing enviroment.
    The particle array information is stored in the iterator_realization variable
    Returns
    -------
    particle_traj : TYPE
        returns the particle trajectory

    """

    j = iterator_realization[1]
    particle_array = iterator_realization[0]

    # Getting the coordinates and cell location from the particle array
    particle_starting_coords = particle_array[:3]
    particle_starting_cell = particle_array[3:].astype(np.int64)
    # Initializing array
    inds, ts, xes, yes, zes = trajectory(
        particle_starting_coords,
        particle_starting_cell,
        face_velocities,
        gvs,
        xedges,
        yedges,
        z_lf,
        z_uf,
        termination,
    )
    jes = np.repeat(j, xes.shape[0])
    particle_traj = np.column_stack((jes, xes, yes, zes, ts))
    return particle_traj


def pollock(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    mode: str = "forward",
    processes: int = 4,
):
    """
    Runs particle tracking in a modflow (flopy) model, using the algorithm of Pollock (1988).
    Parameters
    ----------
    gwfmodel : flopy.mf6.MFModel
        The modflow model
    model_directory : str
        The directory where the model is stored
    particles_starting_location : np.ndarray
        The particle array information
    porosity : float | np.ndarray
        The porosity of the model (can be a constant or an array matching the grid)
    mode : str, optional
        The direction of the particle tracking (forward or backwards). The default is "forward".
    processes : int, optional
        The number of processes to run the particle tracking. The default is 4.
    The particle array information is stored in the iterator_realization variable
    Returns
    -------
    particle_locations : np.ndarray[n,5]
        returns an array with the particle indexes, x,y,z location and travel time (i, x, y, z, t)
    """
    iter_coords = iter(particles_starting_location)
    js = np.arange(particles_starting_location.shape[0])

    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
        gwfmodel, model_directory, porosity
    )

    if mode == "backwards":
        face_velocities = (-1) * face_velocities
        gvs = (-1) * gvs

    good_work = partial(
        work,
        face_velocities=face_velocities,
        gvs=gvs,
        xedges=xedges,
        yedges=yedges,
        z_lf=z_lf,
        z_uf=z_uf,
        termination=termination,
    )

    with Pool(processes=processes) as pool:
        results = pool.map(good_work, zip(iter_coords, js))

    true_results = np.vstack(results)

    return true_results


def pollock_v2(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    mode: str = "forward",
    processes: int = 4,
):
    """
    Runs particle tracking in a modflow (flopy) model, using the algorithm of Pollock (1988).
    Parameters
    ----------
    gwfmodel : flopy.mf6.MFModel
        The modflow model
    model_directory : str
        The directory where the model is stored
    particles_starting_location : np.ndarray
        The particle array information
    porosity : float | np.ndarray
        The porosity of the model (can be a constant or an array matching the grid)
    mode : str, optional
        The direction of the particle tracking (forward or backwards). The default is "forward".
    processes : int, optional
        The number of processes to run the particle tracking. The default is 4.
    The particle array information is stored in the iterator_realization variable
    Returns
    -------
    particle_locations : np.ndarray[n,5]
        returns an array with the particle indexes, x,y,z location and travel time (i, x, y, z, t)
    """
    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
        gwfmodel, model_directory, porosity
    )

    if mode == "backwards":
        face_velocities = (-1) * face_velocities
        gvs = (-1) * gvs

    inds_list, ts_list, pos_list = work_v2(
        particles_starting_location,
        face_velocities,
        gvs,
        xedges,
        yedges,
        z_lf,
        z_uf,
        termination,
    )

    true_results = []

    for i in range(len(inds_list)):
        true_results.append(
            np.column_stack((np.repeat(i, len(inds_list[i])), pos_list[i], ts_list[i]))
        )

    true_results = np.vstack(true_results)

    return true_results
