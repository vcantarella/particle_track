
import flopy
import numba
import numpy as np
from numba import float64, int64

from .particle_track_full import (
    exit_direction,
    exit_location,
    larger_index,
    negative_index,
    numba_max_abs,
    reach_time,
    velocity,
)
from .preprocessing import prepare_arrays

@numba.jit(nopython=True, nogil=True)
def travel_time_cum_reactivity(
    initial_position,
    initial_cell,
    face_velocities,
    gvs,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
    reactivity,
):
    """
    Calculates the travel_time and cumulative reactivity of a particle assuming steady-state flow conditions
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
    reactivity: array with relative reactivity values per cell

    return
    t: (sum of dt)
    r: (sum of dr*dt)
    -------

    """
    # initializing:
    layer, row, col = initial_cell

    continue_tracking = True
    coords = initial_position

    # Initializing the variables:
    dts = 0.0
    reacts = 0.0

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

        # relative reactivity of the cell
        relative_react = reactivity[layer, row, col]

        # velocities at lower coordinate faces:
        v0 = face_velocities[np.array([0, 2, 4]), layer, row, col]
        # velocities at upper coordinate faces:
        v1 = face_velocities[np.array([1, 3, 5]), layer, row, col]

        # current velocities
        v = velocity(coords, v0, gvp, coords_0)

        # Where is it going:
        velocity_gradient_index = np.abs(v0 - v1) > 1e-10 * numba_max_abs(
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

            

        dts += dt  # traveltime calculation
        reacts += relative_react * dt  # relative reactivity

        # termination criteria evaluation: whether the particle has reached a termination layer or out of the system
        has_negative_index = negative_index(np.array([layer, row, col]))
        over_index = larger_index(
            np.array([layer, row, col]), np.array(termination.shape)
        )
        term_value = termination[layer, row, col]

        # new loop:
        coords = exit_point

        if (term_value == 1) | has_negative_index | over_index:
            continue_tracking = False

    final_dt = dts
    final_react = reacts
    return final_dt, final_react


@numba.jit(nopython=True, parallel=True)
def cumulate_react(
    particle_coords,
    particle_cells,
    face_velocities,
    gvs,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
    reactivity,
):
    dts = np.empty((particle_coords.shape[0]))
    reacts = np.empty((particle_coords.shape[0]))
    for i in numba.prange(particle_coords.shape[0]):
        initial_cell = particle_cells[i, :]
        initial_position = particle_coords[i, :]
        dt, react = travel_time_cum_reactivity(
            initial_position,
            initial_cell,
            face_velocities,
            gvs,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
            reactivity,
        )
        dts[i] = dt
        reacts[i] = react
    return dts, reacts


def cumulative_reactivity(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    reactivity: np.ndarray,
):
    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
        gwfmodel, model_directory, porosity
    )

    face_velocities = (-1) * face_velocities
    gvs = (-1) * gvs
    particle_coords = particles_starting_location[:, 0:3]
    particle_cells = particles_starting_location[:, 3:].astype(np.int64)
    tr = cumulate_react(
        particle_coords,
        particle_cells,
        face_velocities,
        gvs,
        xedges,
        yedges,
        z_lf,
        z_uf,
        termination,
        reactivity,
    )

    return tr


@numba.guvectorize(
    [
        (
            float64[:, :],
            int64[:, :],
            float64[:, :, :, :],
            float64[:, :, :, :],
            float64[:],
            float64[:],
            float64[:, :, :],
            float64[:, :, :],
            float64[:, :, :],
            float64[:, :, :],
            float64[:, :],
        )
    ],
    "(n,p),(n,p),(f,l,r,c),(g,l,r,c),(v),(b),(l,r,c),(l,r,c),(l,r,c),(l,r,c)->(n,p)",
    target="parallel",
)
def travel_time_gu(
    initial_position,
    initial_cell,
    face_velocities,
    gvs,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
    reactivity,
    result,
):
    """
    Calculates the travel_time and cumulative reactivity of a particle assuming steady-state flow conditions
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
    reactivity: array with relative reactivity values per cell

    return
    t: (sum of dt)
    r: (sum of dr*dt)
    -------

    """
    for i in range(initial_cell.shape[0]):
        cell = initial_cell[i]
        coords = initial_position[i]
        # initializing:
        layer = cell[0]
        row = cell[1]
        col = cell[2]

        continue_tracking = True

        # Initializing the variables:
        dts = 0.0
        reacts = 0.0

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

            # relative reactivity of the cell
            relative_react = reactivity[layer, row, col]

            # velocities at lower coordinate faces:
            v0 = face_velocities[np.array([0, 2, 4]), layer, row, col]
            # velocities at upper coordinate faces:
            v1 = face_velocities[np.array([1, 3, 5]), layer, row, col]

            # current velocities
            v = velocity(coords, v0, gvp, coords_0)

            # Where is it going:
            velocity_gradient_index = np.abs(v0 - v1) > 1e-10 * numba_max_abs(
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

            # # TODO: check if this statement is still necessary!!
            # if dt == np.inf:
            #     break

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

            dts += dt  # traveltime calculation
            reacts += relative_react * dt  # relative reactivity

            # termination criteria evaluation: whether the particle has reached a termination layer or out of the system
            has_negative_index = negative_index(np.array([layer, row, col]))
            over_index = larger_index(
                np.array([layer, row, col]), np.array(termination.shape)
            )
            term_value = termination[layer, row, col]

            if (term_value == 1) | has_negative_index | over_index:
                continue_tracking = False
            # new loop:
            coords = exit_point

        final_dt = dts
        final_react = reacts
        result[i] = np.array([final_dt, final_react, 0.])


def cumulative_gu(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    reactivity: np.ndarray,
):
    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
        gwfmodel, model_directory, porosity
    )

    face_velocities = (-1) * face_velocities
    gvs = (-1) * gvs
    particle_coords = particles_starting_location[:, 0:3]
    particle_cells = particles_starting_location[:, 3:].astype(np.int64)
    tr = np.empty_like(particle_coords).astype(np.float64)
    tr = travel_time_gu(
        particle_coords,
        particle_cells,
        face_velocities,
        gvs,
        xedges,
        yedges,
        z_lf,
        z_uf,
        termination,
        reactivity,
        tr,
    )
    return tr
