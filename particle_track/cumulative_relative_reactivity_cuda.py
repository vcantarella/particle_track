import math

import flopy
import numpy as np
from numba import boolean, cuda, float64, int16, int32

from .preprocessing import prepare_arrays
eps = np.finfo(np.float64).eps


@cuda.jit(int16(float64, float64, float64), device=True, inline=True)
def exit_direction_cuda(v1, v2, v):
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
    r = 0
    if (v1 > 0.0) & (v2 > 0.0):
        r = 1
    elif (v1 < -0.0) & (v2 < -0.0):
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


@cuda.jit(
    float64(
        int16, boolean, float64, float64, float64, float64, float64, float64, float64
    ),
    device=True,
    fastmath=False,
    inline=True,
)
def reach_time_cuda(exit_ind, gradient_logic, v0, v1, v, gv, x, left_x, right_x):
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
        if gradient_logic:
            v0 = abs(v0)
            v = abs(v)
            tx = math.log(v0/v) / gv# - math.log(v) / gv
        else:
            tx = (-1) * (x - left_x) / v
    else:  # exit_ind == 1
        if gradient_logic:
            tx = math.log(v1/v) / gv# - math.log(v) / gv
        else:
            tx = (right_x - x) / v
    return tx


@cuda.jit(int16(float64, float64, float64), device=True)
def argmin_cuda(dt_x, dt_y, dt_z):
    if dt_x <= dt_y:
        if dt_x <= dt_z:
            return 0
        else:
            return 2
    elif dt_y <= dt_z:
        return 1
    else:
        return 2


@cuda.jit(
    float64(
        int16,
        boolean,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        boolean,
    ),
    device=True,
)
def exit_location_cuda(exit_ind, gradient_logic, dt, v0, v1, v, gv, x, left_x, right_x, is_exit):
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
    if is_exit:
        if exit_ind == 1:
            x_new = right_x
        else:
            x_new = left_x
    elif abs(v) > eps*abs(v):
        if not (gradient_logic):
            x_new = x + v * dt
        elif exit_ind == 1:
            x_new = left_x + (1 / gv) * (v * math.exp(gv * dt) - v0)
        elif exit_ind == -1:
            x_new = right_x + (1 / gv) * (v * math.exp(gv * dt) - v1)
        else:
            x_new = left_x + (1 / gv) * (v * math.exp(gv * dt) - v0)
    else:
        x_new = x

    return x_new


@cuda.jit(boolean(int32, int32, int32), device=True)
def negative_index_cuda(ind_x, ind_y, ind_z):
    if ind_x < 0:
        return True
    elif ind_y < 0:
        return True
    elif ind_z < 0:
        return True
    else:
        return False


@cuda.jit(boolean(int32, int32, int32, int32, int32, int32), device=True)
def larger_index_cuda(test_x, test_y, test_z, reference_x, reference_y, reference_z):
    if test_x > reference_x-1:
        return True
    elif test_y > reference_y-1:
        return True
    elif test_z > reference_z-1:
        return True
    else:
        return False


def travel_time_kernel(
    initial_position,
    initial_cell,
    face_velocities,
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
    # initializing:
    start = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # This calculation gives the total number of threads in the entire grid
    stride = cuda.gridsize(
        1
    )  # 1 = one dimensional thread grid, returns a single value.
    # This Numba-provided convenience function is equivalent to
    # `cuda.blockDim.x * cuda.gridDim.x`

    # This thread will start work at the data element index equal to that of its own
    # unique index in the grid, and then, will stride the number of threads in the grid each
    # iteration so long as it has not stepped out of the data's bounds. In this way, each
    # thread may work on more than one data element, and together, all threads will work on
    # every data element.
    for i in range(start, initial_position.shape[0], stride):
        cell = initial_cell[i, :]
        coords = initial_position[i, :]

        layer = cell[0]
        row = cell[1]
        col = cell[2]
        x = coords[0]
        y = coords[1]
        z = coords[2]

        continue_tracking = True

        # Initializing the variables:
        dts = 0.0
        reacts = 0.0
        count = 0  # error count
        max_count = termination.shape[0] * termination.shape[1] * termination.shape[2]
        error = 0

        while continue_tracking:
            # coordinates at lower and upper faces:
            left_x = xedges[col]
            right_x = xedges[col + 1]
            low_y = yedges[row + 1]
            top_y = yedges[row]
            bt_z = z_lf[layer, row, col]
            up_z = z_uf[layer, row, col]

            # relative reactivity of the cell
            relative_react = reactivity[layer, row, col]

            # velocities at lower coordinate faces:
            v0x = face_velocities[0, layer, row, col]
            v0y = face_velocities[2, layer, row, col]
            v0z = face_velocities[4, layer, row, col]
            # velocities at upper coordinate faces:
            v1x = face_velocities[1, layer, row, col]
            v1y = face_velocities[3, layer, row, col]
            v1z = face_velocities[5, layer, row, col]

            # gradients for the cell
            gvpx = (v1x - v0x) / (right_x - left_x)
            gvpy = (v1y - v0y) / (top_y - low_y)
            gvpz = (v1z - v0z) / (up_z - bt_z)

            # current velocities
            vx = gvpx * (x - left_x) + v0x
            vy = gvpy * (y - low_y) + v0y
            vz = gvpz * (z - bt_z) + v0z

            # Where is it going:
            velocity_gradient_x = abs(v0x - v1x) > eps * max(abs(v0x), abs(v1x))
            velocity_gradient_y = abs(v0y - v1y) > eps * max(abs(v0y), abs(v1y))
            velocity_gradient_z = abs(v0z - v1z) > eps * max(abs(v0z), abs(v1z))

            # Exit direction:
            exit_direction_x = exit_direction_cuda(v0x, v1x, vx)
            exit_direction_y = exit_direction_cuda(v0y, v1y, vy)
            exit_direction_z = exit_direction_cuda(v0z, v1z, vz)

            if (exit_direction_x == 0) and (exit_direction_y == 0) and (exit_direction_z == 0):
                error = 1
                break
            # Time to reach each end
            dt_x = reach_time_cuda(
                exit_direction_x,
                velocity_gradient_x,
                v0x,
                v1x,
                vx,
                gvpx,
                x,
                left_x,
                right_x,
            )
            dt_y = reach_time_cuda(
                exit_direction_y,
                velocity_gradient_y,
                v0y,
                v1y,
                vy,
                gvpy,
                y,
                low_y,
                top_y,
            )
            dt_z = reach_time_cuda(
                exit_direction_z, velocity_gradient_z, v0z, v1z, vz, gvpz, z, bt_z, up_z
            )

            # actual travel time:
            dt = min(dt_x, dt_y, dt_z)
            if dt == np.inf:
                break
            exit_point_loc = argmin_cuda(dt_x, dt_y, dt_z)
            exit_x = False
            exit_y = False
            exit_z = False
            if exit_point_loc == 0:
                exit_x = True
            elif exit_point_loc == 1:
                exit_y = True
            else:
                exit_z = True

            # calculate exit point coordinates
            exit_point_x = exit_location_cuda(
                exit_direction_x,
                velocity_gradient_x,
                dt,
                v0x,
                v1x,
                vx,
                gvpx,
                x,
                left_x,
                right_x,
                exit_x,
            )
            exit_point_y = exit_location_cuda(
                exit_direction_y,
                velocity_gradient_y,
                dt,
                v0y,
                v1y,
                vy,
                gvpy,
                y,
                low_y,
                top_y,
                exit_y,
            )
            exit_point_z = exit_location_cuda(
                exit_direction_z,
                velocity_gradient_z,
                dt,
                v0z,
                v1z,
                vz,
                gvpz,
                z,
                bt_z,
                up_z,
                exit_z,
            )

            if exit_point_loc == 0:
                col = col + exit_direction_x
            if exit_point_loc == 1:
                row = row - exit_direction_y
            if exit_point_loc == 2:
                layer = layer - exit_direction_z

            dts += dt  # traveltime calculation
            reacts += relative_react * dt  # relative reactivity

            # termination criteria evaluation: whether the particle has reached a termination layer or out of the system
            has_negative_index = negative_index_cuda(layer, row, col)
            over_index = larger_index_cuda(
                layer,
                row,
                col,
                termination.shape[0],
                termination.shape[1],
                termination.shape[2],
            )
            if has_negative_index:
                continue_tracking = False
            if over_index:
                continue_tracking = False

            term_value = termination[layer, row, col]
            if term_value == 1:
                continue_tracking = False

            # new loop:
            x = exit_point_x
            y = exit_point_y
            z = exit_point_z
            count += 1
            if count > max_count:
                error = 1
                continue_tracking = False

        result[i, 0] = dts
        result[i, 1] = reacts
        result[i, 2] = error


def cumulative_cuda(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    reactivity: np.ndarray,
    debug: bool = False,
):
    """
    Cumulative reactivity model (Loschko et al 2016) implemented in CUDA for graphics card calculation.
    """
    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
        gwfmodel, model_directory, porosity
    )
    # Reverting the velocities field the tracking direction is backwards:
    face_velocities = (-1) * face_velocities
    face_velocities = cuda.to_device(face_velocities)
    xedges = cuda.to_device(xedges)
    yedges = cuda.to_device(yedges)
    z_lf = cuda.to_device(z_lf)
    z_uf = cuda.to_device(z_uf)
    termination = cuda.to_device(termination)
    reactivity = cuda.to_device(reactivity)
    # Defining cells and particle coordinates
    particle_coords = particles_starting_location[:, 0:3].copy()
    particle_coords = cuda.to_device(particle_coords)
    particle_cells = particles_starting_location[:, 3:].copy().astype(np.int32)
    particle_cells = cuda.to_device(particle_cells)
    # initializing the result array
    tr = cuda.device_array((particles_starting_location.shape[0], 3))
    # invoking the kernel:
    with cuda.defer_cleanup():
        threadsperblock = 256
        blockspergridgs = 22 * 80
        # blockspergrid = (particle_cells.shape[0] + (threadsperblock - 1)) // threadsperblock
        kernel = cuda.jit(travel_time_kernel, debug=debug, opt=not (debug))
        kernel[blockspergridgs, threadsperblock](
            particle_coords,
            particle_cells,
            face_velocities,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
            reactivity,
            tr,
        )
    result = tr.copy_to_host()
    result = result
    return result
