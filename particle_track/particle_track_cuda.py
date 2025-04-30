import math
import flopy
import numpy as np
from numba import cuda

from .preprocessing import prepare_arrays_cuda
from .cumulative_relative_reactivity_cuda import (
    exit_direction_cuda,
    reach_time_cuda,
    argmin_cuda,
    exit_location_cuda,
    negative_index_cuda,
    larger_index_cuda,
)


@cuda.jit()
def count_steps_kernel(
    initial_position,
    initial_cell,
    face_velocities,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
    reactivity,
    steps,
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
    steps: array with the number of steps taken by each particle

    return
    steps: array with the number of steps taken by each particle
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
        #j = 0

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
            velocity_gradient_x = abs(v0x - v1x) > 1e-10 * max(abs(v0x), abs(v1x))
            velocity_gradient_y = abs(v0y - v1y) > 1e-10 * max(abs(v0y), abs(v1y))
            velocity_gradient_z = abs(v0z - v1z) > 1e-10 * max(abs(v0z), abs(v1z))

            # Exit direction:
            exit_direction_x = exit_direction_cuda(v0x, v1x, vx)
            exit_direction_y = exit_direction_cuda(v0y, v1y, vy)
            exit_direction_z = exit_direction_cuda(v0z, v1z, vz)

            if (exit_direction_x == 0) and (exit_direction_y == 0) and (exit_direction_z == 0):
                continue_tracking = False
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
            #end while
        steps[i] = count
        # end for
    # end def



@cuda.jit(debug=True)
def tracking_kernel(
    initial_position,
    initial_cell,
    face_velocities,
    xedges,
    yedges,
    z_lf,
    z_uf,
    termination,
    reactivity,
    cum_steps,
    tracks,
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
    tracks: array with the array positions of each particle

    return
    tracks: array with the array positions of each particle
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
        # dts = 0.0
        # reacts = 0.0
        count = 0  # error count
        max_count = termination.shape[0] * termination.shape[1] * termination.shape[2]
        error = 0
        j = 0
        base_index = cum_steps[i]

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
            velocity_gradient_x = abs(v0x - v1x) > 1e-10 * max(abs(v0x), abs(v1x))
            velocity_gradient_y = abs(v0y - v1y) > 1e-10 * max(abs(v0y), abs(v1y))
            velocity_gradient_z = abs(v0z - v1z) > 1e-10 * max(abs(v0z), abs(v1z))

            # Exit direction:
            exit_direction_x = exit_direction_cuda(v0x, v1x, vx)
            exit_direction_y = exit_direction_cuda(v0y, v1y, vy)
            exit_direction_z = exit_direction_cuda(v0z, v1z, vz)

            if (exit_direction_x == 0) and (exit_direction_y == 0) and (exit_direction_z == 0):
                continue_tracking = False
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

            # dts += dt  # traveltime calculation
            # reacts += relative_react * dt  # relative reactivity

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

            tracks[base_index+j,1] = x
            tracks[base_index+j,2] = y
            tracks[base_index+j,3] = z
            tracks[base_index+j,4] = layer
            tracks[base_index+j,5] = row
            tracks[base_index+j,6] = col
            tracks[base_index+j,7] = dt
            tracks[base_index+j,8] = relative_react*dt
            j += 1
            count += 1
            if count > max_count:
                error = 1
                continue_tracking = False

def pollock_cuda(
    gwfmodel: flopy.mf6.MFModel,
    model_directory: str,
    particles_starting_location: np.ndarray,
    porosity: float | np.ndarray,
    reactivity: np.ndarray,
    debug: bool = False,
    mode: str = "forward",
    na_value: float = -9999.0,
):
    """
    Particle Tracking (Pollock, 1988) implemented in Numba CUDA.
    """
    xedges, yedges, z_lf, z_uf, face_velocities, termination = prepare_arrays_cuda(
        gwfmodel, model_directory, porosity
    )
    # Reverting the velocities field the tracking direction is backwards:
    if mode == "backward":
        face_velocities = (-1) * face_velocities
    
    # sending fixed arrays to the device
    # everything here has to be in the device memory, because the shared memory is very small compared to normal modflow arrays
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
    steps = np.zeros(particles_starting_location.shape[0], dtype=np.int32)
    steps = cuda.to_device(steps)
    
    # Calculate proper grid dimensions based on number of particles
    threadsperblock = 256
    blockspergrid = 22 * 80
    
    # Launch the counting kernel with correct grid dimensions
    with cuda.defer_cleanup():
        count_steps_kernel[blockspergrid, threadsperblock](
            particle_coords,
            particle_cells,
            face_velocities,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
            reactivity,
            steps,
        )
        
        steps = steps.copy_to_host()
        print(steps)
        cum_steps = np.cumsum(steps)
        cum_steps = np.concatenate([np.array([0]),cum_steps[:-1]], dtype= np.int32)
        cum_steps = cuda.to_device(cum_steps)
        
        # create the tracks arrays:
        tracks = np.vstack([np.hstack((np.ones((steps[i],1))*i, np.zeros((steps[i], 8))), dtype=np.float64) for i in range(steps.shape[0])])
        print(tracks.shape)
        assert tracks.shape[0] == np.sum(steps), "The number of particles in the tracks array is not equal to the number of particles in the steps array"
        assert steps.shape[0] == particles_starting_location.shape[0], "The number of particles in the steps array is not equal to the number of particles in the starting location array"
        tracks = cuda.to_device(tracks)
        
        # Launch the tracking kernel with correct grid dimensions
        tracking_kernel[blockspergrid, threadsperblock](
            particle_coords,
            particle_cells,
            face_velocities,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
            reactivity,
            cum_steps,
            tracks,
        )
        
        tracks = tracks.copy_to_host()
    return tracks