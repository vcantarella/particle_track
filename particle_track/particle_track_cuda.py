import math
import flopy
import numpy as np
from numba import boolean, cuda, float64, int16, int32

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
def particle_kernel(
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
        j = 0

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

        result[i, 0] = dts
        result[i, 1] = reacts
        result[i, 2] = error


def particle_track(
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
    # host arrays for the results:
    particle_locs = np.ones((particles_starting_location.shape[0], termination.size, 3), dtype = np.float64)*na_value
    dts = np.ones((particles_starting_location.shape[0], termination.size), dtype = np.float64)*na_value

    # sending fixed arrays to the device
    # everything here has to be in the device memory, because the shared memory is very small compared to normal modflow arrays
    face_velocities = cuda.to_device(face_velocities)
    xedges = cuda.to_device(xedges)
    yedges = cuda.to_device(yedges)
    z_lf = cuda.to_device(z_lf)
    z_uf = cuda.to_device(z_uf)
    termination = cuda.to_device(termination)
    # Defining cells and particle coordinates
    particle_coords = particles_starting_location[:, 0:3].copy()
    particle_coords = cuda.to_device(particle_coords)
    particle_cells = particles_starting_location[:, 3:].copy().astype(np.int32)
    particle_cells = cuda.to_device(particle_cells)
    
    # declaring the threads and blocks (VODOO for now, need to be optimized)

    threadsperblock = 256

    # lets say I have a grid too large to fit all the results in the device memory. Then I need to optimize the blocks for that.
    max_memory = cuda.current_context().get_memory_info()[1]
    max_memory = max_memory - particle_coords.size * 8 - particle_cells.size * 4\
         - termination.size * 4 - reactivity.size * 8 - face_velocities.size * 8\
             - xedges.size * 8 - yedges.size * 8 - z_lf.size * 8 - z_uf.size * 8
    
    

    # memory conservative blocks per grid estimation:
    size_per_particle = termination.size * 8 * 2
    max_particles_parallel = math.floor(max_memory / size_per_particle)
    for threads in np.array([32, 64, 128, 256, 512, 1024]):
        blocks_memory = math.floor(max_particles_parallel / threads)
        if blocks_memory > 0:
            blockspergrid = blocks_memory
            threadsperblock = threads
            break

    blocks_grid = math.ceil(particles_starting_location.shape[0] / threadsperblock)
    blockspergrid = min(blocks_grid, blocks_memory)

    # Pin memory
    with cuda.pinned(particle_locs, dts):
        stream = cuda.stream()

        dev_plocs = cuda.to_device(particle_locs, stream=stream)
        dev_dts = cuda.to_device(dts, stream=stream)
        particle_kernel[blockspergrid, threadsperblock, stream](
            particle_coords,
            particle_cells,
            face_velocities,
            xedges,
            yedges,
            z_lf,
            z_uf,
            termination,
            dev_plocs,
            dev_dts,
        )

        dev_plocs.copy_to_host(particle_locs, stream=stream)
        dev_dts.copy_to_host(dts, stream=stream)
    stream.synchronize()
    
    particle_locs = particle_locs[particle_locs != na_value]
    dts = dts[dts != na_value]

    return particle_locs, dts