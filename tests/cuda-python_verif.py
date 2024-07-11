# Need to: pip install --upgrade cuda-python

from cuda.cuda import CUdevice_attribute, cuDeviceGetAttribute, cuDeviceGetName, cuInit

# Initialize CUDA Driver API
(err,) = cuInit(0)

# Get attributes
err, DEVICE_NAME = cuDeviceGetName(128, 0)
DEVICE_NAME = DEVICE_NAME.decode("ascii").replace("\x00", "")

err, MAX_THREADS_PER_BLOCK = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0
)
err, MAX_BLOCK_DIM_X = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0
)
err, MAX_GRID_DIM_X = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0
)
err, SMs = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
)
err, TOTAL_GLOBAL_MEM = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_GLOBAL_MEMORY, 0
)
err, MAX_SHARED_MEMORY_PER_BLOCK = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0
)
# Convert TOTAL_GLOBAL_MEM from bytes to gigabytes for easier readability
TOTAL_GLOBAL_MEM_GB = TOTAL_GLOBAL_MEM / (1024 ** 3)

print(f"Device Name: {DEVICE_NAME}")
print(f"Maximum number of multiprocessors: {SMs}")
print(f"Maximum number of threads per block: {MAX_THREADS_PER_BLOCK:10}")
print(f"Maximum number of blocks per grid:   {MAX_BLOCK_DIM_X:10}")
print(f"Maximum number of threads per grid:  {MAX_GRID_DIM_X:10}")
print(f"Total global memory: {TOTAL_GLOBAL_MEM_GB:.2f} GB")
print(f"Maximum shared memory per block: {MAX_SHARED_MEMORY_PER_BLOCK} bytes")

#  Device Name: Tesla T4
#  Maximum number of multiprocessors: 40
#  Maximum number of threads per block:       1024
#  Maximum number of blocks per grid:         1024
#  Maximum number of threads per grid:  2147483647