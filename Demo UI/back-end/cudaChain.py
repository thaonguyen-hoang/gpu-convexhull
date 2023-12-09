from flask import Flask, request
from flask_cors import CORS

from numba import cuda, float32, njit, prange, int32
import cupy as cp
# from ray import dataframe
# import ctypes
# from numpy.lib.recfunctions import structured_to_unstructured
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import random, time, sys
from multiprocessing import Process, Pipe, Queue
random.seed(1)

BLOCK_SIZE = 1024

# Hàm đọc dữ liệu điểm
def import_qhull(filename):
    try:
        with open(filename, 'r') as f:
            Dim, nVertex = map(int, f.readline().split())
            pts_x = np.empty(nVertex)
            pts_y = np.empty(nVertex)

            for i in range(nVertex):
                x, y = map(float, f.readline().split())
                pts_x[i] = x
                pts_y[i] = y

    except IOError:
        print(f"\nCannot Open File ! {filename}")
        exit(1)

    return pts_x, pts_y

def export_off(pts, output):
    try:
        with open(output, 'w') as f:
            nVert = len(pts)

            f.write("OFF\n")
            f.write(f"{nVert} 1 0\n")

            for i in range(nVert):
                x = pts[i][0]
                y = pts[i][1]
                z = 0.0
                f.write(f"{x} {y} {z}\n")

            f.write(f"{nVert} ")
            for i in range(nVert):
                f.write(f"{i} ")

    except IOError:
        print(f"\nCannot Save File ! {output}")
        exit(1)

# isLeft(): tests if a point is Left|On|Right of an infinite line.
# Input: three points P0, P1, and P2
# Return:
#	  >0 for P2 left of the line through P0 and P1
#   =0 for P2 on the line
#   <0 for P2 right of the line
# See: Algorithm 1 on Area of Triangles

@cuda.jit(device=True)
def is_left_coords(P0, P1, P2_x, P2_y):
    return (P1[0] - P0[0]) * (P2_y - P0[1]) - (P2_x - P0[0]) * (P1[1] - P0[1])

@cuda.jit(device=True)
def is_inside_coords(P0, P1, P2, P3, pt_x, pt_y):
    if is_left_coords(P0, P1, pt_x, pt_y) <= 0.0:
        return 1
    if is_left_coords(P1, P2, pt_x, pt_y) <= 0.0:
        return 2
    if is_left_coords(P2, P3, pt_x, pt_y) <= 0.0:
        return 3
    if is_left_coords(P3, P0, pt_x, pt_y) <= 0.0:
        return 4
    return 0

@cuda.jit
def kernel_preprocess(d_extreme_x, d_extreme_y, d_x, d_y, d_pos, n):
    s_extreme_pts = cuda.shared.array(shape=(4, 2), dtype=float32)

    if cuda.grid(1) == 0:
        for t in range(4):
            s_extreme_pts[t][0] = d_extreme_x[t]
            s_extreme_pts[t][1] = d_extreme_y[t]

    cuda.syncthreads()

    i = cuda.grid(1)
    if i < n:
        d_pos[i] = is_inside_coords(s_extreme_pts[0], s_extreme_pts[1], s_extreme_pts[2], s_extreme_pts[3], d_x[i], d_y[i])

BSP2 = 4  # 4 region values from 1 to 4, corresponds to 16 threads per block (2^4)
COMPACT_BLOCK_SIZE = 2**BSP2

#CUDA kernel to calculate prefix sum of each block of input array
@cuda.jit()
def prefix_sum_nzmask_block(a, b, s, nzm, length, value):
    ab = cuda.shared.array(shape=(COMPACT_BLOCK_SIZE), dtype=int32)

    # tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x;
    tid = cuda.grid(1)
    ab[cuda.threadIdx.x] = 0
    if tid < length:
        if nzm == 1:
            ab[cuda.threadIdx.x] = int32(a[tid] == value); #Load mask of input data into shared memory
        else:
            ab[cuda.threadIdx.x] = int32(a[tid]);

    for j in range(0, BSP2):
        i = 2**j
        cuda.syncthreads()
        if i <= cuda.threadIdx.x:
            temp = ab[cuda.threadIdx.x]
            temp += ab[cuda.threadIdx.x - i] #Perform scan on shared memory
        cuda.syncthreads()
        if i <= cuda.threadIdx.x:
            ab[cuda.threadIdx.x] = temp
    if tid < length:
        b[tid] = ab[cuda.threadIdx.x]; #Write scanned blocks to global memory

    if(cuda.threadIdx.x == cuda.blockDim.x-1):  #Last thread of block
        s[cuda.blockIdx.x] = ab[cuda.threadIdx.x]; #Write last element of shared memory into global memory

#CUDA kernel to merge the prefix sums of individual blocks
@cuda.jit()
def pref_sum_update(b, s, length):
    tid = (cuda.blockIdx.x + 1) * cuda.blockDim.x + cuda.threadIdx.x; #Skip first block
    if tid < length:
        b[tid] += s[cuda.blockIdx.x] #Accumulate last elements of all previous blocks

#CUDA kernel to copy non-zero entries to the correct index of the output array
@cuda.jit
def map_non_zeros(a, b, c, prefix_sum, partition, length, value):
    tid = cuda.grid(1)

    if tid < length:
        pos = a[tid]
        if pos == value:
            index = prefix_sum[tid] #The correct output index is the value at current index of prefix sum array
            partition[0][index-1] = b[tid]
            partition[1][index-1] = c[tid]
            partition[2][index-1] = pos

#Recursive prefix sum (inclusive scan)
#a and asum should already be in device memory, nzm determines whether first step is masking values (nzm = 1) or not
def pref_sum(a, prefix_sum, nzm, value):
    block = COMPACT_BLOCK_SIZE
    length = a.shape[0]
    grid = int((length + block -1)/block)
    #Create auxiliary array to hold the sum of each block
    bs = cuda.device_array(shape=(grid), dtype=np.int32)

    #Perform partial scan of each block. Store block sum in auxillary array.
    prefix_sum_nzmask_block[grid, block](a, prefix_sum, bs, nzm, length, value)
    if grid > 1:
        bssum = cuda.device_array(shape=(grid), dtype=np.int32)
        pref_sum(bs, bssum, 0, value) #recursively build complete prefix sum
        pref_sum_update[grid-1, block](prefix_sum, bssum, length)

#Apply stream compaction algorithm to get only the non-zero entries from the input array
def get_non_zeros(d_pos, d_x, d_y):
    #Copy input array from host to device
    # ad = cuda.to_device(d_pos)

    #Create prefix sum output array
    indices1 = cuda.device_array_like(d_pos)
    indices2 = cuda.device_array_like(d_pos)
    indices3 = cuda.device_array_like(d_pos)
    indices4 = cuda.device_array_like(d_pos)

    #Perform full prefix sum
    pref_sum(d_pos, indices1, int(1), int(1))
    pref_sum(d_pos, indices2, int(1), int(2))
    pref_sum(d_pos, indices3, int(1), int(3))
    pref_sum(d_pos, indices4, int(1), int(4))

    #The last element of prefix sum contains the total number of non-zero elements
    ones_count = int(indices1[indices1.shape[0]-1])
    twos_count = int(indices2[indices2.shape[0]-1])
    threes_count = int(indices3[indices3.shape[0]-1])
    fours_count = int(indices4[indices4.shape[0]-1])
    #Create device output array to hold ONLY the non-zero entries
    reg1 = cuda.device_array(shape=(3, ones_count), dtype=np.float32)
    reg2 = cuda.device_array(shape=(3, twos_count), dtype=np.float32)
    reg3 = cuda.device_array(shape=(3, threes_count), dtype=np.float32)
    reg4 = cuda.device_array(shape=(3, fours_count), dtype=np.float32)

    # Copy ONLY the non-zero entries
    block = COMPACT_BLOCK_SIZE
    length = d_pos.shape[0]
    grid = int((length + block -1)/block)
    map_non_zeros[grid, block](d_pos, d_x, d_y, indices1, reg1, length, int(1))
    map_non_zeros[grid, block](d_pos, d_x, d_y, indices2, reg2, length, int(2))
    map_non_zeros[grid, block](d_pos, d_x, d_y, indices3, reg3, length, int(3))
    map_non_zeros[grid, block](d_pos, d_x, d_y, indices4, reg4, length, int(4))

    return reg1, reg2, reg3, reg4

@cuda.jit
def kernelCheck_R1_device(reg1_y, pos):
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp = reg1_y[0]
    for t in range(items):
        i = cuda.threadIdx.x * items + t
        if i < len(pos):
            if reg1_y[i] > tmp:
                pos[i] = 0
            else:
                tmp = reg1_y[i]

@cuda.jit
def kernelCheck_R2_device(reg2_x, pos):
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp = reg2_x[0]
    for t in range(items):
        i = cuda.threadIdx.x * items + t
        if i < len(pos):
            if reg2_x[i] < tmp:
                pos[i] = 0
            else:
                tmp = reg2_x[i]

@cuda.jit
def kernelCheck_R3_device(reg3_y, pos):
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp = reg3_y[0]
    for t in range(items):
        i = cuda.threadIdx.x * items + t
        if i < len(pos):
            if reg3_y[i] < tmp:
                pos[i] = 0
            else:
                tmp = reg3_y[i]

@cuda.jit
def kernelCheck_R4_device(reg4_x, pos):
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp = reg4_x[0]
    for t in range(items):
        i = cuda.threadIdx.x * items + t
        if i < len(pos):
            if reg4_x[i] > tmp:
                pos[i] = 0
            else:
                tmp = reg4_x[i]

# input x, input y, output x, output y
def cuda_chain_new(x, y):
    n = len(x)

    # Sao chép dữ liệu từ host sang device
    d_x = cp.asarray(x)
    d_y = cp.asarray(y)
    d_pos = cp.zeros(n, dtype=np.int32) # which region out of 4

    # Tìm min / max index
    minx_id, maxx_id = d_x.argmin(), d_x.argmax()
    miny_id, maxy_id = d_y.argmin(), d_y.argmax()

    # Lưu trữ bốn điểm cực đại
    d_extreme_x = cp.array([d_x[minx_id], d_x[miny_id], d_x[maxx_id], d_x[maxy_id]])
    d_extreme_y = cp.array([d_y[minx_id], d_y[miny_id], d_y[maxx_id], d_y[maxy_id]])
  
    # # 1st discarding
    blockspergrid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    threadsperblock = BLOCK_SIZE
    kernel_preprocess[blockspergrid, threadsperblock](d_extreme_x, d_extreme_y, d_x, d_y, d_pos, n)

    # Set Extreme Points Specifically
    d_pos[minx_id] = 1  # min X
    d_pos[miny_id] = 2  # min Y
    d_pos[maxx_id] = 3  # max X
    d_pos[maxy_id] = 4  # max Y

    # Partition

    d_reg1 = cp.asarray(get_non_zeros(d_pos, d_x, d_y)[0])
    d_reg2 = cp.asarray(get_non_zeros(d_pos, d_x, d_y)[1])
    d_reg3 = cp.asarray(get_non_zeros(d_pos, d_x, d_y)[2])
    d_reg4 = cp.asarray(get_non_zeros(d_pos, d_x, d_y)[3])

    # Partly sort each region
    # Region 1: sắp xếp tăng dần theo X (giảm dần theo Y)
    indices_sort_r1 = np.argsort(d_reg1[0])
    d_reg1_x = d_reg1[0][indices_sort_r1]
    d_reg1_y = d_reg1[1][indices_sort_r1]

    # Region 2: sắp xếp tăng dần theo Y (tăng dần theo X)
    indices_sort_r2 = np.argsort(d_reg2[1])
    d_reg2_x = d_reg2[0][indices_sort_r2]
    d_reg2_y = d_reg2[1][indices_sort_r2]

    # Region 3: sắp xếp giảm dần theo X (tăng dần theo Y)
    indices_sort_r3 = -np.argsort(-d_reg3[0])
    d_reg3_x = d_reg3[0][indices_sort_r3]
    d_reg3_y = d_reg3[1][indices_sort_r3]

    # Region 4: sắp xếp giảm dần theo Y (giảm dần theo X)
    indices_sort_r4 = -np.argsort(-d_reg4[1])
    d_reg4_x = d_reg4[0][indices_sort_r4]
    d_reg4_y = d_reg4[1][indices_sort_r4]

    # Kernel: vòng 2 của việc loại bỏ
    k = 1
    kernelCheck_R1_device[k, min(BLOCK_SIZE, len(d_reg1[2]))](d_reg1_y, d_reg1[2])
    kernelCheck_R2_device[k, min(BLOCK_SIZE, len(d_reg2[2]))](d_reg2_x, d_reg2[2])
    kernelCheck_R3_device[k, min(BLOCK_SIZE, len(d_reg3[2]))](d_reg3_y, d_reg3[2])
    kernelCheck_R4_device[k, min(BLOCK_SIZE, len(d_reg4[2]))](d_reg4_x, d_reg4[2])

    
    d_r1_left = cp.nonzero(d_reg1[2])
    d_r2_left = cp.nonzero(d_reg2[2])
    d_r3_left = cp.nonzero(d_reg3[2])
    d_r4_left = cp.nonzero(d_reg4[2])

    d_hull_x = cp.append(d_reg1_x[d_r1_left], d_reg2_x[d_r2_left])
    d_hull_x = cp.append(d_hull_x, d_reg3_x[d_r3_left])
    d_hull_x = cp.append(d_hull_x, d_reg4_x[d_r4_left])

    d_hull_y = cp.append(d_reg1_y[d_r1_left], d_reg2_y[d_r2_left])
    d_hull_y = cp.append(d_hull_y, d_reg3_y[d_r3_left])
    d_hull_y = cp.append(d_hull_y, d_reg4_y[d_r4_left])


    n = len(d_r1_left[0]) + len(d_r2_left[0]) + len(d_r3_left[0]) + len(d_r4_left[0])
    return n, d_hull_x, d_hull_y

def plot_points(x, y):
    minx_id, maxx_id = x.argmin(), x.argmax()
    miny_id, maxy_id = y.argmin(), y.argmax()
    extreme_x = [x[minx_id], x[miny_id], x[maxx_id], x[maxy_id], x[minx_id]]
    extreme_y = [y[minx_id], y[miny_id], y[maxx_id], y[maxy_id], y[minx_id]]
    plt.plot(extreme_x, extreme_y, marker='*', color='green')
    plt.scatter(x, y)
    # plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.show()

upperhull_queue=Queue()
lowerhull_queue=Queue()

def distance(start, end, pt):
    '''
    Calculate the perpendicular distance between a point and a line segment.

    Parameters:
    - start (list): The starting point of the line segment.
    - end (list): The ending point of the line segment.
    - pt (list): The point for which the distance is to be calculated.

    Returns:
    - float: The perpendicular distance between the point and the line segment.
    '''
    x1, y1 = start
    x2, y2 = end
    x0, y0 = pt
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = ((y2 - y1)**2 + (x2 - x1) ** 2) ** 0.5
    result = numerator / denominator
    return result

def get_min_max_x(list_pts):
    min_x = float('inf')
    max_x = 0
    min_y = 0
    max_y = 0
    for pt in list_pts:
        x=pt[0]
        y=pt[1]
        if x < min_x:
            min_x = x
            min_y = y
        if x > max_x:
            max_x = x
            max_y = y
    return [min_x,min_y], [max_x,max_y]

def get_furthest_point_from_line(start, end, left_points):
    max_dist = 0

    max_point = []

    for point in left_points:
        if point != start and point != end:
            dist = distance(start, end, point)
            if dist > max_dist:
                max_dist = dist
                max_point = point

    return max_point

def get_points_left_of_line(start, end, listPts):
    left_pts = []
    right_pts = []

    for pt in listPts:
        if isCCW(start, end, pt):
            left_pts.append(pt)
        else:
            right_pts.append(pt)

    return left_pts

def isCCW(start, end, pt):
    '''
    Function checks whether 3 points form a counterclockwise (CCW) orientation.
    The function calculates the cross product of vectors formed by these points to determine their orientation.
    '''
    start_x=start[0]
    start_y=start[1]

    end_x=end[0]
    end_y=end[1]

    pt_x=pt[0]
    pt_y=pt[1]

    val = (pt_y - start_y) * (end_x - start_x) - (end_y - start_y) * (pt_x - start_x);
    if val > 0:
        return True
    else:
        return False

def quickhull(listPts, leftmostpoint, rightmostpoint):

    left_of_line_pts = get_points_left_of_line(leftmostpoint, rightmostpoint, listPts)
    furthest_point = get_furthest_point_from_line(leftmostpoint, rightmostpoint, left_of_line_pts)

    if len(furthest_point) < 1:
        return [rightmostpoint]

    hull_points = quickhull(left_of_line_pts, leftmostpoint, furthest_point)
    hull_points = hull_points + quickhull(left_of_line_pts, furthest_point, rightmostpoint)

    return hull_points

def get_hull_points_sequential(listPts, leftmostpoint, rightmostpoint):

    upperhull = quickhull(listPts, leftmostpoint, rightmostpoint)
    lowerhull = quickhull(listPts, rightmostpoint, leftmostpoint)
    return upperhull + lowerhull

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')

@app.route("/points", methods = ['POST', 'GET'])
def convexhull():
    filename = request.get_json()
    x_host, y_host = import_qhull(filename)
    # x_host = x_host[:1000]
    # y_host = y_host[:100]

    n, d_hull_x, d_hull_y = cuda_chain_new(x_host, y_host)
    # print(n)
    hull_x = cp.asnumpy(d_hull_x)
    hull_y = cp.asnumpy(d_hull_y)
    # print(hull_x)
    # print(hull_y)
    hull = np.append(hull_x, hull_y)
    hull = hull.reshape(n, 2)
    
    # Generate convex hull using QuickHull algorithm
    leftmostpoint, rightmostpoint = get_min_max_x(hull)
    quick_hull_result = get_hull_points_sequential(hull, leftmostpoint, rightmostpoint)
    print(quick_hull_result)
    # plot_points(quick_hull_result)

    # # Export convex hull to an OFF file
    # export_off(np.array(quick_hull_result))

if __name__ == '__main__':
    app.run(debug=True)

