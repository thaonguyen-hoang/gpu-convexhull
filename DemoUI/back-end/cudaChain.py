import numpy as np
from flask import request, Flask
from numba import cuda
import cupy as cp
import matplotlib.pyplot as plt

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

    return pts_x, pts_y, nVertex


# Hàm viết kết quả ra file output
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


# Các hàm phụ trợ (thực hiện trên device) chuẩn bị cho tiền xử lý lần 1
@cuda.jit(device=True)
def is_left(P0, P1, P2_x, P2_y):
    '''
    Kiểm tra xem 1 điểm nằm bên Trái|Phải|Trên 1 đường thẳng

    Input: 2 điểm P0, P1, và tọa độ x, y của điểm P2
    Return:
        >0 nếu P2 nằm bên trái so với đường thẳng tạo bởi P0 và P1
        =0 nếu P2 nằm trên đường thẳng (3 điểm thẳng hàng)
        <0 nếu P2 nằm bên phải đường thẳng
    '''
    return (P1[0] - P0[0]) * (P2_y - P0[1]) - (P2_x - P0[0]) * (P1[1] - P0[1])


@cuda.jit(device=True)
def is_inside(P0, P1, P2, P3, pt_x, pt_y):
    '''
    Phân vùng các điểm vào vị trí tương ứng:
    R1 = các điểm nằm bên trái Pxmin-Pymin
    R2 = các điểm nằm bên trái Pymin-Pxmax
    R3 = các điểm nằm bên trái Pxmax-Pymax
    R4 = các điểm nằm bên trái Pymax-Pxmin
    R0 = các điểm không thỏa mãn 4 vùng trên (các điểm nằm bên trong)

    Input: 4 điểm Pxmin, Pymin, Pxmax, Pymax được đánh số lần lượt P0, P1, P2, P3, và tọa độ x, y của điểm cần xét
    Return:
        1 nếu nằm ở vùng R1
        2 nếu nằm ở vùng R2
        3 nếu nằm ở vùng R3
        4 nếu nằm ở vùng R4
        0 nếu nằm ở vùng R0
    '''
    if is_left(P0, P1, pt_x, pt_y) <= 0.0:
        return 1
    if is_left(P1, P2, pt_x, pt_y) <= 0.0:
        return 2
    if is_left(P2, P3, pt_x, pt_y) <= 0.0:
        return 3
    if is_left(P3, P0, pt_x, pt_y) <= 0.0:
        return 4
    return 0


'''
(GPU)
Tiền xử lý lần 1: Phân loại các điểm thành các vùng R1, R2, R3, R4, R0, 
trong đó R0 là các điểm nằm bên trong hình tạo bởi 4 điểm cực đại

Thiết kế: 1 kernel cho toàn bộ tập điểm
    số luồng/1 block = BLOCK_SIZE = 1024
    số block/1 grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
'''


@cuda.jit
def kernel_preprocess(d_extreme_pts, d_x, d_y, d_pos, n):
    '''
    Input:  d_extreme_pts: mảng 2 chiều chứa tọa độ các điểm cực đại Pxmin, Pymin, Pxmax, Pymax
            d_x: mảng chứa tọa độ x của tập điểm
            d_y: mảng chứa tọa độ y của tập điểm
            d_pos: mảng trống ghi lại vùng của các điểm sau khi phân loại
            n: tổng số điểm (độ dài mảng d_x, d_y, d_pos)

    Mỗi thread thực hiện phân vùng các điểm ở vị trí tương đương trong từng grid
    (sử dụng vòng lặp for với bước nhảy stride = blockDim.x * gridDim.x)
    '''
    offset = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(offset, n, stride):
        d_pos[i] = is_inside(d_extreme_pts[0], d_extreme_pts[1], d_extreme_pts[2], d_extreme_pts[3], d_x[i], d_y[i])


# Các hàm phụ trợ chuẩn bị cho phân vùng
@cuda.reduce
def sum_reduce(a, b):
    '''
    Tính tổng các phần tử trong mảng sử dụng phương pháp song song
    (See also: Parallel Reduction)
    '''
    return a + b


@cuda.jit
def mask(d_pos, d_mask, value, n):
    '''
    Tạo mask cho mảng để đếm số lần xuất hiện của từng vùng (đánh số 1 nếu là vùng cần xét, 0 nếu không phải)
    Input:
        d_pos: mảng chứa vị trí các vùng của tập điểm
        d_mask: mảng chứa vị trí sau khi mask
        value: giá trị để so sánh (giá trị của từng vùng)
        n: tổng số điểm (độ dài mảng d_pos)

    Mỗi thread xử lý các điểm ở vị trí tương đương trong từng grid,
    kiểm tra d_pos[i] có bằng giá trị value đã cho, ghi kết quả vào d_mask[i] = 1 nếu true, 0 nếu false
    '''
    offset = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(offset, n, stride):
        d_mask[i] = d_pos[i] == value


def partition(d_pos, value):
    '''
    Đếm số các điểm trong mỗi vùng
    '''
    n = len(d_pos)

    # Tạo mask (0/1) cho các điểm trong d_pos
    d_mask = cuda.device_array(n)
    block = BLOCK_SIZE
    grid = (n + block - 1) // block
    mask[grid, block](d_pos, d_mask, value, n)

    # Đếm số lần xuất hiện của từng vị trí = tổng các giá trị trong d_mask
    counter = sum_reduce(d_mask)
    return counter


'''
(GPU)
Tiền xử lý lần 2: Tiếp tục loại bỏ các điểm nằm bên trong

Thiết kế: 4 kernel, mỗi kernel cho 1 trong 4 vùng lần lượt là R1, R2, R3, R4 với:
    số block = 1
    số luồng/1 block = BLOCK_SIZE = 1024
Mỗi thread thực hiện kiểm tra liên tiếp cho (n + BLOCK_SIZE - 1) // BLOCK_SIZE điểm, với n là tổng số điểm cần xét
'''


@cuda.jit
def kernelCheck_R1_device(reg1_y, pos):
    '''
    Kiểm tra các điểm trong vùng R1 (được sắp xếp theo thứ tự tọa độ x tăng dần)
    Một điểm Pi nếu tọa độ y > tọa độ y của điểm trước đó P(i-1) thì nằm bên trong, sẽ được gán giá trị pos[i] = 0
                    tọa độ y < tọa độ y của P(i-1), gán giá trị điểm mốc tmp = y(i-1) và tiếp tục xét các điểm tiếp theo
    Input:
        reg1_y: tọa độ y của các điểm thuộc vùng R1
        pos: vị trí các điểm của R1 (pos[i] = 1 với mọi i)
    '''
    # Số điểm mà 1 thread cần xử lý
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tx = cuda.threadIdx.x

    # Vị trí điểm đầu tiên lấy làm mốc so sánh cho từng thread
    tmp = reg1_y[tx * items]
    # Lần lượt xét các phần tử tiếp theo so với phần tử đầu tiên
    for t in range(BLOCK_SIZE):
        i = tx * items + t
        if i < len(pos):
            if reg1_y[i] - tmp > 0.0:
                pos[i] = 0
            else:
                tmp = reg1_y[i]


@cuda.jit
def kernelCheck_R2_device(reg2_x, pos):
    '''
    Kiểm tra các điểm trong vùng R2 (được sắp xếp theo thứ tự tọa độ y tăng dần)
    Một điểm Pi nếu tọa độ x < tọa độ x của điểm trước đó P(i-1) thì nằm bên trong, sẽ được gán giá trị pos[i] = 0
                    tọa độ x > tọa độ x của P(i-1), gán giá trị điểm mốc tmp = x(i-1) và tiếp tục xét các điểm tiếp theo
    Input:
        reg2_x: tọa độ x của các điểm thuộc vùng R2
        pos: vị trí các điểm của R2 (pos[i] = 2 với mọi i)
    '''
    # Số điểm mà 1 thread cần xử lý
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tx = cuda.threadIdx.x

    # Vị trí điểm đầu tiên lấy làm mốc so sánh cho từng thread
    tmp = reg2_x[tx * items]
    # Lần lượt xét các phần tử tiếp theo so với phần tử đầu tiên
    for t in range(BLOCK_SIZE):
        i = tx * items + t
        if i < len(pos):
            if reg2_x[i] - tmp < 0.0:
                pos[i] = 0
            else:
                tmp = reg2_x[i]


@cuda.jit
def kernelCheck_R3_device(reg3_y, pos):
    '''
    Kiểm tra các điểm trong vùng R3 (được sắp xếp theo thứ tự tọa độ x giảm dần)
    Một điểm Pi nếu tọa độ y < tọa độ y của điểm trước đó P(i-1) thì nằm bên trong, sẽ được gán giá trị pos[i] = 0
                    tọa độ y > tọa độ y của P(i-1), gán giá trị điểm mốc tmp = y(i-1) và tiếp tục xét các điểm tiếp theo
    Input:
        reg3_y: tọa độ y của các điểm thuộc vùng R3
        pos: vị trí các điểm của R3 (pos[i] = 3 với mọi i)
    '''
    # Số điểm mà 1 thread cần xử lý
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tx = cuda.threadIdx.x

    # Vị trí điểm đầu tiên lấy làm mốc so sánh cho từng thread
    tmp = reg3_y[tx * items]
    # Lần lượt xét các phần tử tiếp theo so với phần tử đầu tiên
    for t in range(BLOCK_SIZE):
        i = tx * items + t
        if i < len(pos):
            if reg3_y[i] - tmp < 0.0:
                pos[i] = 0
            else:
                tmp = reg3_y[i]


@cuda.jit
def kernelCheck_R4_device(reg4_x, pos):
    '''
    Kiểm tra các điểm trong vùng R4 (được sắp xếp theo thứ tự tọa độ y giảm dần)
    Một điểm Pi nếu tọa độ x > tọa độ x của điểm trước đó P(i-1) thì nằm bên trong, sẽ được gán giá trị pos[i] = 0
                    tọa độ x < tọa độ x của P(i-1), gán giá trị điểm mốc tmp = x(i-1) và tiếp tục xét các điểm tiếp theo
    Input:
        reg4_x: tọa độ x của các điểm thuộc vùng R4
        pos: vị trí các điểm của R4 (pos[i] = 4 với mọi i)
    '''
    # Số điểm mà 1 thread cần xử lý
    items = (len(pos) + BLOCK_SIZE - 1) // BLOCK_SIZE
    tx = cuda.threadIdx.x

    # Vị trí điểm đầu tiên lấy làm mốc so sánh cho từng thread
    tmp = reg4_x[tx * items]
    # Lần lượt xét các phần tử tiếp theo so với phần tử đầu tiên
    for t in range(BLOCK_SIZE):
        i = tx * items + t
        if i < len(pos):
            if reg4_x[i] - tmp > 0.0:
                pos[i] = 0
            else:
                tmp = reg4_x[i]


'''
(Hàm CPU gọi tới các kernel GPU)
Giai đoạn Tiền xử lý
Lần 1: 
    - Xác định 4 điểm cực đại Pxmin, Pymin, Pxmax, Pymax
    - Phân bố các điểm thành 5 vùng R0, R1, R2, R3, R4 và loại bỏ các điểm nằm trong (các điểm thuộc R0)
Lần 2:
    - Sắp xếp các điểm của mỗi vùng lần lượt theo quy luật
        + R1: tọa độ X tăng dần
        + R2: tọa độ Y tăng dần
        + R3: tọa độ X giảm dần
        + R4: tọa độ Y giảm dần
    - Tiếp tục kiểm tra và loại bỏ các điểm nằm bên trong

Input: Tọa độ x, y của tập điểm ban đầu
Return: Tọa độ x, y của tập các điểm sau khi xử lý, trở thành input để dựng bao lồi
'''


def preprocess(x, y):
    n = len(x)

    # Sao chép dữ liệu từ bộ nhớ của CPU (host) sang GPU (device)
    d_x = cp.asarray(x)
    d_y = cp.asarray(y)
    # Tạo mảng rỗng có độ dài n chứa các giá trị biểu diễn vùng được phân loại của các điểm ()
    d_pos = cp.zeros(n, dtype=np.int32)

    # Tìm chỉ số min/max theo tọa độ x/y
    minx_id, maxx_id = d_x.argmin(), d_x.argmax()
    miny_id, maxy_id = d_y.argmin(), d_y.argmax()

    # Xác định và lưu trữ bốn điểm cực đại Pxmin, Pymin, Pxmax, Pymax
    d_extreme_pts = cp.array([[d_x[minx_id], d_y[minx_id]],
                              [d_x[miny_id], d_y[miny_id]],
                              [d_x[maxx_id], d_y[maxx_id]],
                              [d_x[maxy_id], d_y[maxy_id]]])

    '''
    Tiền xử lý lần 1: Gọi tới kernel_preprocess() thực hiện loại bỏ các điểm nằm bên trong hình tứ giác P
                      tạo bởi 4 điểm Pxmin, Pymin, Pxmax, Pymax
    Thiết kế: 1 kernel cho toàn bộ tập điểm
              số luồng/1 block = BLOCK_SIZE = 1024
              số block/1 grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    Mỗi thread thực hiện kiểm tra cho các điểm ở vị trí tương đương trong từng grid 
    '''
    blockspergrid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    threadsperblock = BLOCK_SIZE
    kernel_preprocess[blockspergrid, threadsperblock](d_extreme_pts, d_x, d_y, d_pos, n)

    # Đánh số cụ thể cho vùng của các điểm cực đại
    d_pos[minx_id] = 1  # Pminx
    d_pos[miny_id] = 2  # Pminy
    d_pos[maxx_id] = 3  # Pmaxx
    d_pos[maxy_id] = 4  # Pmaxy

    # Lọc lấy các điểm nằm bên ngoài hình P (các điểm nằm trong R1, R2, R3, R4)
    outer_order = cp.nonzero(d_pos)
    d_outer_pos = d_pos[outer_order]
    d_outer_x = d_x[outer_order]
    d_outer_y = d_y[outer_order]
    outer_count = len(d_outer_pos)

    # Sắp xếp toàn bộ tập điểm theo vị trí tăng dần (R1->R2->R3->R4)
    pos_sort_order = cp.argsort(d_outer_pos)
    d_outer_pos = d_outer_pos[pos_sort_order]
    d_outer_x = d_outer_x[pos_sort_order]
    d_outer_y = d_outer_y[pos_sort_order]

    # Đếm số lần xuất hiện của R1/R2/R3/R4 (số các điểm trong từng vùng)
    countR4 = int(partition(d_outer_pos, 4))
    countR3 = int(partition(d_outer_pos[:(outer_count - countR4)], 3))
    countR2 = int(partition(d_outer_pos[:(outer_count - countR4 - countR3)], 2))
    countR1 = outer_count - countR4 - countR3 - countR2

    # Partly sort each region
    # Region 1: sắp xếp tăng dần theo X
    indices_sort_r1 = cp.argsort(d_outer_x[:countR1])
    d_reg1_x = d_outer_x[:countR1][indices_sort_r1]
    d_reg1_y = d_outer_y[:countR1][indices_sort_r1]

    # Region 2: sắp xếp tăng dần theo Y
    last_of_R2 = countR1 + countR2
    indices_sort_r2 = cp.argsort(d_outer_y[countR1:last_of_R2])
    d_reg2_x = d_outer_x[countR1:last_of_R2][indices_sort_r2]
    d_reg2_y = d_outer_y[countR1:last_of_R2][indices_sort_r2]

    # Region 3: sắp xếp giảm dần theo X
    last_of_R3 = outer_count - countR4
    indices_sort_r3 = cp.argsort(-d_outer_x[last_of_R2:last_of_R3])
    d_reg3_x = d_outer_x[last_of_R2:last_of_R3][indices_sort_r3]
    d_reg3_y = d_outer_y[last_of_R2:last_of_R3][indices_sort_r3]

    # Region 4: sắp xếp giảm dần theo Y
    indices_sort_r4 = cp.argsort(-d_outer_y[last_of_R3:])
    d_reg4_x = d_outer_x[last_of_R3:][indices_sort_r4]
    d_reg4_y = d_outer_y[last_of_R3:][indices_sort_r4]

    '''
    Tiền xử lý lần 2: Gọi tới kernelCheck của từng vùng sau khi sắp xếp các điểm theo quy luật riêng biệt, 
                      tiếp tục loại bỏ các điểm nằm trong (gán bằng R0)
    Thiết kế: 4 kernel, mỗi kernel cho 1 trong 4 vùng lần lượt là R1, R2, R3, R4 với
              số block = 1
              số luồng/1 block = BLOCK_SIZE = 1024
    Mỗi thread thực hiện kiểm tra liên tiếp cho (n + BLOCK_SIZE - 1) // BLOCK_SIZE điểm, với n là tổng số điểm cần xét
    '''
    k = 1  # số block/1 grid
    kernelCheck_R1_device[k, BLOCK_SIZE](d_reg1_y, d_outer_pos[:countR1])
    kernelCheck_R2_device[k, BLOCK_SIZE](d_reg2_x, d_outer_pos[countR1:last_of_R2])
    kernelCheck_R3_device[k, BLOCK_SIZE](d_reg3_y, d_outer_pos[last_of_R2:last_of_R3])
    kernelCheck_R4_device[k, BLOCK_SIZE](d_reg4_x, d_outer_pos[last_of_R3:])

    # Lọc bỏ các điểm nằm bên trong, giữ lại các điểm không thuộc R0
    d_r1_left = cp.nonzero(d_outer_pos[:countR1])
    d_r2_left = cp.nonzero(d_outer_pos[countR1:last_of_R2])
    d_r3_left = cp.nonzero(d_outer_pos[last_of_R2:last_of_R3])
    d_r4_left = cp.nonzero(d_outer_pos[last_of_R3:])

    # Ghép các tọa độ x của từng vùng R1/R2/R3/R4 vào 1 mảng d_hull_x
    d_hull_x = cp.append(d_reg1_x[d_r1_left], d_reg2_x[d_r2_left])
    d_hull_x = cp.append(d_hull_x, d_reg3_x[d_r3_left])
    d_hull_x = cp.append(d_hull_x, d_reg4_x[d_r4_left])

    # Ghép các tọa độ y của từng vùng R1/R2/R3/R4 vào 1 mảng d_hull_y
    d_hull_y = cp.append(d_reg1_y[d_r1_left], d_reg2_y[d_r2_left])
    d_hull_y = cp.append(d_hull_y, d_reg3_y[d_r3_left])
    d_hull_y = cp.append(d_hull_y, d_reg4_y[d_r4_left])

    # Copy dữ liệu từ GPU sang CPU
    hull_x = cp.asnumpy(d_hull_x)
    hull_y = cp.asnumpy(d_hull_y)

    return hull_x, hull_y


'''
(CPU)
Tính toán và xây dựng bao lồi từ tập điểm sau khi xử lý sử dụng thuật toán (Andrew's) Monotone Chain

Input: Tập các điểm đã được sắp xếp tăng dần theo tọa độ x, sau đó theo tọa độ y
Return: Tập các đỉnh của bao lồi xếp theo thứ tự ngược chiều kim đồng hồ, bắt đầu từ điểm có tọa độ x bé nhất
'''


def is_ccw(start, end, pt):
    '''
    Kiểm tra xem nếu đi lần lượt qua 3 điểm cho trước có quay một góc ngược chiều kim đồng hồ (counterclock wise)

    Input: 3 điểm start, end, pt
    Return:
        >0 nếu xoay góc ngược chiều kim đồng hồ
        =0 nếu đi thẳng
        <0 nếu xoay góc xuôi chiều kim đồng hồ
    '''
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]
    pt_x = pt[0]
    pt_y = pt[1]
    return (pt_y - start_y) * (end_x - start_x) - (end_y - start_y) * (pt_x - start_x)


def simple_hull_2d(points):
    '''
    Dựng bao lồi từ tập các điểm đã sắp xếp theo thứ tự ngược chiều kim đồng hồ, bắt đầu từ điểm có tọa độ x bé nhất.

    Xét lần lượt các điểm, nếu điểm Pi được xét tạo với 2 điểm P(i-1) và P(i-2) một đường thẳng đi theo hướng ngược chiều kim đồng hồ
    thì điểm đó là 1 đỉnh của bao lồi, ngược lại thì là một điểm nằm trong và được loại bỏ khỏi tập đỉnh đầu ra
    '''
    if len(points) <= 3:
        return points

    ch = []
    for p in points:
        while len(ch) >= 2 and is_ccw(ch[-2], ch[-1], p) <= 0:
            ch.pop()
        ch.append(p)
    return ch


def plot_points(convex_hull, org_x, org_y):
    '''
    Đồ thị minh họa kết quả dựng bao lồi
    Input:
        convex_hull: tập các đỉnh của bao lồi
        org_x: tọa độ x của tập điểm ban đầu
        org_y: tọa độ y của tập điểm ban đầu
    '''
    convex_hull.append(convex_hull[0])
    x, y = zip(*convex_hull)
    plt.plot(x, y, 'r')
    plt.scatter(org_x, org_y)
    plt.show()

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')

@app.route("/points", methods = ['POST', 'GET'])
def convexhull():
    filename = request.get_json()
    # Đọc dữ liệu đầu vào từ file input
    x_host, y_host, n = import_qhull(filename)

    # Tiền xử lý
    hull_x, hull_y = preprocess(x_host, y_host)

    # Tạo mảng mới 2 chiều chứa các điểm là các các cặp tọa độ [x, y]
    hull = [[hull_x[i], hull_y[i]] for i in range(len(hull_x))]

    # Dựng bao lồi
    convex_hull = simple_hull_2d(hull)

    # Vẽ đồ thị minh họa
    plot_points(convex_hull, x_host, y_host)

    # Export kết quả ra file output.txt
    export_off(convex_hull, 'output.txt')

if __name__ == '__main__':
    app.run(debug=True)

