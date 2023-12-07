import random, time, sys
from multiprocessing import Process, Pipe, Queue
random.seed(1)

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

def get_hull_points_parallel(listPts, leftmostpoint, rightmostpoint, hull_queue):

    convexhull_points_parallel = quickhull(listPts, leftmostpoint, rightmostpoint)
    hull_queue.put(convexhull_points_parallel)

def parallel_quickhull(points):

    leftmostpoint, rightmostpoint = get_min_max_x(points)

    upperhull_proc = Process(target=get_hull_points_parallel, args=(points, leftmostpoint, rightmostpoint, upperhull_queue))
    lowerhull_proc = Process(target=get_hull_points_parallel, args=(points, rightmostpoint, leftmostpoint, lowerhull_queue))

    start = time.time()

    upperhull_proc.start()
    lowerhull_proc.start()

    elapsed = time.time() - start

    upper_convexhull_points = upperhull_queue.get()
    lower_convexhull_points = lowerhull_queue.get()
    convexhull_points = upper_convexhull_points + lower_convexhull_points

    return elapsed

def get_hull_points_sequential(listPts, leftmostpoint, rightmostpoint):

    upperhull = quickhull(listPts, leftmostpoint, rightmostpoint)
    lowerhull = quickhull(listPts, rightmostpoint, leftmostpoint)
    return upperhull + lowerhull

def sequential_quickhull(points):

    leftmostpoint, rightmostpoint = get_min_max_x(points)

    start = time.time()
    convexhull_points_sequential = get_hull_points_sequential(points, leftmostpoint, rightmostpoint)
    elapsed = time.time() - start

    return elapsed

if __name__ == "__main__":
    total_points_list = [1000,100000,200000,500000,1000000,2000000,4000000,5000000]
    output_time_dict={} ##{"50":"1\t2"}

    fin = 'more_i.txt'
    f = open(fin, 'r')
    lng = []
    lat = []
    n = 0
    for line in f:
        coord = line.strip().split()
        lng.append(float(coord[0]))
        lat.append(float(coord[1]))
        n += 1

    for total_point in total_points_list:
        points = []
        for i in range(total_point):
            x = lng[i]
            y = lat[i]
            points.append([x, y])
        #print(points)
        #seqQuickHull(points)
        #parallelQuickHull(points)

        sequential_time = sequential_quickhull(points)
        parallel_time = parallel_quickhull(points)


        key=str(total_point)
        val = "\t\t"+str(sequential_time).rjust(12)+"\t\t"+str(parallel_time).rjust(12)
        output_time_dict[key]=val

    print("Total Points\tSequential Time(sec)\tParallel Time(sec)")

    for key,val in output_time_dict.items():
        print(key.rjust(12),val)