import matplotlib.pyplot as plt
import matplotlib.patches as patches

def import_convex_hull(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    points = [list(map(float, pair.split())) for pair in lines]
    for i in range(len(points)-1):
        segment = (points[i], points[i+1])
        data.append(segment)

    # Add the closing segment
    data.append((points[-1], points[0]))

    return data

# Xác định các điểm nằm trên đường biên bao lồi

# Hàm kiểm tra điểm có nằm trong ô không
def is_in_tile(point, tile):
    x, y = point
    x1, y1, x2, y2 = tile
    return x1 <= x <= x2 and y1 <= y <= y2

# Tìm các ô nằm trên biên convex hull
def find_vertex_tiles(grid_of_tiles, convexhull):
    result = set()
    for tile in grid_of_tiles:
        for segment in convexhull:
            # Kiểm tra xem mỗi điểm của convexhull có nằm trong tile không
            if any(is_in_tile(point, tile) for point in segment):
                result.add(tile)
    return list(result)


# Xác định hướng cắt của một segment đối với một ô

# Xét xem 3 điểm có nằm ngược chiều kim đồng hồ không
# Kiểm tra xem hai đoạn thẳng có giao nhau không
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Kiểm tra segment có cắt cạnh nào của ô không, nếu có cắt 1 cạnh bất kỳ, dừng lại.
# Đồng thời trả về hướng cắt của của segment đó đối với ô
def segment_intersects_tile(segment, tile):
    A = segment[0]
    B = segment[1]

    x_min, y_min, x_max, y_max = tile

    C_left = (x_min, y_min)
    D_left = (x_max - 1, y_max - 1)

    C_right = (x_min + 1, y_min)
    D_right = (x_max, y_max)

    C_top = (x_min, y_min + 1)
    D_top = (x_max, y_max)

    C_bottom = (x_min, y_min)
    D_bottom = (x_max, y_max - 1)

    # Kiểm tra giao điểm của đoạn và các cạnh của tile
    if ccw(A, C_left, D_left) != ccw(B, C_left, D_left) and ccw(A, B, C_left) != ccw(A, B, D_left):
        return 'left'

    if ccw(A, C_right, D_right) != ccw(B, C_right, D_right) and ccw(A, B, C_right) != ccw(A, B, D_right):
        return 'right'

    if ccw(A, C_top, D_top) != ccw(B, C_top, D_top) and ccw(A, B, C_top) != ccw(A, B, D_top):
        return 'top'

    if ccw(A, C_bottom, D_bottom) != ccw(B, C_bottom, D_bottom) and ccw(A, B, C_bottom) != ccw(A, B, D_bottom):
        return 'bottom'
    return 'no_intersection'

def find_the_next_tile(segment, current_tile):
    exit_direction = segment_intersects_tile(segment, current_tile)
    if exit_direction == 'left':
        return (current_tile[0] - 1, current_tile[1], current_tile[2] - 1, current_tile[3])
    elif exit_direction == 'right':
        return (current_tile[0] + 1, current_tile[1], current_tile[2] + 1, current_tile[3])
    elif exit_direction == 'top':
        return (current_tile[0], current_tile[1] + 1, current_tile[2], current_tile[3] + 1)
    elif exit_direction == 'bottom':
        return (current_tile[0], current_tile[1] - 1, current_tile[2], current_tile[3] - 1)
    else:
        return current_tile

def find_inside_tiles(border_tiles, rows):
  left_most_tile = min(border_tiles, key=lambda tile: tile[0])
  right_most_tile = max(border_tiles, key=lambda tile: tile[2])

  # Tìm inside_tile
  inside_tile = []
  for x in range(left_most_tile[0], right_most_tile[2]):
      for y in range(rows):
        y_min_border_tiles = min(tile[1] for tile in border_tiles if tile[0] <= x <= tile[2])
        y_max_border_tiles = max(tile[3] for tile in border_tiles if tile[0] <= x <= tile[2])
        cur_tile = (x, y, x + 1, y + 1)

        if (cur_tile not in border_tiles) and (y_min_border_tiles < y) and (y+1 < y_max_border_tiles):
            # Kiểm tra xem cur_tile có chứa trong các tiles của res không
          inside_tile.append(cur_tile)
  return inside_tile

def row_col_max(convexhull):
  row_max = 0
  for segment in convexhull:
      if len(segment) >= 2:  
          row_max_segment = max(segment[0][0], segment[1][0])
          row_max = max(row_max, row_max_segment)
  col_max = 0
  for segment in convexhull:
      if len(segment) >= 2: 
          col_max_segment = max(segment[0][1], segment[1][1])
          col_max = max(col_max, col_max_segment)
  return int(row_max)+1, int(col_max)+1

# Hàm biểu diễn hình học
def plot_grid_and_movement(grid_of_tiles, border_tiles, convexhull):
    fig, ax = plt.subplots()

    # Vẽ ô lưới
    for tile in grid_of_tiles:
        rectangle = patches.Rectangle((tile[0], tile[1]), tile[2] - tile[0], tile[3] - tile[1], linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rectangle)

    # Tô màu các ô cần tìm
    for border_tile in border_tiles:
        rectangle = patches.Rectangle((border_tile[0], border_tile[1]), border_tile[2] - border_tile[0], border_tile[3] - border_tile[1], linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rectangle)

    # Biểu diễn convex hull
    for segment in convexhull:
        x_values = [point[0] for point in segment]
        y_values = [point[1] for point in segment]
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')

    ax.set_xlim(min(point[0] for segment in convexhull for point in segment) - 1, int(max(point[0] for segment in convexhull for point in segment)) + 1)
    ax.set_ylim(min(point[1] for segment in convexhull for point in segment) - 1, int(max(point[1] for segment in convexhull for point in segment)) + 1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# test
def main():
  filepath = "" # Điền file convex hull
  convexhull = import_convex_hull()

  rows, cols = row_col_max(convexhull)
  print(rows)
  print(cols)
  grid_of_tiles = []
  for row in range(rows+1):
      for col in range(cols+1):
          tile = (col, row, col + 1, row + 1)
          grid_of_tiles.append(tile)

  vertex_tiles = find_vertex_tiles(grid_of_tiles, convexhull)
  beside_tiles = []
  for tile in vertex_tiles:
      for segment in convexhull:
          if is_in_tile(segment[0], tile) or is_in_tile(segment[1], tile):
              next_tile = find_the_next_tile(segment, tile)
              if next_tile not in beside_tiles:
                  beside_tiles.append(next_tile)

  border_tiles = vertex_tiles + beside_tiles
  inside_tiles = find_inside_tiles(border_tiles, rows)

  res = border_tiles + inside_tiles
  
  print("Các ô giao với bao lồi: ", res)
  plot_grid_and_movement(grid_of_tiles, res, convexhull)

if __name__ == "__main__":
  main()

