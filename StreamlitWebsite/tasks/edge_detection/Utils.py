import numpy as np
import math

def ZeroCrossing(src, ratio=5):
    threshold = np.max(src) * ratio / 100

    dst = np.zeros_like(src, dtype=np.uint8)
    rows, cols = src.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = src[y, x]
            left = src[y, x - 1]
            right = src[y, x + 1]
            top = src[y - 1, x]
            bottom = src[y + 1, x]

            if center == 0:
                continue

            if (left * right < 0 and abs(left - right) <= 7 and abs(left) >= threshold and abs(right) >= threshold) \
                    or (top * bottom < 0 and abs(top - bottom) <= 7 and abs(top) >= threshold and abs(bottom) >= threshold):
                dst[y, x] = 255

    return dst

def NonMaxSuppression(src, dirs):
    dst = np.copy(src)

    rows, cols = src.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            val = src[y, x]
            dir = dirs[y, x]

            if (0 <= dir < 22.5 or 157.5 <= dir <= 180) or (67.5 <= dir < 112.5): # Horizontal
                val1 = src[y, x + 1]
                val2 = src[y, x - 1]
            elif 22.5 <= dir < 67.5 or 112.5 <= dir < 157.5: # Vertical
                val1 = src[y - 1, x]
                val2 = src[y + 1, x]
            elif 67.5 <= dir < 112.5:                       # Diagonal 1
                val1 = src[y - 1, x + 1]
                val2 = src[y + 1, x - 1]
            else:                                           # Diagonal 2
                val1 = src[y - 1, x - 1]
                val2 = src[y + 1, x + 1]
                
            if val <= val1 or val <= val2:
                dst[y, x] = 0
                
    return dst

def CalEdgeDirections(gx, gy, degree=True):
    dirs = np.zeros_like(gx)

    if degree:
        for x in range(gx.shape[1]):
            for y in range(gx.shape[0]):
                dir = math.atan2(gy[y, x], gx[y, x]) * 180 / math.pi
                if dir < 0:
                    dir += 180
                dirs[y, x] = dir
    else:
        for x in range(gx.shape[1]):
            for y in range(gx.shape[0]):
                dir = math.atan2(gy[y, x], gx[y, x])
                if dir < 0:
                    dir += math.pi
                dirs[y, x] = dir

    return dirs

def Hysteresis(g, low_threshold, high_threshold):
    edges = np.zeros_like(g, dtype=np.uint8)

    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]

    rows, cols = g.shape

    def follow_edge(x, y):
        for k in range(8):
            nx = x + dx[k]
            ny = y + dy[k]
            if 0 <= nx < cols and 0 <= ny < rows and edges[ny, nx] == 0:
                if low_threshold <= g[ny, nx] < high_threshold:
                    edges[ny, nx] = 255
                    follow_edge(nx, ny)

    for y in range(rows):
        for x in range(cols):
            if g[y, x] >= high_threshold and edges[y, x] == 0:
                edges[y, x] = 255
                follow_edge(x, y)

    return edges

def FollowEdge(g, edges, x, y, lowThreshold, highThreshold):
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]

    for k in range(8):
        nx = x + dx[k]
        ny = y + dy[k]
        if 0 <= nx < g.shape[1] and 0 <= ny < g.shape[0] and edges[ny, nx] == 0:
            if lowThreshold <= g[ny, nx] < highThreshold:
                edges[ny, nx] = 255
                FollowEdge(g, edges, nx, ny, lowThreshold, highThreshold)