"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 02:09:19
 ----------------------------------------------------


"""

import numpy as np
import sys
import os
import cv2


#################
# create triangles from point cloud(grid)
#################


def gen_triangle(mask):
    ## 1. prepare index for triangle
    index_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    index = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            if (mask[i, j] == 0):
                continue

            index_map[i, j] = index
            index += 1

    ## 2. create flags for triangle
    mask_bool = np.array(mask, dtype='bool')

    ## flag1
    left_up = mask_bool.copy()
    left_up[1:mask.shape[0], :] *= mask_bool[0:mask.shape[0] - 1, :]  # multiply up movement
    left_up[:, 1:mask.shape[1]] *= mask_bool[:, 0:mask.shape[1] - 1]  # multiply left movement
    left_up[0, :] = False
    left_up[:, 0] = False

    ## flag2
    right_down = mask_bool.copy()
    right_down[0:mask.shape[0] - 1, :] *= mask_bool[1:mask.shape[0], :]  # multiply down movement
    right_down[:, 0:mask.shape[1] - 1] *= mask_bool[:, 1:mask.shape[1]]  # multiply right movement
    right_down[mask.shape[0] - 1, :] = False
    right_down[:, mask.shape[1] - 1] = False

    '''
      (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
          |              |               |
          |              |               |
          |              |               |
       (i, j-1) ------ (i, j) ------ (i, j+1)
          |              |               |
          |              |               |
          |              |               |
      (i+1, j-1) ----(i+1, j)------(i+1, j+1)
  
    flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
    flag1 means: Δ{ (i, j), (i+1, j), (i, j+1) }
  
    otherwise:
      case1: is not locate on edge(i, j ==0) and exist left up point
      --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
  
      case2: is not locate on edge(i, j ==shape-1) and exist right down
      --> Δ{ (i, j), (i+1, j+1), (i, j+1) }
  
    '''

    ## 3. fill triangle list like above
    triangle = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            ## outside --> ignore
            if (not (mask_bool[i, j])):
                continue

            ## flag1
            if (left_up[i, j]):
                triangle.append([index_map[i, j], index_map[i - 1, j], index_map[i, j - 1]])

            ## flag2
            if (right_down[i, j]):
                triangle.append([index_map[i, j], index_map[i + 1, j], index_map[i, j + 1]])

            ## otherwise
            if (not (left_up[i, j]) and not (right_down[i, j])):

                ## case1
                if (i != 0 and j != 0 and mask_bool[i, j - 1] and mask_bool[i - 1, j - 1]):
                    triangle.append([index_map[i, j], index_map[i - 1, j - 1], index_map[i, j - 1]])

                ## case2
                if (i != mask_bool.shape[0] - 1 and j != mask_bool.shape[1] - 1 and mask_bool[i, j + 1] and mask_bool[
                    i + 1, j + 1]):
                    triangle.append([index_map[i, j], index_map[i + 1, j + 1], index_map[i, j + 1]])

    return np.array(triangle, dtype=np.int64)


def Depth2VerTri(depth, mask=None):
    if mask is None:
        mask = np.ones(depth.shape[:2], dtype=np.uint8) * 255

    ## 1. prepare index for triangle and vertex array
    index_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    index = 0
    vertex = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            # print(i, j, depth[i, j], type(depth[i, j]), depth[i, j].shape if hasattr(depth[i, j], "shape") else "")

            if (mask[i, j] == 0):
                continue

            index_map[i, j] = index
            index += 1

            vertex.append([i, j, depth[i, j, 0]])

    ## 2. create flags for triangle
    mask_bool = np.array(mask, dtype='bool')

    ## flag1
    left_up = mask_bool.copy()
    left_up[1:mask.shape[0], :] *= mask_bool[0:mask.shape[0] - 1, :]  # multiply up movement
    left_up[:, 1:mask.shape[1]] *= mask_bool[:, 0:mask.shape[1] - 1]  # multiply left movement
    left_up[0, :] = False
    left_up[:, 0] = False

    ## flag2
    right_down = mask_bool.copy()
    right_down[0:mask.shape[0] - 1, :] *= mask_bool[1:mask.shape[0], :]  # multiply down movement
    right_down[:, 0:mask.shape[1] - 1] *= mask_bool[:, 1:mask.shape[1]]  # multiply right movement
    right_down[mask.shape[0] - 1, :] = False
    right_down[:, mask.shape[1] - 1] = False

    '''
      (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
          |              |               |
          |              |               |
          |              |               |
       (i, j-1) ------ (i, j) ------ (i, j+1)
          |              |               |
          |              |               |
          |              |               |
      (i+1, j-1) ----(i+1, j)------(i+1, j+1)
  
    flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
    flag1 means: Δ{ (i, j), (i+1, j), (i, j+1) }
  
    otherwise:
      case1: not on edge(i, j ==0) and exist left up point
      --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
  
      case2: not on edge(i, j ==shape-1) and exist right down
      --> Δ{ (i, j), (i+1, j+1), (i, j+1) }
  
    '''

    ## 3. fill triangle list like above
    triangle = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            ## outside --> ignore
            if (not (mask_bool[i, j])):
                continue

            ## flag1
            if (left_up[i, j]):
                triangle.append([index_map[i, j], index_map[i - 1, j], index_map[i, j - 1]])

            ## flag2
            if (right_down[i, j]):
                triangle.append([index_map[i, j], index_map[i + 1, j], index_map[i, j + 1]])

            ## otherwise
            if (not (left_up[i, j]) and not (right_down[i, j])):

                ## case1
                if (i != 0 and j != 0 and mask_bool[i, j - 1] and mask_bool[i - 1, j - 1]):
                    triangle.append([index_map[i, j], index_map[i - 1, j - 1], index_map[i, j - 1]])

                ## case2
                if (i != mask_bool.shape[0] - 1 and j != mask_bool.shape[1] - 1 and mask_bool[i, j + 1] and mask_bool[
                    i + 1, j + 1]):
                    triangle.append([index_map[i, j], index_map[i + 1, j + 1], index_map[i, j + 1]])

    return np.array(vertex, dtype=np.float32), np.array(triangle, dtype=np.int64)


def save_as_ply2(filename, depth, normal, albedo, mask=None, triangle=None,
                 fx=1.0, fy=1.0, cx=0.0, cy=0.0):
    if mask is None:
        mask = np.ones(normal.shape[:2], dtype=np.uint8) * 255

    mask_bool = mask.astype(bool)
    Np = np.count_nonzero(mask_bool)

    # Asegura uint8 y rango correcto para color
    albedo_uint8 = np.clip(albedo, 0, 255).astype(np.uint8)

    with open(filename, 'w') as fp:
        fp.write('ply\n')
        fp.write('format ascii 1.0\n')
        fp.write(f'element vertex {Np}\n')
        fp.write('property float x\n')
        fp.write('property float y\n')
        fp.write('property float z\n')
        fp.write('property float nx\n')
        fp.write('property float ny\n')
        fp.write('property float nz\n')
        fp.write('property uchar red\n')
        fp.write('property uchar green\n')
        fp.write('property uchar blue\n')

        if triangle is not None:
            fp.write(f'element face {len(triangle)}\n')
            fp.write('property list uchar int vertex_indices\n')

        fp.write('end_header\n')

        idx_map = -np.ones_like(mask, dtype=int)
        idx = 0
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if mask_bool[i, j]:
                    z = float(depth[i, j])
                    x = (j - cx) * z / fx
                    y = (i - cy) * z / fy
                    fp.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {} {}\n'.format(
                        x, y, z,
                        float(normal[i, j, 0]), float(normal[i, j, 1]), float(normal[i, j, 2]),
                        int(albedo_uint8[i, j, 0]), int(albedo_uint8[i, j, 1]), int(albedo_uint8[i, j, 2])
                    ))
                    idx_map[i, j] = idx
                    idx += 1

        if triangle is not None:
            for t in triangle:
                fp.write(f'3 {t[0]} {t[1]} {t[2]}\n')


# def Depth2VerTri(depth, mask=None):
#     if mask is None:
#         mask = np.ones(depth.shape[:2], dtype=np.uint8) * 255

#     h, w = depth.shape[:2]
#     cx, cy = w / 2, h / 2
#     pixel_size = 1.0  # ajusta según corresponda

#     index_map = np.zeros((h, w), dtype=np.int64)
#     index = 0
#     vertex = []

#     for i in range(h):
#         for j in range(w):
#             if mask[i, j] == 0:
#                 continue

#             index_map[i, j] = index
#             index += 1

#             x = (j - cx) * pixel_size
#             y = (cy - i) * pixel_size  # invierte Y para coord. más intuitiva
#             z = float(depth[i, j].item()) if hasattr(depth[i,j], "item") else float(depth[i,j])
#             vertex.append([x, y, z])

#     mask_bool = np.array(mask, dtype=bool)

#     left_up = mask_bool.copy()
#     left_up[1:, :] = left_up[1:, :] & mask_bool[:-1, :]
#     left_up[:, 1:] = left_up[:, 1:] & mask_bool[:, :-1]
#     left_up[0, :] = False
#     left_up[:, 0] = False

#     right_down = mask_bool.copy()
#     right_down[:-1, :] = right_down[:-1, :] & mask_bool[1:, :]
#     right_down[:, :-1] = right_down[:, :-1] & mask_bool[:, 1:]
#     right_down[-1, :] = False
#     right_down[:, -1] = False

#     triangle = []
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):

#             if not mask_bool[i, j]:
#                 continue

#             if left_up[i, j]:
#                 triangle.append([index_map[i, j], index_map[i - 1, j], index_map[i, j - 1]])

#             if right_down[i, j]:
#                 triangle.append([index_map[i, j], index_map[i + 1, j], index_map[i, j + 1]])

#             if not left_up[i, j] and not right_down[i, j]:
#                 if i != 0 and j != 0 and mask_bool[i, j - 1] and mask_bool[i - 1, j - 1]:
#                     triangle.append([index_map[i, j], index_map[i - 1, j - 1], index_map[i, j - 1]])

#                 if i != mask_bool.shape[0] - 1 and j != mask_bool.shape[1] - 1 and mask_bool[i, j + 1] and mask_bool[i + 1, j + 1]:
#                     triangle.append([index_map[i, j], index_map[i + 1, j + 1], index_map[i, j + 1]])

#     return np.array(vertex, dtype=np.float32), np.array(triangle, dtype=np.int64)


#################
# for .obj
#################


def loadobj(path):
    vertices = []
    # texcoords = []
    triangles = []
    normals = []

    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue

            pieces = line.split(' ')

            if pieces[0] == 'v':
                vertices.append([float(x) for x in pieces[1:4]])
                # elif pieces[0] == 'vt':
            #   texcoords.append([float(x) for x in pieces[1:]])
            elif pieces[0] == 'f':
                if pieces[1] == '':
                    triangles.append([int(x.split('/')[0]) - 1 for x in pieces[2:]])
                else:
                    triangles.append([int(x.split('/')[0]) - 1 for x in pieces[1:]])
            elif pieces[0] == 'vn':
                normals.append([float(x) for x in pieces[1:]])
            else:
                pass

    return (np.array(vertices, dtype=np.float32),
            # np.array(texcoords, dtype=np.float32),
            np.array(triangles, dtype=np.int32))  # ,
    # np.array(normals, dtype=np.float32))


def writeobj(filepath, vertices, triangles):
    with open(filepath, "w") as f:
        for i in range(vertices.shape[0]):
            f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        for i in range(triangles.shape[0]):
            f.write("f {} {} {}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))


def writeobj_with_uv(filepath, vertices, triangles, uv):
    basename = os.path.basename(filepath)
    with open(filepath, "w") as f:
        f.write("mtllib " + basename + "_material.mtl\n")

        for i in range(vertices.shape[0]):
            f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        for i in range(uv.shape[0]):
            f.write("vt {} {}\n".format(uv[i, 0], uv[i, 1]))

        f.write("usemtl tex\n")
        # f.write("s off\n")
        for i in range(triangles.shape[0]):
            f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))

    with open(filepath + "_material.mtl", "w") as f:
        # f.write("newmtl emerald\n")
        # f.write("Ns 600\n")
        # f.write("d 1\n")
        # f.write("Ni 0.001\n")
        # f.write("illum 2\n")
        # f.write("Ka 0.0215  0.1745   0.0215\n")
        # f.write("Kd 0.07568 0.61424  0.07568\n")
        # f.write("Ks 0.633   0.727811 0.633\n")
        # f.write("map_Kd sand.jpg\n")

        f.write("newmtl tex\n")
        f.write("Ns 96.078431\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 0.640000 0.640000 0.640000\n")
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("map_Kd sand.jpg\n")


def writepoint(filepath, vertices):
    with open(filepath, "w") as f:
        for i in range(vertices.shape[0]):
            f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        # for i in range(triangles.shape[0]):
        #   f.write("f {} {} {}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))


def writeoff(filepath, vertices, triangles):
    with open(filepath, "w") as f:
        f.write("OFF\n")
        f.write("# convert by tsukasa\n")
        f.write("\n")
        f.write("{} {} {}\n".format(vertices.shape[0], triangles.shape[0], 0))
        for i in range(vertices.shape[0]):
            f.write("{} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        for i in range(triangles.shape[0]):
            f.write("3 {} {} {}\n".format(triangles[i, 0], triangles[i, 1], triangles[i, 2]))


#################
# for .ply
#################

# def readply():


def save_as_ply(filename, depth, normal, albedo, mask=None, triangle=None):
    if mask is None:
        mask = np.ones(normal.shape[:2], dtype=np.uint8) * 255

    mask_bool = np.array(mask, dtype='bool')
    Np = np.count_nonzero(mask)
    # bgr = np.array(albedo * 255, dtype=np.uint8)
    bgr = np.array(albedo, dtype=np.uint8)

    ## write ply file
    with open(filename, 'w') as fp:

        ## header infomation
        fp.write('ply\n')
        fp.write('format ascii 1.0\n')
        fp.write('element vertex {0}\n'.format(Np))
        fp.write('property float x\n')
        fp.write('property float y\n')
        fp.write('property float z\n')
        fp.write('property float nx\n')
        fp.write('property float ny\n')
        fp.write('property float nz\n')
        fp.write('property uchar blue\n')
        fp.write('property uchar green\n')
        fp.write('property uchar red\n')
        fp.write('element face {0}\n'.format(len(triangle)))
        fp.write('property list uchar int vertex_indices\n')
        fp.write('end_header\n')

        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):

                if (not (mask_bool[i, j])):
                    continue

                # fp.write('{0:e} {1:e} {2:e} {3:e} {4:e} {5:e} {6:d} {7:d} {8:d}\n'.format(i, j, depth[i, j], normal[i, j, 0], normal[i, j, 1], normal[i, j, 2], bgr[i, j, 0], bgr[i, j, 1], bgr[i, j, 2]))
                fp.write('{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:d} {7:d} {8:d}\n'.format(
                    float(i), float(j),
                    float(depth[i, j, 0]),
                    float(normal[i, j, 0]),
                    float(normal[i, j, 1]),
                    float(normal[i, j, 2]),
                    int(bgr[i, j, 0]),
                    int(bgr[i, j, 1]),
                    int(bgr[i, j, 2])
                ))

        for i in range(len(triangle)):
            fp.write('3 {0:d} {1:d} {2:d}\n'.format(triangle[i][0], triangle[i][1], triangle[i][2]))

        # for i in range(depth.shape[0]):
        #     for j in range(depth.shape[1]):
        #         if not mask_bool[i, j]:
        #             continue

        #         d = depth[i, j]
        #         if hasattr(d, 'item'):
        #             d = d.item()
        #         d = float(d)

        #         nx = normal[i, j, 0]
        #         ny = normal[i, j, 1]
        #         nz = normal[i, j, 2]
        #         if hasattr(nx, 'item'):
        #             nx = nx.item()
        #             ny = ny.item()
        #             nz = nz.item()
        #         nx, ny, nz = float(nx), float(ny), float(nz)

        #         r = bgr[i, j, 0]
        #         g = bgr[i, j, 1]
        #         b = bgr[i, j, 2]
        #         if hasattr(r, 'item'):
        #             r = r.item()
        #             g = g.item()
        #             b = b.item()
        #         r, g, b = int(r), int(g), int(b)

        #         fp.write(f"{i:e} {j:e} {d:e} {nx:e} {ny:e} {nz:e} {b:d} {g:d} {r:d}\n")
        # for i in range(len(triangle)):
        #   fp.write('3 {0:d} {1:d} {2:d}\n'.format(triangle[i][0], triangle[i][1], triangle[i][2]))
