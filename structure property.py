# -*- coding: UTF-8 -*-
'''
structure property.py
Script to compute the structure property of the whole system.
Use conditions: [1] Cuboid sample
                [2] Aperiodic boundary
Usage: python structure-property.py -scenario 1000 -d50 0.02
'''
# Reference:
# [1] Qi Wang* & Anubhav Jain*. A transferable machine learning framework linking interstice distribution and plastic heterogeneity in metallic glasses.
# [2] E.D.Cubuk* R.J.S.Ivancic* S.S.Schoenholz*.Structure-property relationships from universal signatures of plasticity in disordered solids.
# [3] E. D. Cubuk,1,∗ S. S. Schoenholz (Equal contribution),2,† J. M. Rieser. Identifying structural ﬂow defects in disordered solids using machine learning methods.
from __future__ import division
import re
import math
import os
import pyvoro
import boo
import requests
import openpyxl
import pandas as pd
import numpy as np
from numba import jit
from sys import argv, exit
from scipy import special
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, ConvexHull


def mkdir(path_write):
    # 判断目录是否存在
    # 存在：True
    # 不存在：False
    folder = os.path.exists(path_write)
    # 判断结果
    if not folder:
        # 如果不存在，则创建新目录
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        # 如果目录已存在，则不创建，提示目录已存在
        print('目录已存在')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@jit(nopython=True)
def compute_cos_ijk(posi, posj, posk):
    # 计算向量ij与ik的夹角的cos值
    eij = np.array([posj[0] - posi[0], posj[1] - posi[1], posj[2] - posi[2]])
    eik = np.array([posk[0] - posi[0], posk[1] - posi[1], posk[2] - posi[2]])
    cos = np.dot(eij, eik) / (np.linalg.norm(eij) * np.linalg.norm(eik))
    return cos


@jit(nopython=True)
def compute_dis(posj, posk):
    # 计算三维空间中两点的距离
    ejk = np.array([posk[0] - posj[0], posk[1] - posj[1], posk[2] - posj[2]])
    dis = np.linalg.norm(ejk)
    return dis


@jit(nopython=True)
def compute_tetrahedron_volume(vertice1, vertice2, vertice3, vertice4):
    # 计算四面体的体积，通过给定四面体的四个顶点
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    return abs(np.dot(eil, np.cross(eij, eik))) / 6


@jit(nopython=True)
def compute_solide_angle(vertice1, vertice2, vertice3, vertice4):
    # 计算固体角
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    len_eij = np.linalg.norm(eij)
    len_eik = np.linalg.norm(eik)
    len_eil = np.linalg.norm(eil)
    return 2 * math.atan2(abs(np.dot(eij, np.cross(eik, eil))),
                          (len_eij * len_eik * len_eil + np.dot(eij, eik) * len_eil
                           + np.dot(eij, eil) * len_eik + np.dot(eik, eil) * len_eij))


@jit(nopython=True)
def compute_simplice_area(vertice1, vertice2, vertice3):
    # 计算三角形的面积，通过给定的三个顶点
    # problem1: compute error -> eij = eik causes error.
    # neglect
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    h = (np.dot(eij, eij) - (np.dot(eij, eik) / (np.linalg.norm(eik))) ** 2) ** 0.5
    return (np.linalg.norm(eik)) * h / 2


def calc_beta_rad(pvec):
    """
    polar angle [0, pi]
    """
    return np.arccos(pvec[2])  # arccos:[0, pi]


def calc_gamma_rad(pvec):
    """
    azimuth angle [0, 2pi]
    """
    gamma = np.arctan2(pvec[1], pvec[0])
    if gamma < 0.0:
        gamma += 2 * np.pi
    return gamma


def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0 + (a[2] - b[2]) ** 2.0)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@jit(nopython=True)
def compute_angular_element(neigh_id_input, distance_input, points_input, a_list, b_list, c_list, radius, like_input,
                            neigh_id_length_index_input):
    angular_value_in = np.empty_like(like_input)
    for a in range(len(neigh_id_length_index_input)):
        value1 = 0.0
        value2 = 0.0
        value3 = 0.0
        value4 = 0.0
        value5 = 0.0
        value6 = 0.0
        value7 = 0.0
        value8 = 0.0
        value9 = 0.0
        value10 = 0.0
        value11 = 0.0
        value12 = 0.0
        value13 = 0.0
        value14 = 0.0
        value15 = 0.0
        value16 = 0.0
        value17 = 0.0
        value18 = 0.0
        value19 = 0.0
        value20 = 0.0
        value21 = 0.0
        value22 = 0.0
        for b in range(neigh_id_length_index_input[a]):
            for k_ in range(neigh_id_length_index_input[a] - b - 1):
                k = k_ + b + 1
                rij = distance_input[a][b]
                rik = distance_input[a][k]
                posi = points_input[a]
                posj = points_input[neigh_id_input[a][b]]
                posk = points_input[neigh_id_input[a][k]]
                rjk = compute_dis(posj, posk)
                cos_ijk = compute_cos_ijk(posi, posj, posk)
                r_2 = rij ** 2 + rik ** 2 + rjk ** 2
                value1 += math.exp(-(r_2 / (radius * a_list[0]) ** 2)) * (1 + b_list[0] * cos_ijk) ** c_list[0]
                value2 += math.exp(-(r_2 / (radius * a_list[1]) ** 2)) * (1 + b_list[1] * cos_ijk) ** c_list[1]
                value3 += math.exp(-(r_2 / (radius * a_list[2]) ** 2)) * (1 + b_list[2] * cos_ijk) ** c_list[2]
                value4 += math.exp(-(r_2 / (radius * a_list[3]) ** 2)) * (1 + b_list[3] * cos_ijk) ** c_list[3]
                value5 += math.exp(-(r_2 / (radius * a_list[4]) ** 2)) * (1 + b_list[4] * cos_ijk) ** c_list[4]
                value6 += math.exp(-(r_2 / (radius * a_list[5]) ** 2)) * (1 + b_list[5] * cos_ijk) ** c_list[5]
                value7 += math.exp(-(r_2 / (radius * a_list[6]) ** 2)) * (1 + b_list[6] * cos_ijk) ** c_list[6]
                value8 += math.exp(-(r_2 / (radius * a_list[7]) ** 2)) * (1 + b_list[7] * cos_ijk) ** c_list[7]
                value9 += math.exp(-(r_2 / (radius * a_list[8]) ** 2)) * (1 + b_list[8] * cos_ijk) ** c_list[8]
                value10 += math.exp(-(r_2 / (radius * a_list[9]) ** 2)) * (1 + b_list[9] * cos_ijk) ** c_list[9]
                value11 += math.exp(-(r_2 / (radius * a_list[10]) ** 2)) * (1 + b_list[10] * cos_ijk) ** c_list[10]
                value12 += math.exp(-(r_2 / (radius * a_list[11]) ** 2)) * (1 + b_list[11] * cos_ijk) ** c_list[11]
                value13 += math.exp(-(r_2 / (radius * a_list[12]) ** 2)) * (1 + b_list[12] * cos_ijk) ** c_list[12]
                value14 += math.exp(-(r_2 / (radius * a_list[13]) ** 2)) * (1 + b_list[13] * cos_ijk) ** c_list[13]
                value15 += math.exp(-(r_2 / (radius * a_list[14]) ** 2)) * (1 + b_list[14] * cos_ijk) ** c_list[14]
                value16 += math.exp(-(r_2 / (radius * a_list[15]) ** 2)) * (1 + b_list[15] * cos_ijk) ** c_list[15]
                value17 += math.exp(-(r_2 / (radius * a_list[16]) ** 2)) * (1 + b_list[16] * cos_ijk) ** c_list[16]
                value18 += math.exp(-(r_2 / (radius * a_list[17]) ** 2)) * (1 + b_list[17] * cos_ijk) ** c_list[17]
                value19 += math.exp(-(r_2 / (radius * a_list[18]) ** 2)) * (1 + b_list[18] * cos_ijk) ** c_list[18]
                value20 += math.exp(-(r_2 / (radius * a_list[19]) ** 2)) * (1 + b_list[19] * cos_ijk) ** c_list[19]
                value21 += math.exp(-(r_2 / (radius * a_list[20]) ** 2)) * (1 + b_list[20] * cos_ijk) ** c_list[20]
                value22 += math.exp(-(r_2 / (radius * a_list[21]) ** 2)) * (1 + b_list[21] * cos_ijk) ** c_list[21]
        angular_value_in[0][a] = value1
        angular_value_in[1][a] = value2
        angular_value_in[2][a] = value3
        angular_value_in[3][a] = value4
        angular_value_in[4][a] = value5
        angular_value_in[5][a] = value6
        angular_value_in[6][a] = value7
        angular_value_in[7][a] = value8
        angular_value_in[8][a] = value9
        angular_value_in[9][a] = value10
        angular_value_in[10][a] = value11
        angular_value_in[11][a] = value12
        angular_value_in[12][a] = value13
        angular_value_in[13][a] = value14
        angular_value_in[14][a] = value15
        angular_value_in[15][a] = value16
        angular_value_in[16][a] = value17
        angular_value_in[17][a] = value18
        angular_value_in[18][a] = value19
        angular_value_in[19][a] = value20
        angular_value_in[20][a] = value21
        angular_value_in[21][a] = value22
    return angular_value_in


@jit(nopython=True)
def compute_radial_value(delta_radius_input, distance_length_index_input, distance_input,
                         like_input, delta_r_input):
    radial_value_in = np.empty_like(like_input)
    for a in range(len(delta_radius_input)):
        delta_radius_now = delta_radius_input[a]
        for b in range(len(distance_input)):
            value = 0.0
            for c in range(distance_length_index_input[b]):
                value += math.exp(-0.5 * ((distance_input[b][c] - delta_radius_now) / delta_r_input) ** 2)
            radial_value_in[a][b] = value
    return radial_value_in


def compute_symmetry_functions(points, radius, d50):
    # Compute symmetry function values of the whole granular system.
    # Reference: [1] E. D. Cubuk, R. J. S. Ivancic, S. S. Schoenholz. Structure-property relationships from universal signatures of plasticity in disordered solids. DOI: 10.1126/science.aai8830. Science(2017)
    #            [2] E. D. Cubuk,1, ∗ S. S. Schoenholz. Identifying structural ﬂow defects in disordered solids using machine learning methods.PRL(2015).
    # step1. set the constant
    particle_number = len(points)
    radius_for_input = d50 / 2
    a_list_for_input = np.array([14.638, 14.638, 14.638, 14.638, 2.554, 2.554, 2.554, 2.554, 1.648, 1.648, 1.204, 1.204,
                                 1.204, 1.204, 0.933, 0.933, 0.933, 0.933, 0.695, 0.695, 0.695, 0.695])
    b_list_for_input = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    c_list_for_input = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 4, 16, 1, 2, 4, 16, 1, 2, 4, 16])
    like_for_angular = np.empty(shape=[22, particle_number])
    like_for_radial = np.empty(shape=[50, particle_number])
    single_radius = d50 / 2
    delta_r = 0.1 * single_radius
    delta_radius = np.array([np.linspace(0.1, 5.0, 50)[i] * single_radius for i in range(50)])
    # step2. compute neighbour information by KDTree
    # 2.1 kd tree
    max_distance = 5.0 * d50 / 2
    kd_tree = KDTree(points)
    pairs = list(kd_tree.query_pairs(max_distance))
    # 2.2 distance for every pairs
    dis_use = []
    for x in range(len(pairs)):
        dis = ((points[pairs[x][0]][0] - points[pairs[x][1]][0]) ** 2
               + (points[pairs[x][0]][1] - points[pairs[x][1]][1]) ** 2
               + (points[pairs[x][0]][2] - points[pairs[x][1]][2]) ** 2) ** 0.5
        if dis <= max_distance:
            dis_use.append(dis)
    # 2.3 bonds of every particle
    bonds = []
    for x in range(particle_number):
        bonds.append([])
    for x in range(len(pairs)):
        bonds[pairs[x][0]].append(pairs[x][1])
        bonds[pairs[x][1]].append(pairs[x][0])
    # step3. modify neighbour information for next compute
    # 3.1 number of neighbours of every particle
    neigh_id_length_index = []
    for i in range(len(bonds)):
        neigh_id_length_index.append(len(bonds[i]))
    # 3.2 neighbour id array
    neigh_id = np.zeros(shape=[particle_number, max(neigh_id_length_index)], dtype=int)
    for i in range(len(bonds)):
        for j in range(len(bonds[i])):
            neigh_id[i][j] = bonds[i][j]
    neigh_id_length_index_array = np.array(neigh_id_length_index)
    # 3.3 neighbour distance information
    distance = []
    for x in range(particle_number):
        distance.append([])
    for x in range(len(dis_use)):
        distance[pairs[x][0]].append(dis_use[x])
        distance[pairs[x][1]].append(dis_use[x])
    distance_length_index = []
    for i in range(len(distance)):
        distance_length_index.append(len(distance[i]))
    distance_array = np.zeros(shape=[particle_number, max(distance_length_index)])
    for i in range(len(distance)):
        for j in range(len(distance[i])):
            distance_array[i][j] = distance[i][j]
    distance_length_index_array = np.array(distance_length_index)
    # step4. compute
    # 4.1 angular value
    angular_value = compute_angular_element(neigh_id, distance_array, points, a_list_for_input, b_list_for_input,
                                            c_list_for_input, radius_for_input, like_for_angular,
                                            neigh_id_length_index_array).T
    # 4.2 radial value
    radial_value = compute_radial_value(delta_radius, distance_length_index_array, distance_array, like_for_radial,
                                        delta_r).T
    # 4.3 stack
    symmetry_function_value = np.hstack((angular_value, radial_value))
    return symmetry_function_value


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_interstice_distance(voronoi_neighbour_use_input, distance_input, radius_input):
    interstice_distance_in = []
    for a in range(len(voronoi_neighbour_use_input)):
        interstice_distance_now = []
        for b in range(len(voronoi_neighbour_use_input[a])):
            interstice = (distance_input[a][b] - (radius_input[voronoi_neighbour_use_input[a][b]] + radius_input[a])) / \
                         distance_input[a][b]
            interstice_distance_now.append(interstice)
        interstice_distance_in.append([np.min(interstice_distance_now),
                                       np.max(interstice_distance_now),
                                       np.mean(interstice_distance_now),
                                       np.std(interstice_distance_now)])
    return np.array(interstice_distance_in)


def compute_interstice_area_polysize(voronoi_neighbour_use_input, points_input, radius_input):
    interstice_area_in = []
    for a in range(len(voronoi_neighbour_use_input)):
        if len(voronoi_neighbour_use_input[a]) >= 4:
            points_now = []
            radius_now = []
            for b in range(len(voronoi_neighbour_use_input[a])):
                points_now.append(points_input[voronoi_neighbour_use_input[a][b]])
                radius_now.append(radius_input[voronoi_neighbour_use_input[a][b]])
            ch = ConvexHull(points_now)
            simplice = np.array(ch.simplices)
            interstice_area_mid = np.zeros(shape=[len(simplice), ])
            interstice_area_x = compute_interstice_area_polysize_single_particle(simplice,
                                                                                 np.array(points_now),
                                                                                 np.array(radius_now),
                                                                                 interstice_area_mid)
            interstice_area_in.append([np.min(interstice_area_x),
                                       np.max(interstice_area_x),
                                       np.mean(interstice_area_x),
                                       np.std(interstice_area_x, ddof=1)])
        else:
            interstice_area_in.append([0.0, 0.0, 0.0, 0.0])
    return np.array(interstice_area_in)


@jit(nopython=True)
def compute_interstice_area_polysize_single_particle(simplice, points_now, radius_now, interstice_area_mid):
    interstice_area_x = interstice_area_mid
    for a in range(len(simplice)):
        area_triangle = compute_simplice_area(points_now[simplice[a][0]],
                                              points_now[simplice[a][1]], points_now[simplice[a][2]])
        area_pack = (math.acos(compute_cos_ijk(points_now[simplice[a][0]],
                                               points_now[simplice[a][1]], points_now[simplice[a][2]]))
                     * radius_now[simplice[a][0]] ** 2 +
                     math.acos(compute_cos_ijk(points_now[simplice[a][2]],
                                               points_now[simplice[a][0]], points_now[simplice[a][1]]))
                     * radius_now[simplice[a][2]] ** 2 +
                     math.acos(compute_cos_ijk(points_now[simplice[a][1]],
                                               points_now[simplice[a][2]], points_now[simplice[a][0]]))
                     * radius_now[simplice[a][1]] ** 2) / 2
        interstice_area_x[a] = (area_triangle - area_pack) / area_triangle
    return interstice_area_x


def compute_interstice_area_monosize(voronoi_neighbour_use_input, points_input, radius_input):
    interstice_area_in = []
    for a in range(len(voronoi_neighbour_use_input)):
        if len(voronoi_neighbour_use_input[a]) >= 4:
            points_now = []
            for b in range(len(voronoi_neighbour_use_input[a])):
                points_now.append(points_input[voronoi_neighbour_use_input[a][b]])
            ch = ConvexHull(points_now)
            simplice = np.array(ch.simplices)
            interstice_area_mid = np.zeros(shape=[len(simplice), ])
            interstice_area_x = compute_interstice_area_monosize_single_particle(simplice,
                                                                                 np.array(points_now),
                                                                                 radius_input,
                                                                                 interstice_area_mid)
            interstice_area_in.append([np.min(interstice_area_x),
                                       np.max(interstice_area_x),
                                       np.mean(interstice_area_x),
                                       np.std(interstice_area_x, ddof=1)])
        else:
            interstice_area_in.append([0.0, 0.0, 0.0, 0.0])
    return np.array(interstice_area_in)


@jit(nopython=True)
def compute_interstice_area_monosize_single_particle(simplice, points_now, radius_input, interstice_area_mid):
    interstice_area_x = interstice_area_mid
    for a in range(len(simplice)):
        area_triangle = compute_simplice_area(points_now[simplice[a][0]],
                                              points_now[simplice[a][1]], points_now[simplice[a][2]])
        area_pack = (math.pi * radius_input ** 2) / 4
        interstice_area_x[a] = (area_triangle - area_pack) / area_triangle
    return interstice_area_x


def compute_interstice_volume(voronoi_neighbour_use_input, points_input, radius_input):
    interstice_volume_in = []
    for a in range(len(voronoi_neighbour_use_input)):
        if len(voronoi_neighbour_use_input[a]) >= 4:
            points_now = []
            radius_now = []
            origin_particle = points_input[a]
            origin_radius = radius_input[a]
            for b in range(len(voronoi_neighbour_use_input[a])):
                points_now.append(points_input[voronoi_neighbour_use_input[a][b]])
                radius_now.append(radius_input[voronoi_neighbour_use_input[a][b]])
            ch = ConvexHull(points_now)
            simplice = np.array(ch.simplices)
            interstice_volume_mid = np.zeros(shape=[len(simplice), ])
            interstice_volume_x = compute_interstice_volume_single_particle(simplice,
                                                                            np.array(points_now),
                                                                            np.array(radius_now),
                                                                            interstice_volume_mid,
                                                                            origin_particle,
                                                                            origin_radius)

            interstice_volume_in.append([np.min(interstice_volume_x),
                                         np.max(interstice_volume_x),
                                         np.mean(interstice_volume_x),
                                         np.std(interstice_volume_x, ddof=1)])
        else:
            interstice_volume_in.append([0.0, 0.0, 0.0, 0.0])
    return np.array(interstice_volume_in)


@jit(nopython=True)
def compute_interstice_volume_single_particle(simplice, points_now, radius_now, interstice_volume_mid,
                                              origin_particle, origin_radius):
    interstice_volume_x = interstice_volume_mid
    for a in range(len(simplice)):
        volume_triangle = compute_tetrahedron_volume(points_now[simplice[a][0]],
                                                     points_now[simplice[a][1]], points_now[simplice[a][2]],
                                                     origin_particle)
        if volume_triangle == 0:
            # 处于边界上的颗粒计算时会触发错误，在此随机赋予一个值，因为边界上的颗粒不参与到以后的计算中
            interstice_volume_x[a] = 0
        else:
            volume_pack = (compute_solide_angle(origin_particle, points_now[simplice[a][0]], points_now[simplice[a][1]],
                                                points_now[simplice[a][2]])
                           * origin_radius ** 3 +
                           compute_solide_angle(points_now[simplice[a][2]], origin_particle,
                                                points_now[simplice[a][0]], points_now[simplice[a][1]])
                           * radius_now[simplice[a][2]] ** 3 +
                           compute_solide_angle(points_now[simplice[a][1]],
                                                points_now[simplice[a][2]], origin_particle, points_now[simplice[a][0]])
                           * radius_now[simplice[a][1]] ** 3 +
                           compute_solide_angle(points_now[simplice[a][0]], points_now[simplice[a][1]],
                                                points_now[simplice[a][2]], origin_particle) * radius_now[
                               simplice[a][0]] ** 3) \
                          / 3
            interstice_volume_x[a] = (volume_triangle - volume_pack) / volume_triangle
    return interstice_volume_x


@jit(nopython=True)
def interstice_distribution_MRO(interstice_distance, interstice_area, interstice_volume,
                                MRO_array, f_use_array, voronoi_neighbour, neigh_id_length_index):
    feature_MRO = np.empty_like(MRO_array)
    for aa in range(4):
        a = 5 * aa
        feature_now = interstice_distance[:, aa]
        for b in range(len(voronoi_neighbour)):
            f_use_not = np.zeros_like(f_use_array)
            for c in range(neigh_id_length_index[b]):
                f_use_not[c] = feature_now[voronoi_neighbour[b][c]]
            f_use = f_use_not[0: neigh_id_length_index[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(4):
        a = 5 * aa + 20
        feature_now = interstice_area[:, aa]
        for b in range(len(voronoi_neighbour)):
            f_use_not = np.zeros_like(f_use_array)
            for c in range(neigh_id_length_index[b]):
                f_use_not[c] = feature_now[voronoi_neighbour[b][c]]
            f_use = f_use_not[0: neigh_id_length_index[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(4):
        a = 5 * aa + 40
        feature_now = interstice_volume[:, aa]
        for b in range(len(voronoi_neighbour)):
            f_use_not = np.zeros_like(f_use_array)
            for c in range(neigh_id_length_index[b]):
                f_use_not[c] = feature_now[voronoi_neighbour[b][c]]
            f_use = f_use_not[0: neigh_id_length_index[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    return feature_MRO.T


def compute_interstice_distribution(neighbour, points, radius):
    # compute interstice distribution of the whole granular system, include SRO(short range order), MRO(medium range order)
    # Reference: [1] Qi Wang1* & Anubhav Jain. A transferable machine-learning framework linking interstice distribution and plastic heterogeneity in metallic glasses. https://doi.org/10.1038/s41467-019-13511-9. NATURE COMMUNICATIONS.(2019)
    # step1. set constant
    particle_number = len(neighbour)
    # step2. modify origin voronoi neighbour
    voronoi_neighbour = []
    for x in range(len(neighbour)):
        voronoi_neighbour_now = []
        for value in neighbour[x]:
            if value >= 0:
                voronoi_neighbour_now.append(value)
        voronoi_neighbour.append(voronoi_neighbour_now)
    bonds = []
    for x in range(len(voronoi_neighbour)):
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds.append([x, voronoi_neighbour[x][y]])
    bonds = np.array(bonds)
    voronoi_neighbour_use = []
    for x in range(particle_number):
        voronoi_neighbour_use.append([])
    for x in range(len(bonds)):
        voronoi_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voronoi_neighbour_use[bonds[x][1]].append(bonds[x][0])
    neigh_id_length_index = []
    for x in range(len(voronoi_neighbour_use)):
        neigh_id_length_index.append(len(voronoi_neighbour_use[x]))
    neigh_id = np.zeros(shape=[particle_number, max(neigh_id_length_index)], dtype=int)
    for x in range(len(voronoi_neighbour_use)):
        for y in range(len(voronoi_neighbour_use[x])):
            neigh_id[x][y] = int(voronoi_neighbour_use[x][y])
    neigh_id_length_index = np.array(neigh_id_length_index)
    # step3. distance for every pairs
    dis_use = []
    for x in range(len(bonds)):
        dis = ((points[bonds[x][0]][0] - points[bonds[x][1]][0]) ** 2
               + (points[bonds[x][0]][1] - points[bonds[x][1]][1]) ** 2
               + (points[bonds[x][0]][2] - points[bonds[x][1]][2]) ** 2) ** 0.5
        dis_use.append(dis)
    # step4. neighbour distance of every particle
    distance = []
    for x in range(particle_number):
        distance.append([])
    for x in range(len(bonds)):
        distance[bonds[x][0]].append(dis_use[x])
        distance[bonds[x][1]].append(dis_use[x])
    # step5. compute
    # 5.1 compute interstice distance
    interstice_distance = compute_interstice_distance(voronoi_neighbour_use, distance, radius)
    # 5.2 compute interstice area
    # interstice_area = compute_interstice_area_monosize(voronoi_neighbour_use, points, radius[0])
    interstice_area = compute_interstice_area_polysize(voronoi_neighbour_use, points, radius)
    # 5.3 compute interstice volume
    interstice_volume = compute_interstice_volume(voronoi_neighbour_use, points, radius)
    # 5.4 MRO, compute the medium range order feature of interstice_distance, interstice_area and interstice_volume
    MRO_array = np.empty(shape=[60, particle_number])
    f_use_array = np.empty(shape=[particle_number, ])
    MRO_interstice_distribution = interstice_distribution_MRO(interstice_distance, interstice_area, interstice_volume,
                                                              MRO_array, f_use_array, neigh_id, neigh_id_length_index)
    return MRO_interstice_distribution


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_coordination_number_cutoff_distance(points_input, radius_input):
    maxdistance = 3.0 * radius_input[0]
    kdtree = KDTree(points_input)
    pairs = list(kdtree.query_pairs(maxdistance))
    cutoff_bonds = []
    for a in range(len(points_input)):
        cutoff_bonds.append([])
    for a in range(len(pairs)):
        cutoff_bonds[pairs[a][0]].append(pairs[a][1])
        cutoff_bonds[pairs[a][1]].append(pairs[a][0])
    coordination_number_by_cutoff_distance_in = np.zeros(shape=[len(points_input), ])
    for a in range(len(cutoff_bonds)):
        coordination_number_by_cutoff_distance_in[a] = len(cutoff_bonds[a])
    return coordination_number_by_cutoff_distance_in


def compute_coordination_number_cutoff_distance_polysize(points_input, cutoff_distance):
    maxdistance = cutoff_distance
    kdtree = KDTree(points_input)
    pairs = list(kdtree.query_pairs(maxdistance))
    cutoff_bonds = []
    for a in range(len(points_input)):
        cutoff_bonds.append([])
    for a in range(len(pairs)):
        cutoff_bonds[pairs[a][0]].append(pairs[a][1])
        cutoff_bonds[pairs[a][1]].append(pairs[a][0])
    coordination_number_by_cutoff_distance_in = np.zeros(shape=[len(points_input), ])
    for a in range(len(cutoff_bonds)):
        coordination_number_by_cutoff_distance_in[a] = len(cutoff_bonds[a])
    return coordination_number_by_cutoff_distance_in


def compute_cellfraction(voro_input, radius_input):
    volume = []
    for a in range(len(voro_input)):
        volume.append(voro_input[a]['volume'])
    ball_volume = (np.max(radius_input) ** 3) * 4 * math.pi / 3
    cellfraction_in = [ball_volume / volume[a] for a in range(len(voro_input))]
    return cellfraction_in


def compute_cellfraction_polysize(voro_input, radius_input):
    volume = []
    for a in range(len(voro_input)):
        volume.append(voro_input[a]['volume'])
    ball_volume = [(radius_input[a] ** 3) * 4 * math.pi / 3 for a in range(len(radius_input))]
    cellfraction_in = [ball_volume[a] / volume[a] for a in range(len(voro_input))]
    return cellfraction_in


def compute_weight_i_fold_symm(voro_input, area_all_input):
    # Reference: [1] Qi Wang1* & Anubhav Jain. A transferable machine-learning framework linking interstice distribution and plastic heterogeneity in metallic glasses. https://doi.org/10.1038/s41467-019-13511-9. NATURE COMMUNICATIONS.(2019)
    particle_number = len(voro_input)
    area_weight_i_fold_symm = {}
    area_weight_i_fold_symm3_in = np.zeros(shape=[particle_number, ])
    area_weight_i_fold_symm4_in = np.zeros(shape=[particle_number, ])
    area_weight_i_fold_symm5_in = np.zeros(shape=[particle_number, ])
    area_weight_i_fold_symm6_in = np.zeros(shape=[particle_number, ])
    area_weight_i_fold_symm7_in = np.zeros(shape=[particle_number, ])
    for a in range(len(voro_input)):
        faces_in = voro_input[a]['faces']
        vertices_id_length = []
        for b in range(len(faces_in)):
            vertices_id_length.append(len(faces_in[b]['vertices']))
        id3 = [c for c, b in enumerate(vertices_id_length) if b == 3]
        id4 = [c for c, b in enumerate(vertices_id_length) if b == 4]
        id5 = [c for c, b in enumerate(vertices_id_length) if b == 5]
        id6 = [c for c, b in enumerate(vertices_id_length) if b == 6]
        id7 = [c for c, b in enumerate(vertices_id_length) if b == 7]
        area3, area4, area5, area6, area7 = 0, 0, 0, 0, 0
        for b in range(len(id3)):
            area3 += area_all_input[a][id3[b]]
        for b in range(len(id4)):
            area4 += area_all_input[a][id4[b]]
        for b in range(len(id5)):
            area5 += area_all_input[a][id5[b]]
        for b in range(len(id6)):
            area6 += area_all_input[a][id6[b]]
        for b in range(len(id7)):
            area7 += area_all_input[a][id7[b]]
        area_total = area3 + area4 + area5 + area6 + area7
        area_weight_i_fold_symm3_in[a] = area3 / area_total
        area_weight_i_fold_symm4_in[a] = area4 / area_total
        area_weight_i_fold_symm5_in[a] = area5 / area_total
        area_weight_i_fold_symm6_in[a] = area6 / area_total
        area_weight_i_fold_symm7_in[a] = area7 / area_total
    area_weight_i_fold_symm['3'] = area_weight_i_fold_symm3_in
    area_weight_i_fold_symm['4'] = area_weight_i_fold_symm4_in
    area_weight_i_fold_symm['5'] = area_weight_i_fold_symm5_in
    area_weight_i_fold_symm['6'] = area_weight_i_fold_symm6_in
    area_weight_i_fold_symm['7'] = area_weight_i_fold_symm7_in
    return area_weight_i_fold_symm


def compute_voronoi_idx(voro_input):
    # Reference: [1] Qi Wang1* & Anubhav Jain. A transferable machine-learning framework linking interstice distribution and plastic heterogeneity in metallic glasses. https://doi.org/10.1038/s41467-019-13511-9. NATURE COMMUNICATIONS.(2019)
    #            [2] H.L.Peng M.Z.Li* and W.H.Wang. Structural signature of plastic deformation in metallic glasses.
    particle_number = len(voro_input)
    voronoi_idx = {}
    i_fold_symm = {}
    voronoi_idx3_in = np.zeros(shape=[particle_number, ])
    voronoi_idx4_in = np.zeros(shape=[particle_number, ])
    voronoi_idx5_in = np.zeros(shape=[particle_number, ])
    voronoi_idx6_in = np.zeros(shape=[particle_number, ])
    voronoi_idx7_in = np.zeros(shape=[particle_number, ])
    i_fold_symm3_in = np.zeros(shape=[particle_number, ])
    i_fold_symm4_in = np.zeros(shape=[particle_number, ])
    i_fold_symm5_in = np.zeros(shape=[particle_number, ])
    i_fold_symm6_in = np.zeros(shape=[particle_number, ])
    i_fold_symm7_in = np.zeros(shape=[particle_number, ])
    for a in range(len(voro_input)):
        faces_in = voro_input[a]['faces']
        vertices_id_length = []
        for b in range(len(faces_in)):
            vertices_id_length.append(len(faces_in[b]['vertices']))
        count_all = vertices_id_length.count(3) + vertices_id_length.count(4) \
                    + vertices_id_length.count(5) + vertices_id_length.count(6) \
                    + vertices_id_length.count(7)
        voronoi_idx3_in[a] = vertices_id_length.count(3)
        voronoi_idx4_in[a] = vertices_id_length.count(4)
        voronoi_idx5_in[a] = vertices_id_length.count(5)
        voronoi_idx6_in[a] = vertices_id_length.count(6)
        voronoi_idx7_in[a] = vertices_id_length.count(7)
        i_fold_symm3_in[a] = vertices_id_length.count(3) / count_all
        i_fold_symm4_in[a] = vertices_id_length.count(4) / count_all
        i_fold_symm5_in[a] = vertices_id_length.count(5) / count_all
        i_fold_symm6_in[a] = vertices_id_length.count(6) / count_all
        i_fold_symm7_in[a] = vertices_id_length.count(7) / count_all
    voronoi_idx['3'] = voronoi_idx3_in
    voronoi_idx['4'] = voronoi_idx4_in
    voronoi_idx['5'] = voronoi_idx5_in
    voronoi_idx['6'] = voronoi_idx6_in
    voronoi_idx['7'] = voronoi_idx7_in
    i_fold_symm['3'] = i_fold_symm3_in
    i_fold_symm['4'] = i_fold_symm4_in
    i_fold_symm['5'] = i_fold_symm5_in
    i_fold_symm['6'] = i_fold_symm6_in
    i_fold_symm['7'] = i_fold_symm7_in
    return voronoi_idx, i_fold_symm


def compute_boop(voronoi_neighbour, points, cutoff_distance):
    # compute boo based on voronoi neighbour and cutoff neighbour
    # Reference: [1] Qi Wang1* & Anubhav Jain. A transferable machine-learning framework linking interstice distribution and plastic heterogeneity in metallic glasses. https://doi.org/10.1038/s41467-019-13511-9. NATURE COMMUNICATIONS.(2019)
    #            [2] https://pyboo.readthedocs.io/en/latest/intro.html
    # step1. compute boo based on voronoi neighbour
    bonds1 = []
    for x in range(len(voronoi_neighbour)):
        # 使用这种方法剔除了邻域不互相对称的颗粒，由剔除面积小于平均面积百分之五的邻域点所造成的不对称
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds1.append([x, voronoi_neighbour[x][y]])
    bonds1 = np.array(bonds1)
    inside1 = np.array([True] * len(points))

    q2m_1 = boo.bonds2qlm(points, bonds1, l=2)
    q4m_1 = boo.bonds2qlm(points, bonds1, l=4)
    q6m_1 = boo.bonds2qlm(points, bonds1, l=6)
    q8m_1 = boo.bonds2qlm(points, bonds1, l=8)
    q10m_1 = boo.bonds2qlm(points, bonds1, l=10)

    Q2m_1, inside1_2_2 = boo.coarsegrain_qlm(q2m_1, bonds1, inside1)
    Q4m_1, inside1_2_4 = boo.coarsegrain_qlm(q4m_1, bonds1, inside1)
    Q6m_1, inside1_2_6 = boo.coarsegrain_qlm(q6m_1, bonds1, inside1)
    Q8m_1, inside1_2_8 = boo.coarsegrain_qlm(q8m_1, bonds1, inside1)
    Q10m_1, inside1_2_10 = boo.coarsegrain_qlm(q10m_1, bonds1, inside1)

    q2_1 = boo.ql(q2m_1)
    q4_1 = boo.ql(q4m_1)
    q6_1 = boo.ql(q6m_1)
    q8_1 = boo.ql(q8m_1)
    q10_1 = boo.ql(q10m_1)

    w2_1 = boo.wl(q2m_1)
    w4_1 = boo.wl(q4m_1)
    w6_1 = boo.wl(q6m_1)
    w8_1 = boo.wl(q8m_1)
    w10_1 = boo.wl(q10m_1)

    Q2_1 = boo.ql(Q2m_1)
    Q4_1 = boo.ql(Q4m_1)
    Q6_1 = boo.ql(Q6m_1)
    Q8_1 = boo.ql(Q8m_1)
    Q10_1 = boo.ql(Q10m_1)

    W2_1 = boo.wl(Q2m_1)
    W4_1 = boo.wl(Q4m_1)
    W6_1 = boo.wl(Q6m_1)
    W8_1 = boo.wl(Q8m_1)
    W10_1 = boo.wl(Q10m_1)
    # step2. compute boo based on cutoff neighbour
    max_distance = cutoff_distance
    kdtree = KDTree(points)
    bonds2 = np.array(list(kdtree.query_pairs(max_distance)))
    inside2 = np.array([True] * len(points))

    q2m_2 = boo.bonds2qlm(points, bonds2, l=2)
    q4m_2 = boo.bonds2qlm(points, bonds2, l=4)
    q6m_2 = boo.bonds2qlm(points, bonds2, l=6)
    q8m_2 = boo.bonds2qlm(points, bonds2, l=8)
    q10m_2 = boo.bonds2qlm(points, bonds2, l=10)

    Q2m_2, inside2_2_2 = boo.coarsegrain_qlm(q2m_2, bonds2, inside2)
    Q4m_2, inside2_2_4 = boo.coarsegrain_qlm(q4m_2, bonds2, inside2)
    Q6m_2, inside2_2_6 = boo.coarsegrain_qlm(q6m_2, bonds2, inside2)
    Q8m_2, inside2_2_8 = boo.coarsegrain_qlm(q8m_2, bonds2, inside2)
    Q10m_2, inside2_2_10 = boo.coarsegrain_qlm(q10m_2, bonds2, inside2)

    q2_2 = boo.ql(q2m_2)
    q4_2 = boo.ql(q4m_2)
    q6_2 = boo.ql(q6m_2)
    q8_2 = boo.ql(q8m_2)
    q10_2 = boo.ql(q10m_2)

    w2_2 = boo.wl(q2m_2)
    w4_2 = boo.wl(q4m_2)
    w6_2 = boo.wl(q6m_2)
    w8_2 = boo.wl(q8m_2)
    w10_2 = boo.wl(q10m_2)

    Q2_2 = boo.ql(Q2m_2)
    Q4_2 = boo.ql(Q4m_2)
    Q6_2 = boo.ql(Q6m_2)
    Q8_2 = boo.ql(Q8m_2)
    Q10_2 = boo.ql(Q10m_2)

    W2_2 = boo.wl(Q2m_2)
    W4_2 = boo.wl(Q4m_2)
    W6_2 = boo.wl(Q6m_2)
    W8_2 = boo.wl(Q8m_2)
    W10_2 = boo.wl(Q10m_2)

    boop_all = np.array(list(zip(q2_1, q4_1, q6_1, q8_1, q10_1,
                                 w2_1, w4_1, w6_1, w8_1, w10_1,
                                 q2_2, q4_2, q6_2, q8_2, q10_2,
                                 w2_2, w4_2, w6_2, w8_2, w10_2,
                                 Q2_1, Q4_1, Q6_1, Q8_1, Q10_1,
                                 W2_1, W4_1, W6_1, W8_1, W10_1,
                                 Q2_2, Q4_2, Q6_2, Q8_2, Q10_2,
                                 W2_2, W4_2, W6_2, W8_2, W10_2)))

    return boop_all


def compute_modified_boop(voronoi, points, area_all):
    # Bond orientation order parameter(BOO) and modified BOO
    #
    # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
    #
    # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
    # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
    # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
    # reference:
    # [1] https://pyboo.readthedocs.io/en/latest/intro.html
    # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
    # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
    # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974
    modified_boop = {}
    l_index = [4, 6, 8]
    qlm_4 = np.zeros(shape=[len(points), 5]).astype(complex)
    qlm_6 = np.zeros(shape=[len(points), 7]).astype(complex)
    qlm_8 = np.zeros(shape=[len(points), 9]).astype(complex)
    ql = np.zeros(shape=[len(points), 3])
    for i in range(len(points)):
        area_now = area_all[i]
        faces = voronoi[i]['faces']
        adjacent_cell = []
        for y in range(len(faces)):
            adjacent_cell.append(faces[y]['adjacent_cell'])
        pt_coord = np.zeros(shape=[len(adjacent_cell), 3])
        for j in range(len(adjacent_cell)):
            if adjacent_cell[j] >= 0:
                pt_coord[j][0] = points[adjacent_cell[j]][0]
                pt_coord[j][1] = points[adjacent_cell[j]][1]
                pt_coord[j][2] = points[adjacent_cell[j]][2]
            else:
                pt_coord[j][0] = 0.0
                pt_coord[j][1] = 0.0
                pt_coord[j][2] = 0.0
        sum_area = 0.0
        for j in range(len(adjacent_cell)):
            if adjacent_cell[j] >= 0:
                sum_area += area_now[j]
        area_weight = np.zeros(shape=[len(adjacent_cell), ])
        for j in range(len(adjacent_cell)):
            if adjacent_cell[j] >= 0:
                area_weight[j] = area_now[j] / sum_area
            if adjacent_cell[j] < 0:
                area_weight[j] = 0.0
        for j in range(len(l_index)):
            ql[i][j], qlm = boo_ql(points[i], pt_coord, area_weight, l=l_index[j], modified=True)
            if l_index[j] == 4:
                for k in range(5):
                    qlm_4[i][k] = qlm[k + 4]
            if l_index[j] == 6:
                for k in range(7):
                    qlm_6[i][k] = qlm[k + 6]
            if l_index[j] == 8:
                for k in range(9):
                    qlm_8[i][k] = qlm[k + 8]
    w4 = boo.wl(qlm_4)
    w6 = boo.wl(qlm_6)
    w8 = boo.wl(qlm_8)
    # 对w4.w6.w8值进行缩放
    # reference：Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
    scale_4 = (9 / (4 * math.pi)) ** 1.5
    scale_6 = (13 / (4 * math.pi)) ** 1.5
    scale_8 = (17 / (4 * math.pi)) ** 1.5
    w4_dot = [w4[x] / (ql[x][0] ** 3 * scale_4) for x in range(len(w4))]
    w6_dot = [w6[x] / (ql[x][1] ** 3 * scale_6) for x in range(len(w6))]
    w8_dot = [w6[x] / (ql[x][2] ** 3 * scale_8) for x in range(len(w8))]
    wl = {}
    wl['w4'] = np.array(w4_dot)
    wl['w6'] = np.array(w6_dot)
    wl['w8'] = np.array(w8_dot)
    modified_boop['ql'] = ql
    modified_boop['wl'] = wl
    return modified_boop


def boo_ql(origin, pt_coord, area_weight, l, modified):
    # l: order of symmetry
    # origin: center of the cluster
    # pt_coord:
    m_list = np.arange(-l, l + 1, 1)
    pt_num = len(pt_coord)
    qlm = np.zeros(len(m_list)).astype(complex)
    bond_weight = np.zeros(pt_num)
    polar_angle = np.zeros(pt_num)
    azimuth_angle = np.zeros(pt_num)
    for j in range(pt_num):
        bond_weight[j] = area_weight[j]
        bond = unit_vector((pt_coord[j] - origin))
        polar_angle[j] = calc_beta_rad(bond)
        azimuth_angle[j] = calc_gamma_rad(bond)
    sum_weight = np.sum(bond_weight)
    if modified:
        for i, m in enumerate(m_list):
            for j in range(pt_num):
                qlm[i] += special.sph_harm(m, l, azimuth_angle[j], polar_angle[j]) * bond_weight[j] / sum_weight
    else:
        for i, m in enumerate(m_list):
            for j in range(pt_num):
                qlm[i] += special.sph_harm(m, l, azimuth_angle[j], polar_angle[j])
            qlm[i] /= pt_num
    ql = np.sum(np.abs(qlm) ** 2.0)
    ql = np.sqrt(4 * np.pi / (2 * l + 1) * ql)
    return ql, qlm


def compute_cluster_packing_efficiency(voronoi_neighbour_use_input, points_input, radius_input):
    # compute cluster packing efficiency
    # Reference: Yang, L. et al. Atomic-scale mechanisms of the glass-forming ability in metallic glasses. Phys. Rev. Lett. 109, 105502 (2012).
    cluster_packing_efficiency = np.zeros(shape=[len(voronoi_neighbour_use_input), ])
    for a in range(len(voronoi_neighbour_use_input)):
        if len(voronoi_neighbour_use_input[a]) >= 4:
            points_now = []
            radius_now = []
            origin_particle = points_input[a]
            origin_radius = radius_input[a]
            for b in range(len(voronoi_neighbour_use_input[a])):
                points_now.append(points_input[voronoi_neighbour_use_input[a][b]])
                radius_now.append(radius_input[voronoi_neighbour_use_input[a][b]])
            cpe_ch = ConvexHull(points_now)
            cpe_simplice = np.array(cpe_ch.simplices)
            interstice_volume_mid = np.zeros(shape=[len(cpe_simplice), ])
            cluster_packing_efficiency_x = compute_cluster_packing_efficiency_single_particle(cpe_simplice,
                                                                                              np.array(points_now),
                                                                                              np.array(radius_now),
                                                                                              interstice_volume_mid,
                                                                                              origin_particle,
                                                                                              origin_radius)

            cluster_packing_efficiency[a] = cluster_packing_efficiency_x
        else:
            cluster_packing_efficiency[a] = 0.0
    return cluster_packing_efficiency


@jit(nopython=True)
def compute_cluster_packing_efficiency_single_particle(simplice_input, points_now, radius_now, interstice_volume_mid,
                                                       origin_particle, origin_radius):
    triangle_volume_x = np.zeros_like(interstice_volume_mid)
    pack_volume_x = np.zeros_like(interstice_volume_mid)
    for b in range(len(simplice_input)):
        volume_triangle = compute_tetrahedron_volume(points_now[simplice_input[b][0]],
                                                     points_now[simplice_input[b][1]],
                                                     points_now[simplice_input[b][2]],
                                                     origin_particle)
        volume_pack = (compute_solide_angle(origin_particle, points_now[simplice_input[b][0]],
                                            points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]])
                       * origin_radius ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][2]], origin_particle,
                                            points_now[simplice_input[b][0]], points_now[simplice_input[b][1]])
                       * radius_now[simplice_input[b][2]] ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]], origin_particle,
                                            points_now[simplice_input[b][0]])
                       * radius_now[simplice_input[b][1]] ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][0]], points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]], origin_particle) * radius_now[
                           simplice_input[b][0]] ** 3) / 3
        triangle_volume_x[b] = volume_triangle
        pack_volume_x[b] = volume_pack
    return np.sum(pack_volume_x) / np.sum(triangle_volume_x)


@jit(nopython=True)
def minkowski_tensor_w1_02(area, face_normal_vector, w1_02_array_input):
    # compute anisotropic of the voronoi by Minkowski tensor W1_02.
    # Reference: G. E. Schr¨oder-Turk1(a),W.Mickel1,M.Schr¨oter2. Disordered spherical bead packs are anisotropic. doi: 10.1209/0295-5075/90/34001 May.
    w1_02_array = w1_02_array_input
    for a in range(len(area)):
        w1_02_array += np.outer(face_normal_vector[a], face_normal_vector[a]) * area[a]
    return w1_02_array


def compute_anisotropy_w1_02(area_all, face_normal_vector):
    anisotropic_coefficient = np.zeros(shape=[len(area_all), ])
    for x in range(len(area_all)):
        w1_02_array = np.zeros(shape=[3, 3])
        W1_02 = minkowski_tensor_w1_02(area_all[x], face_normal_vector[x], w1_02_array)
        eig = np.linalg.eig(np.mat(W1_02))[0]
        anisotropic_coefficient[x] = np.min(eig) / np.max(eig)
    return anisotropic_coefficient


@jit(nopython=True)
def MRO(old_feature_SRO_array_input, boop_SRO_array_input, cpe_SRO_array_input,
        MRO_array_input,
        f_use_array_input, voronoi_neighbour_input, neigh_id_length_index_input):
    # medium range order
    feature_MRO = np.empty_like(MRO_array_input)
    for aa in range(18):
        a = 5 * aa
        feature_now = old_feature_SRO_array_input[:, aa]
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(1):
        a = (18 + aa) * 5
        feature_now = cpe_SRO_array_input
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(20):
        a = (19 + aa) * 5
        feature_now = boop_SRO_array_input[:, aa]
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(20):
        a = aa + 195
        feature_now = boop_SRO_array_input[:, aa + 20]
        for b in range(len(voronoi_neighbour_input)):
            feature_MRO[a][b] = feature_now[b]
    '''
    for aa in range(10):
        a = aa * 5 + 215
        feature_now = modified_boop_SRO_array[:, aa]
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    '''
    return feature_MRO


def zip_feature(Coordination_number_by_Voronoi_tessellation, Coordination_number_by_cutoff_distance,
                Voronoi_idx, cellfraction, i_fold_symm, area_weight_i_fold_symm):
    feature_all = np.array(list(zip(Coordination_number_by_Voronoi_tessellation,
                                    Coordination_number_by_cutoff_distance,
                                    Voronoi_idx['3'], Voronoi_idx['4'], Voronoi_idx['5'], Voronoi_idx['6'],
                                    Voronoi_idx['7'],
                                    cellfraction,
                                    i_fold_symm['3'], i_fold_symm['4'], i_fold_symm['5'], i_fold_symm['6'],
                                    i_fold_symm['7'],
                                    area_weight_i_fold_symm['3'],
                                    area_weight_i_fold_symm['4'],
                                    area_weight_i_fold_symm['5'],
                                    area_weight_i_fold_symm['6'],
                                    area_weight_i_fold_symm['7'])))
    return feature_all


def select_important_SRO_features(Coordination_number_by_Voronoi_tessellation, cellfraction,
                                  modified_boop, Cpe, anisotropic):
    feature_all = np.array(list(zip(Coordination_number_by_Voronoi_tessellation,
                                    cellfraction,
                                    modified_boop['ql'][:, 0],
                                    modified_boop['ql'][:, 1],
                                    modified_boop['ql'][:, 2],
                                    modified_boop['wl']['w4'],
                                    modified_boop['wl']['w6'],
                                    modified_boop['wl']['w8'],
                                    Cpe,
                                    anisotropic)))
    return feature_all


def compute_conventional_feature(points, area_all, face_normal_vector, neighbour, voronoi, radius, MRO_option,
                                 cutoff_distance):
    # step1. set constant
    particle_number = len(points)
    MRO_array = np.empty(shape=[215, particle_number])
    f_use_array = np.empty(shape=[particle_number, ])
    # step1. modify voronoi neighbour information
    voronoi_neighbour = []
    for x in range(len(neighbour)):
        voronoi_neighbour_now = []
        for value in neighbour[x]:
            if value >= 0:
                voronoi_neighbour_now.append(value)
        voronoi_neighbour.append(voronoi_neighbour_now)
    bonds = []
    for x in range(len(voronoi_neighbour)):
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds.append([x, voronoi_neighbour[x][y]])
    bonds = np.array(bonds)
    voronoi_neighbour_use = []
    for x in range(len(neighbour)):
        voronoi_neighbour_use.append([])
    for x in range(len(bonds)):
        voronoi_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voronoi_neighbour_use[bonds[x][1]].append(bonds[x][0])
    neigh_id_length_index = []
    for x in range(len(voronoi_neighbour_use)):
        neigh_id_length_index.append(len(voronoi_neighbour_use[x]))
    neigh_id = np.zeros(shape=[particle_number, max(neigh_id_length_index)], dtype=int)
    for x in range(len(voronoi_neighbour_use)):
        for y in range(len(voronoi_neighbour_use[x])):
            neigh_id[x][y] = int(voronoi_neighbour_use[x][y])
    neigh_id_length_index = np.array(neigh_id_length_index)
    # step2. compute
    if MRO_option:
        # 2.1 coordination number by voronoi tessellation
        coordination_number_voronoi_tessellation = np.zeros(shape=[len(points), ])
        for x in range(len(voronoi_neighbour)):
            coordination_number_voronoi_tessellation[x] = len(voronoi_neighbour[x])
        # 2.2 weighted i-fold symm
        area_weight_i_fold_symm = compute_weight_i_fold_symm(voronoi, area_all)
        # 2.3 coordination number by cutoff distance
        coordination_number_cutoff_distance = compute_coordination_number_cutoff_distance_polysize(points, cutoff_distance)
        # 2.4 cell fraction
        cellfraction = compute_cellfraction_polysize(voronoi, radius)
        # 2.5 voronoi index and i-fold symm
        Voronoi_idx, i_fold_symm = compute_voronoi_idx(voronoi)
        # 2.6 zip feature above
        feature_all = zip_feature(coordination_number_voronoi_tessellation, coordination_number_cutoff_distance,
                                  Voronoi_idx, cellfraction, i_fold_symm, area_weight_i_fold_symm)
        # 2.7.1 boop
        boop_all = compute_boop(voronoi_neighbour, points, cutoff_distance)
        # 2.7.2 modified boop
        # modified_boop = compute_modified_boop(voronoi, points, area_all)
        # 2.8 cluster packing efficiency
        cpe = compute_cluster_packing_efficiency(voronoi_neighbour, points, radius)
        # 2.9 anisotropic of voronoi cell by Minkowski tensor W1_02
        # anisotropic = anisotropic_by_W1_02(area_all, face_normal_vector)
        # 2.10 select MRO
        old_feature_SRO_array = feature_all
        boop_SRO_array = boop_all
        # modified_boop_SRO_array = modified_boop
        cpe_SRO_array = cpe
        feature_out = MRO(old_feature_SRO_array, boop_SRO_array, cpe_SRO_array, MRO_array, f_use_array, neigh_id,
                          neigh_id_length_index).T
    else:
        # 2.1 coordination number by voronoi tessellation
        coordination_number_voronoi_tessellation = np.zeros(shape=[len(points), ])
        for x in range(len(voronoi_neighbour)):
            coordination_number_voronoi_tessellation[x] = len(voronoi_neighbour[x])
        # 2.2 cell fraction
        cellfraction = compute_cellfraction_polysize(voronoi, radius)
        # 2.3 modified boop
        modified_boop = compute_modified_boop(voronoi, points, area_all)
        # 2.4 cluster packing efficiency
        cpe = compute_cluster_packing_efficiency(voronoi_neighbour, points, radius)
        # 2.5 anisotropy of voronoi cell by Minkowski tensor W1_02
        anisotropy = compute_anisotropy_w1_02(area_all, face_normal_vector)
        # 2.6
        feature_out = select_important_SRO_features(coordination_number_voronoi_tessellation, cellfraction,
                                                    modified_boop, cpe, anisotropy)
    return feature_out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_position_information(dump_path, frame):
    # 读取颗粒位置信息
    particle_info = open(dump_path + '/dump-' + str(frame) + '.sample', 'r')
    lines = particle_info.readlines()
    particle_info.close()
    lines = lines[9:]
    Par_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    Par_id_read = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    Par_xcor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    Par_ycor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    Par_zcor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    Par_radius_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    Par_id.sort()
    Par_xcor = [Par_xcor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_ycor = [Par_ycor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_zcor = [Par_zcor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_radius = [Par_radius_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_coord = np.array(list(zip(Par_xcor, Par_ycor, Par_zcor)))
    x_min = np.min([(Par_xcor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    y_min = np.min([(Par_ycor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    z_min = np.min([(Par_zcor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    x_max = np.max([(Par_xcor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    y_max = np.max([(Par_ycor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    z_max = np.max([(Par_zcor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    # x_min = float('%.4f' % (np.min(Par_xcor) - Par_radius[0]))
    # x_max = float('%.4f' % (np.max(Par_xcor) + Par_radius[0]))
    # y_min = float('%.4f' % (np.min(Par_ycor) - Par_radius[0]))
    # y_max = float('%.4f' % (np.max(Par_ycor) + Par_radius[0]))
    # z_min = float('%.4f' % (np.min(Par_zcor) - Par_radius[0]))
    # z_max = np.max(Par_zcor) + Par_radius[0]
    boundary = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    return Par_coord, Par_radius, boundary


def compute_area(vertices_input, adjacent_cell_input, vertices_id_input, simplice_input):
    area_judge_in = np.zeros(shape=[len(adjacent_cell_input), ], dtype=int)
    area_in = np.zeros(shape=[len(adjacent_cell_input), ])
    sing_area = np.zeros(shape=[len(simplice_input), ])
    for a in range(len(simplice_input)):
        sing_area[a] = compute_simplice_area(vertices_input[simplice_input[a][0]],
                                             vertices_input[simplice_input[a][1]],
                                             vertices_input[simplice_input[a][2]])
    for a in range(len(simplice_input)):
        for b in range(len(adjacent_cell_input)):
            if simplice_input[a][0] in vertices_id_input[b]:
                if simplice_input[a][1] in vertices_id_input[b]:
                    if simplice_input[a][2] in vertices_id_input[b]:
                        area_in[b] += sing_area[a]
    average_area = np.mean(area_in)
    for a in range(len(adjacent_cell_input)):
        if area_in[a] >= 0.05 * average_area:
            area_judge_in[a] = 1
    return area_judge_in, area_in


def compute_face_normal_vector(vertices_id, vertices, origin):
    face_normal_vector = np.zeros(shape=[len(vertices_id), 3])
    for i in range(len(vertices_id)):
        normal = np.cross(vertices[int(vertices_id[i][0])] - vertices[int(vertices_id[i][1])],
                          vertices[int(vertices_id[i][0])] - vertices[int(vertices_id[i][2])])
        normal /= np.linalg.norm(normal)
        # if np.dot(vertices[int(vertices_id[i][0])] - origin, normal) < 0:
        # normal *= -1
        for j in range(3):
            face_normal_vector[i][j] = normal[j]
    return face_normal_vector


def eliminate_useless_adjacent_cell(voronoi, points):
    # 剔除面积小于平均面积百分之五的邻域点,这可能会造成互为邻域颗粒之间的不对称，后面的程序需要逐一处理
    adjacent_cell_all = []
    area_all_particle = []
    face_normal_vector_all = []
    for x in range(len(voronoi)):
        vertices = voronoi[x]['vertices']
        ch = ConvexHull(vertices)
        simplice = np.array(ch.simplices)
        faces = voronoi[x]['faces']
        adjacent_cell = []
        for y in range(len(faces)):
            adjacent_cell.append(faces[y]['adjacent_cell'])
        vert_id = []
        for y in range(len(faces)):
            vert_id.append(faces[y]['vertices'])
        area_judge, area = compute_area(vertices, adjacent_cell, vert_id, simplice)
        adjacent_cell_use = []
        for y in range(len(adjacent_cell)):
            if area_judge[y] == 1:
                adjacent_cell_use.append(adjacent_cell[y])
        adjacent_cell_all.append(adjacent_cell_use)
        area_all_particle.append(area)
        face_normal_vector = compute_face_normal_vector(vert_id, vertices, points[x])
        face_normal_vector_all.append(face_normal_vector)
    return adjacent_cell_all, area_all_particle, face_normal_vector_all


def compute_voronoi_neighbour(points, radius, limits, d50):
    dispersion = 5 * d50 / 2
    voronoi = pyvoro.compute_voronoi(points, limits, dispersion, radius, periodic=[False] * 3)
    neighbour, area, face_normal_vector = eliminate_useless_adjacent_cell(voronoi, points)
    return voronoi, neighbour, area, face_normal_vector


def main_function(path, path_output, scenario, MRO_option, compute_feature_category, d50):
    # dump files
    mkdir(path_output)
    dump_path = path
    list_dir = os.listdir(dump_path)
    dump_frame = []
    file_prefix = 'dump-'
    file_suffix = '.sample'
    prefix_len = len(file_prefix)
    suffix_len = len(file_suffix)
    for file in list_dir:
        dump_frame.append(int(file[prefix_len:][:-suffix_len]))
    dump_frame = sorted(dump_frame)
    start_frame = np.min(dump_frame)
    end_frame = np.max(dump_frame)
    frame_interval = (end_frame - start_frame) / scenario
    frame_list = np.arange(start_frame, end_frame, frame_interval)
    frame_list = np.append(frame_list, end_frame)
    frame_list = frame_list.astype(int)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 循环开始，提取每一步数据
    #
    for idx, frame in enumerate(frame_list):
        if idx == 0:
            continue
        # step1. Gets the prepared coordinates information and neighborhood information
        Par_coord, Par_radius, boundary = read_position_information(path, frame)
        voronoi, voronoi_neighbour, area_all, face_normal_vector = compute_voronoi_neighbour(Par_coord, Par_radius,
                                                                                             boundary, d50)
        print(60 * '*')
        print('The %d th frame' % frame)
        print(60 * '*')
        # step2. Compute structure property(symmetry feature, interstice distribution and conventional feature) and output structure property
        # 2.1 set constant
        cutoff_distance = 1.3 * d50
        # the first minimum of g(r)
        # 2.2
        writer = pd.ExcelWriter(path_output + '/feature_all-' + str(frame) + '.xlsx')
        for feature_category in compute_feature_category:
            if feature_category == 'symmetry_feature':
                symmetry_feature = compute_symmetry_functions(points=Par_coord, radius=Par_radius, d50=d50)
                pd.DataFrame(symmetry_feature).to_excel(writer, sheet_name='symmetry feature')
            elif feature_category == 'interstice_distribution':
                interstice_distribution = compute_interstice_distribution(neighbour=voronoi_neighbour, points=Par_coord,
                                                                          radius=Par_radius)
                pd.DataFrame(interstice_distribution).to_excel(writer, sheet_name='interstice distribution')
            elif feature_category == 'conventional_feature':
                if MRO_option:
                    conventional_feature = compute_conventional_feature(points=Par_coord, area_all=area_all,
                                                                        face_normal_vector=face_normal_vector,
                                                                        neighbour=voronoi_neighbour, voronoi=voronoi,
                                                                        radius=Par_radius, MRO_option=MRO_option,
                                                                        cutoff_distance=cutoff_distance)
                    pd.DataFrame(conventional_feature).to_excel(writer, sheet_name='conventional feature')
                else:
                    conventional_feature = compute_conventional_feature(points=Par_coord, area_all=area_all,
                                                                        face_normal_vector=face_normal_vector,
                                                                        neighbour=voronoi_neighbour, voronoi=voronoi,
                                                                        radius=Par_radius, MRO_option=MRO_option,
                                                                        cutoff_distance=cutoff_distance)
                    columns_dict = ['coordination_number_voronoi_tessellation', 'cellfraction', 'q4', 'q6', 'q8',
                                    'w4', 'w6', 'w8', 'Cpe', 'anisotropic']
                    df = pd.DataFrame(conventional_feature, columns=columns_dict)
                    df.to_excel(writer, sheet_name='conventional feature')
        writer.save()
        writer.close()


# ==================================================================
# S T A R T
#
if __name__ == '__main__':
    path_ = 'D:/循环剪切试验和机器学习/cyc5300fric01shearrate025/sort position'
    path_output_ = 'D:/循环剪切试验和机器学习/cyc5300fric01shearrate025'
    scenario = 2499
    d50 = 0.02
    MRO_option = False
    # Compute_feature_category = ['symmetry_feature', 'interstice_distribution', 'conventional_feature']
    compute_feature_category = ['conventional_feature']
    argList = argv
    argc = len(argList)
    n = 0
    while n < argc:
        if argList[n][:2] == "-d":
            n += 1
            d50 = float(argList[n])
        elif argList[n][:4] == "-sce":
            n += 1
            scenario = int(argList[n])
        elif argList[n][:2] == "-h":
            print(__doc__)
            exit(0)
        n += 1
    print(path_)
    print("Running scenario:  %d" % scenario)
    main_function(path_, path_output_, scenario, MRO_option, compute_feature_category, d50)
