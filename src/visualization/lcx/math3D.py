# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
# -*- coding: utf-8 -*-
"""
#==============================================================================
File name : math3D.py
#==============================================================================

[History]
2016.06.01 : Taizo Yoshikawa : Original framework was created.
2016.09.05 : Taizo Yoshikawa : First release. 
2016.10.23 : Taizo Yoshikawa : Second release. 
2018.02.06 : Taizo Yoshikawa : Modified for ASTOM

[Description]
Math library for 3D model. 

#==============================================================================
Copyright (c) 2018, HONDA R&D Co., Ltd.
#==============================================================================
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted within NDA of HONDA:

Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer. Redistributions in binary 
form must reproduce the above copyright notice, this list of conditions and the 
following disclaimer in the documentation and/or other materials provided with 
the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGE.
#==============================================================================
"""

from math import *
import numpy as np

_EPS = np.finfo(float).eps * 4.0
PAI = 3.14159265


# ------------------------------------------------------------------------------
# zero3
# ------------------------------------------------------------------------------
def zero3():
    """
    zero3    
    """
    return (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------
# copy3
# ------------------------------------------------------------------------------
def copy3(v):
    """
    copy3    
    """
    return (v[0], v[1], v[2])


# ------------------------------------------------------------------------------
# inverse3
# ------------------------------------------------------------------------------
def inverse3(v):
    """
    inverse3    
    """
    return (-v[0], -v[1], -v[2])


# ------------------------------------------------------------------------------
# add3
# ------------------------------------------------------------------------------
def add3(v1, v2):
    """
    add3    
    """
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


# ------------------------------------------------------------------------------
# sub3
# ------------------------------------------------------------------------------
def sub3(v1, v2):
    """
    sub3    
    """
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


# ------------------------------------------------------------------------------
# scale3
# ------------------------------------------------------------------------------
def scale3(v, s):
    """
    scale3    
    """
    return (v[0] * s, v[1] * s, v[2] * s)


# ------------------------------------------------------------------------------
# lengthSq3
# ------------------------------------------------------------------------------
def lengthSq3(v):
    """
    lengthSq3    
    """
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


# ------------------------------------------------------------------------------
# length3
# ------------------------------------------------------------------------------
def length3(v):
    """
    length3    
    """
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


# ------------------------------------------------------------------------------
# normalize3
# ------------------------------------------------------------------------------
def normalize3(v):
    """
    normalize3    
    """
    l = length3(v)
    if l == 0:
        reurn(0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


# ------------------------------------------------------------------------------
# dot3
# ------------------------------------------------------------------------------
def dot3(v1, v2):
    """
    dot3    
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


# ------------------------------------------------------------------------------
# cross3
# ------------------------------------------------------------------------------
def cross3(v1, v2):
    """
    cross3    
    """
    return (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])


# ------------------------------------------------------------------------------
# perpendicular3
# ------------------------------------------------------------------------------
def perpendicular3(v):
    """
    perpendicular3    
    """
    if v[1] == 0 and v[2] == 0:
        return cross3(v, add3(v, (0, 1, 0)))
    return cross3(v, add3(v, (1, 0, 0)))


# ------------------------------------------------------------------------------
# Qzero
# ------------------------------------------------------------------------------
def Qzero():
    """
    Qzero    
    """
    return (1.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------
# Qcopy
# ------------------------------------------------------------------------------
def Qcopy(q):
    """
    Qcopy    
    """
    return (q[0], q[1], q[2], q[3])


# ------------------------------------------------------------------------------
# Qadd
# ------------------------------------------------------------------------------
def Qadd(q1, q2):
    """
    Qadd    
    """
    return (q1[0] + q2[0], q1[1] + q2[1], q1[2] + q2[2], q1[3] + q2[3])


# ------------------------------------------------------------------------------
# Qsub
# ------------------------------------------------------------------------------
def Qsub(q1, q2):
    """
    Qsub    
    """
    return (q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2], q1[3] - q2[3])


# ------------------------------------------------------------------------------
# Qscale
# ------------------------------------------------------------------------------
def Qscale(q, s):
    """
    Qscale    
    """
    return (q[0] * s, q[1] * s, q[2] * s, q[3] * s)


# ------------------------------------------------------------------------------
# QmagnitudeSq
# ------------------------------------------------------------------------------
def QmagnitudeSq(q):
    """
    QmagnitudeSq    
    """
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]


# ------------------------------------------------------------------------------
# Qmagnitude
# ------------------------------------------------------------------------------
def Qmagnitude(q):
    """
    Qmagnitude    
    """
    return np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


# ------------------------------------------------------------------------------
# Qconjugate
# ------------------------------------------------------------------------------
def Qconjugate(q):
    """
    Qconjugate    
    """
    return (q[0], -q[1], -q[2], -q[3])


# ------------------------------------------------------------------------------
# Qmultiply
# ------------------------------------------------------------------------------
def Qmultiply(q1, q2):
    """
    Qmultiply    
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2)


# ------------------------------------------------------------------------------
# Qnormalize
# ------------------------------------------------------------------------------
def Qnormalize(q):
    """
    Qnormalize    
    """
    m = Qmagnitude(q)
    if m == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / m, q[1] / m, q[2] / m, q[3] / m)


# ------------------------------------------------------------------------------
# Qinverse
# ------------------------------------------------------------------------------
def Qinverse(q):
    """
    Qinverse    
    """
    m2 = QmagnitudeSq(q)
    return (q[0] / m2, -q[1] / m2, -q[2] / m2, -q[3] / m2)


# ------------------------------------------------------------------------------
# Qdot
# ------------------------------------------------------------------------------
def Qdot(q1, q2):
    """
    Qdot    
    """
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


# ------------------------------------------------------------------------------
# QfromAngleAxis
# ------------------------------------------------------------------------------
def QfromAngleAxis(radians, x, y, z):
    """
    QfromAngleAxis    
    """
    radians /= 2.0
    s = np.sin(radians) / np.sqrt(x * x + y * y + z * z)
    return Qnormalize((np.cos(radians), x * s, y * s, z * s))


# ------------------------------------------------------------------------------
# QtoMatrix4x4
# ------------------------------------------------------------------------------
def QtoMatrix4x4(q):
    """
    QtoMatrix4x4    
    """
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    xx = 2.0 * qx * qx
    yy = 2.0 * qy * qy
    zz = 2.0 * qz * qz
    xy = 2.0 * qx * qy
    xz = 2.0 * qx * qz
    xw = 2.0 * qx * qw
    yw = 2.0 * qy * qw
    yz = 2.0 * qy * qz
    zw = 2.0 * qz * qw
    """
    return np.array([[1.0-yy-zz, xy+zw, xz-yw, 0.0], 
                     [xy-zw, 1.0-xx-zz, yz+xw, 0.0],
                     [xz+yw, yz-xw, 1.0-xx-yy, 0.0], 
                     [0.0, 0.0, 0.0, 1.0]])
    """
    return np.array([[1.0 - yy - zz, xy - zw, xz + yw, 0.0],
                     [xy + zw, 1.0 - xx - zz, yz - xw, 0.0],
                     [xz - yw, yz + xw, 1.0 - xx - yy, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # ,'f')
    # """


def QtoMatrix4x4_2(q, pos):
    """
    QtoMatrix4x4    
    """
    sqw = q[0] * q[0]
    sqx = q[1] * q[1]
    sqy = q[2] * q[2]
    sqz = q[3] * q[3]
    m00 = sqx - sqy - sqz + sqw  # since sqw + sqx + sqy + sqz =1
    m11 = -sqx + sqy - sqz + sqw
    m22 = -sqx - sqy + sqz + sqw

    tmp1 = q[1] * q[2]
    tmp2 = q[3] * q[0]
    m01 = 2.0 * (tmp1 + tmp2)
    m10 = 2.0 * (tmp1 - tmp2)

    tmp1 = q[1] * q[3]
    tmp2 = q[2] * q[0]
    m02 = 2.0 * (tmp1 - tmp2)
    m20 = 2.0 * (tmp1 + tmp2)

    tmp1 = q[2] * q[3]
    tmp2 = q[1] * q[0]
    m12 = 2.0 * (tmp1 + tmp2)
    m21 = 2.0 * (tmp1 - tmp2)

    a1 = 0
    a2 = 0
    a3 = 0
    a1 = pos[0]
    a2 = pos[1]
    a3 = pos[2]

    m03 = a1 - a1 * m00 - a2 * m01 - a3 * m02
    m13 = a2 - a1 * m10 - a2 * m11 - a3 * m12
    m23 = a3 - a1 * m20 - a2 * m21 - a3 * m22
    m30 = 0.0
    m31 = 0.0
    m32 = 0.0
    m33 = 1.0

    return np.array([[m00, m01, m02, m03],
                     [m10, m11, m12, m13],
                     [m20, m22, m22, m23],
                     [m30, m31, m32, m33]])


# ------------------------------------------------------------------------------
# QtoMatrix3x3
# ------------------------------------------------------------------------------
def QtoMatrix16(q):
    """
    QtoMatrix3x3    
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    return (1 - y * y - z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w,
            0,
            2 * x * y + 2 * z * w,
            1 - x * x - z * z,
            2 * y * z - 2 * x * w,
            0,
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - x * x - y * y,
            0,
            0,
            0,
            0,
            1)

    """
    return (1-y*y-z*z, 
            2*x*y+2*z*w, 
            2*x*z-2*y*w,
            0,
            2*x*y-2*z*w,
            1-x*x-z*z,
            2*y*z+2*x*w,
            0,
            2*x*z-2*y*w,
            2*y*z-2*x*w,
            1-x*x-y*y,
            0,
            0,
            0,
            0,
            1)

    
    return (w*w+x*x-y*y-z*z, 
            2*x*y+2*z*w, 
            2*x*z+2*y*w,
            0,
            2*x*y-2*z*w,
            w*w-x*x+y*y-z*z,
            2*y*z-2*z*w,
            0,
            2*x*z-2*y*w,
            2*y*z-2*x*w,
            w*w-x*x-y*y+z*z,
            0,
            0,
            0,
            0,
            1)
    """


# ------------------------------------------------------------------------------
# Matrix3x3toQ
# ------------------------------------------------------------------------------
def Matrix3x3toQ(matrix):
    """
    Matrix3x3toQ    
    """

    """
    np_matrix= np.matrix([[ matrix[0][0], matrix[0][1], matrix[0][2]],
                          [ matrix[1][0], matrix[1][1], matrix[1][2]],
                          [ matrix[2][0], matrix[2][1], matrix[2][2]]])
    """
    MatrixT = matrix.transpose()
    den = np.array([1.0 + MatrixT[0, 0] - MatrixT[1, 1] - MatrixT[2, 2],
                    1.0 - MatrixT[0, 0] + MatrixT[1, 1] - MatrixT[2, 2],
                    1.0 - MatrixT[0, 0] - MatrixT[1, 1] + MatrixT[2, 2],
                    1.0 + MatrixT[0, 0] + MatrixT[1, 1] + MatrixT[2, 2]])

    max_idx = np.flatnonzero(den == max(den))[0]

    q = np.zeros(4)
    q[max_idx] = 0.5 * np.sqrt(max(den))
    denom = 4.0 * q[max_idx]
    if (max_idx == 0):
        q[1] = (MatrixT[1, 0] + MatrixT[0, 1]) / denom
        q[2] = (MatrixT[2, 0] + MatrixT[0, 2]) / denom
        q[3] = -(MatrixT[2, 1] - MatrixT[1, 2]) / denom
    if (max_idx == 1):
        q[0] = (MatrixT[1, 0] + MatrixT[0, 1]) / denom
        q[2] = (MatrixT[2, 1] + MatrixT[1, 2]) / denom
        q[3] = -(MatrixT[0, 2] - MatrixT[2, 0]) / denom
    if (max_idx == 2):
        q[0] = (MatrixT[2, 0] + MatrixT[0, 2]) / denom
        q[1] = (MatrixT[2, 1] + MatrixT[1, 2]) / denom
        q[3] = -(MatrixT[1, 0] - MatrixT[0, 1]) / denom
    if (max_idx == 3):
        q[0] = -(MatrixT[2, 1] - MatrixT[1, 2]) / denom
        q[1] = -(MatrixT[0, 2] - MatrixT[2, 0]) / denom
        q[2] = -(MatrixT[1, 0] - MatrixT[0, 1]) / denom

    return q


# ------------------------------------------------------------------------------
# QrotateVector
# ------------------------------------------------------------------------------
def QrotateVector(q, v):
    """
    QrotateVector    
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    x, y, z = v[0], v[1], v[2]

    ww = qw * qw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz

    return (ww * x + xx * x - yy * x - zz * x + 2 * ((xy - wz) * y + (xz + wy) * z),
            ww * y - xx * y + yy * y - zz * y + 2 * ((xy + wz) * x + (yz - wx) * z),
            ww * z - xx * z - yy * z + zz * z + 2 * ((xz - wy) * x + (yz + wx) * y))


# ------------------------------------------------------------------------------
# Qinterpolate
# ------------------------------------------------------------------------------
def Qinterpolate(q1, q2, s, shortest=True):
    """
    Qinterpolate    
    """
    ca = Qdot(q1, q2)
    if shortest and ca < 0:
        ca = -ca
        neg_q2 = True
    else:
        neg_q2 = False
    o = np.acos(ca)
    so = np.sin(o)

    if (abs(so) <= 1E-12):
        return Qcopy(q1)

    a = np.sin(o * (1.0 - s)) / so
    b = np.sin(o * s) / so
    if neg_q2:
        return Qsub(scale(q1, a), Qscale(q2, b))
    else:
        return Qadd(scale(q1, a), Qscale(q2, b))


# ------------------------------------------------------------------------------
# setRotationMatrix
# ------------------------------------------------------------------------------
def setRotationMatrix(angle, axis):
    """
    setRotationMatrix    
    """
    if (axis == 0):
        return np.matrix([[1., 0., 0.],
                          [0., np.cos(angle), np.sin(angle)],
                          [0., -np.sin(angle), np.cos(angle)]])

    elif (axis == 1):

        return np.matrix([[np.cos(angle), 0., -np.sin(angle)],
                          [0., 1., 0.],
                          [np.sin(angle), 0., np.cos(angle)]])

    elif (axis == 2):

        return np.matrix([[np.cos(angle), np.sin(angle), 0.],
                          [-np.sin(angle), np.cos(angle), 0.],
                          [0., 0., 1.]])


# ------------------------------------------------------------------------------
# rotationMatrix_Array
# ------------------------------------------------------------------------------
def rotationMatrix_Inverse(matrix):
    """
    rotationMatrix_Inverse    
    """
    np_matrix = np.matrix([[matrix[0][0], matrix[0][1], matrix[0][2]],
                           [matrix[1][0], matrix[1][1], matrix[0][0]],
                           [matrix[2][0], matrix[2][1], matrix[2][2]]])

    return np.linalg.inv(np_matrix)


# ------------------------------------------------------------------------------
# toQuaternion
# ------------------------------------------------------------------------------
def toQuaternion(pitch, roll, yaw):
    """
    toQuaternion    
    """

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    q = [w, x, y, z]

    return q


# ------------------------------------------------------------------------------
# quaternion_to_euler_angle
# ------------------------------------------------------------------------------
def quaternion_to_euler_angle(w, x, y, z):
    """
    quaternion_to_euler_angle
    """
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z


# ------------------------------------------------------------------------------
# QtoEulerAngle
# ------------------------------------------------------------------------------
def QtoEulerAngle(q):
    """
    QtoEulerAngle    
    """
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # ysqr = y * y

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    th1 = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)

    t2 = 1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    th2 = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    th3 = np.arctan2(t3, t4)

    return th1, th2, th3


# ------------------------------------------------------------------------------
# QtoEulerAngle
# ------------------------------------------------------------------------------
def QtoEulerAngle2(q):
    """
    QtoEulerAngle    
    """
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    th1 = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    th2 = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    th3 = np.arctan2(t3, t4)

    th = [th1, th2, th3]

    return th


# ------------------------------------------------------------------------------
# EulerAngleToQ
# ------------------------------------------------------------------------------
def EulerAngleToQ(euler_angle):
    """
    EulerAngletoQ    
    """
    th1 = euler_angle[0]
    th2 = euler_angle[1]
    th3 = euler_angle[2]

    q1 = np.cos(th1 / 2) * np.cos(th2 / 2) * np.cos(th3 / 2) + np.sin(th1 / 2) * np.sin(th2 / 2) * np.sin(th3 / 2)
    q2 = np.sin(th1 / 2) * np.cos(th2 / 2) * np.cos(th3 / 2) - np.cos(th1 / 2) * np.sin(th2 / 2) * np.sin(th3 / 2)
    q3 = np.cos(th1 / 2) * np.sin(th2 / 2) * np.cos(th3 / 2) + np.sin(th1 / 2) * np.cos(th2 / 2) * np.sin(th3 / 2)
    q4 = np.cos(th1 / 2) * np.cos(th2 / 2) * np.sin(th3 / 2) - np.sin(th1 / 2) * np.sin(th2 / 2) * np.cos(th3 / 2)

    q = [q1, q2, q3, q4]

    return q


# ------------------------------------------------------------------------------
# quaternionToMatrix
# ------------------------------------------------------------------------------
def quaternionToMatrix(quaternion):
    """
    quaternionToMatrix    
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    q *= sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


# ------------------------------------------------------------------------------
# matrixToQuaternion
# ------------------------------------------------------------------------------
def matrixToQuaternion(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# ------------------------------------------------------------------------------
# setRotationMatrix
# ------------------------------------------------------------------------------
def rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS):
    """
    setRotationMatrix    
    """
    Mat_x = np.array([[1., 0., 0.],
                      [0., np.cos(VIEW_OFFSET_ROT[0]), np.sin(VIEW_OFFSET_ROT[0])],
                      [0., -np.sin(VIEW_OFFSET_ROT[0]), np.cos(VIEW_OFFSET_ROT[0])]])

    Mat_y = np.array([[np.cos(VIEW_OFFSET_ROT[1]), 0., -np.sin(VIEW_OFFSET_ROT[1])],
                      [0., 1., 0.],
                      [np.sin(VIEW_OFFSET_ROT[1]), 0., np.cos(VIEW_OFFSET_ROT[1])]])

    Mat_z = np.array([[np.cos(VIEW_OFFSET_ROT[2]), np.sin(VIEW_OFFSET_ROT[2]), 0.],
                      [-np.sin(VIEW_OFFSET_ROT[2]), np.cos(VIEW_OFFSET_ROT[2]), 0.],
                      [0., 0., 1.]])

    matrix = np.dot(Mat_x, np.dot(Mat_y, Mat_z))
    X_rot = np.dot(matrix, link_pos_1)

    return X_rot + VIEW_OFFSET_POS


# ------------------------------------------------------------------------------
# vectorsToEulerAngles
# ------------------------------------------------------------------------------
def vectorsToEulerAngles(vector_1, vector_2):
    """
    vectorsToEulerAngles    
    """
    a_vec = vector_1 / np.linalg.norm(vector_1)
    b_vec = vector_2 / np.linalg.norm(vector_2)

    cross = np.cross(a_vec, b_vec)
    ab_angle = np.arccos(np.dot(a_vec, b_vec))

    vx = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    Rmat = np.identity(3) * np.cos(ab_angle) + (1 - np.cos(ab_angle)) * np.outer(cross, cross) + np.sin(ab_angle) * vx

    rotationMatrixToEulerAngles(Rmat)
    # return rotationMatrixToEulerAngles(Rmat)
    return Rmat  # euler_angles_from_rotation_matrix(Rmat)
    # return rotation_matrix_to_euler_angles(Rmat)


# ------------------------------------------------------------------------------
# vectorsToEulerAngles
# ------------------------------------------------------------------------------
def vectorsTo4x4Matrix(vector_1, vector_2):
    """
    vectorsToEulerAngles    
    """
    a_vec = vector_1 / np.linalg.norm(vector_1)
    b_vec = vector_2 / np.linalg.norm(vector_2)

    cross = np.cross(a_vec, b_vec)
    ab_angle = np.arccos(np.dot(a_vec, b_vec))

    vx = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    Rmat = np.identity(3) * np.cos(ab_angle) + (1 - np.cos(ab_angle)) * np.outer(cross, cross) + np.sin(ab_angle) * vx
    Rmat44 = np.array([[Rmat[0][0], Rmat[0][1], Rmat[0][2], 0], [Rmat[1][0], Rmat[1][1], Rmat[1][2], 0],
                       [Rmat[2][0], Rmat[2][1], Rmat[2][2], 0], [0, 0, 0, 1]])

    return Rmat44  # euler_angles_from_rotation_matrix(Rmat)


# ------------------------------------------------------------------------------
# vectorsToEulerAngles
# ------------------------------------------------------------------------------
def vectors2EulerAngles(vector_1, vector_2, type):
    """
    vectorsToEulerAngles    
    """
    a_vec = vector_1 / np.linalg.norm(vector_1)
    b_vec = vector_2 / np.linalg.norm(vector_2)

    cross = np.cross(a_vec, b_vec)
    ab_angle = np.arccos(np.dot(a_vec, b_vec))

    vx = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    Rmat = np.identity(3) * np.cos(ab_angle) + (1 - np.cos(ab_angle)) * np.outer(cross, cross) + np.sin(ab_angle) * vx
    # return rotationMatrixToEulerAngles(Rmat)
    # return euler_angles_from_rotation_matrix(Rmat)
    return rotation_matrix_2_euler_angles(Rmat, type)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_2_euler_angles(R, type):
    if (type == "YXZ"):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        th_x = np.arctan(-R[2, 2] / R[2, 1])
        y = np.arctan(-sy / R[2, 0])
        z = np.arctan(-R[0, 0] / R[1, 0])
    if (type == "ZYX"):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        th_x = np.arctan(-R[2, 2] / R[2, 1])
        y = np.arctan(-sy / R[2, 0])
        z = np.arctan(-R[0, 0] / R[1, 0])

    if (sy > 1e-6):
        th_x = np.arctan(-R[2, 2] / R[2, 1])
        y = np.arctan(-sy / R[2, 0])
        z = np.arctan(-R[0, 0] / R[1, 0])
    else:
        x = np.arctan(-R[1, 1] / R[1, 2])
        y = np.arctan(-sy / R[2, 0])
        z = 0

    return x, y, z  # np.array([x, y, z])


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    # assert(isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if (sy > 1e-6):
        x = np.arctan(R[2, 2] / R[2, 1])
        y = np.arctan(-sy / R[2, 0])
        z = np.arctan(R[0, 0] / R[1, 0])
    else:
        x = np.arctan(-R[1, 1] / R[1, 2])
        y = np.arctan(-sy / R[2, 0])
        z = 0

    return x, y, z  # np.array([x, y, z])


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def euler_angles_from_rotation_matrix(R):
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = PAI / 2.0
        psi = np.arctan(R[0, 2] / R[0, 1])
    elif isclose(R[2, 0], 1.0):
        theta = -PAI / 2.0
        psi = np.arctan(R[0, 2] / R[0, 1])
    else:
        theta = -np.arcsin(R[2, 0])
        psi = np.arctan((R[2, 2]) / (R[2, 1]))
        phi = np.arctan((R[0, 0]) / (R[1, 0]))
    return psi, theta, phi
    # return theta, psi, phi

# ==============================================================================
# End Of File
# ==============================================================================
