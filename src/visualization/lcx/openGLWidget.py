# -*- coding: utf-8 -*-
"""
#==============================================================================
File name : QTWidget.py
#==============================================================================

[History]
2017.11.29 : Taizo Yoshikawa : Original framework was created.
2018.02.06 : Taizo Yoshikawa : Modified for ASTOM

[Description]
GUI software to display OpenGL 3D model. 

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
from OpenGL import GL, GLU
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtOpenGL import *

import numpy as np

from .OBJ_parser import OBJ_parser
from .math3D import Qzero, QtoMatrix4x4, rotateAndTrans

# -------------------------------------------------------------------

PAI = 3.14159265
RAD2DEG = 180 / PAI
DEG2RAD = 1 / RAD2DEG
DELTA_T = 0.005

light_ambient0 = [0.2, 0.2, 0.2, 1.0]
light_diffuse0 = [0.8, 0.8, 0.8, 1.0]
light_specular0 = [0.0, 0.0, 0.0, 1.0]
light_position0 = [5.0, 5.0, 5.0, 0.0]

light_ambient1 = [0.2, 0.2, 0.2, 1.0]
light_diffuse1 = [0.8, 0.8, 0.8, 1.0]
light_specular1 = [0.0, 0.0, 0.0, 1.0]
light_position1 = [-5.0, 5.0, 2.0, 0.0]

initial_position = [0, 0, 0.838961,
                    -0.029408328, -0.001456459, 0.927124,
                    -0.045147217, -0.00376149, 1.027718,
                    -0.052426016, -0.005954558, 1.11942,
                    -0.052679665, -0.008228739, 1.211408,
                    -0.04560564, -0.01149259, 1.340861,
                    -0.009713214, -0.013609861, 1.418622,
                    -0.04816721, -0.038337177, 1.282924,
                    -0.04469396, -0.167907811, 1.253383,
                    -0.013442628, -0.160954295, 0.969709,
                    0.023433564, -0.185046428, 0.740712,
                    -0.049294567, 0.018235645, 1.284412,
                    -0.048349854, 0.149557606, 1.263684,
                    -0.013498041, 0.185779783, 0.98267,
                    0.033606161, 0.251052675, 0.763807,
                    -0.000111992, -0.075679498, 0.837805,
                    -0.03822161, -0.150738501, 0.452848,
                    -0.117720183, -0.213230186, 0.082873,
                    0.023097166, 0.255377602, 0.020483,
                    0.0000580, 0.075675289, 0.840361,
                    -0.020986851, 0.152738582, 0.454491,
                    -0.036620911, 0.231686709, 0.079485,
                    0.104056992, 0.275676561, 0.01727]

POLYGON_MODEL_MODE = True
LIGHTING = True
TWO_LIGHTING = True
BACKGROUND_COLOR = [0.3, 0.3, 0.8, 0.0]
RL_test = 1
VIEW_OFFSET_POS = [0, 0, 2.5]
VIEW_OFFSET_POS2 = [0, 0, 5.0]
VIEW_OFFSET_ROT = [0 * DEG2RAD, 180 * DEG2RAD, 0 * DEG2RAD]


# ==============================================================================
# OpenGL Display
# ==============================================================================
class openGLWidget(QGLWidget):
    xRotationChanged = QtCore.pyqtSignal(int)
    yRotationChanged = QtCore.pyqtSignal(int)
    zRotationChanged = QtCore.pyqtSignal(int)

    # -------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------
    def __init__(self, parent, width, height, modelIndex):
        QGLWidget.__init__(self, parent)
        # self.setMinimumSize(300, 300)

        self.gl_width = width
        self.gl_height = height

        self.index_displayList = []
        self.indexID = modelIndex

        self.init_model()

    # -------------------------------------------------------------------
    # init
    # -------------------------------------------------------------------
    def init(width, height):

        self.init_view()

    # -------------------------------------------------------------------
    # init
    # -------------------------------------------------------------------
    def init_view(self):

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glShadeModel(GL.GL_SMOOTH)  # (GL.GL_FLAT) (GL.GL_SMOOTH)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        ambient = np.array([0.6, 0.6, 0.6, 1.0])
        diffuse = np.array([1.0, 0.8, 0.0, 1.0])
        specular = np.array([1.0, 1.0, 1.0, 1.0])
        shine = 1.0
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, ambient)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, diffuse)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, specular)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, shine)

        lightZeroPosition = [0., 10., 10., 1]
        glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
        glEnable(GL_LIGHT0)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # GL_DIFFUSE)

    # -------------------------------------------------------------------
    # init_model
    # -------------------------------------------------------------------
    def init_model(self):
        """
        init_model
        """
        self.object_1 = 0
        self.object_2 = 0
        self.xRot = 90
        self.yRot = 30
        self.zRot = 0
        self.zoom = 0.7

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        self.link_node_num = len(self.indexID)

        self.position = np.zeros((self.link_node_num, 3))
        for node in range(0, self.link_node_num):
            self.position[node][0] = initial_position[node * 3]
            self.position[node][1] = initial_position[node * 3 + 1]
            self.position[node][2] = initial_position[node * 3 + 2]

        self.quaternion = np.zeros((self.link_node_num, 4))

        self.foot_point_R = np.array(
            [[0.185, -0.004, -0.1], [0.16, -0.05, -0.1], [0.1, -0.06, -0.1], [0.027, -0.05, -0.1],
             [-0.038, -0.03, -0.1], [-0.048, -0.025, -0.1], [-0.055, -0.01, -0.1], [-0.05, 0.005, -0.1],
             [-0.043, 0.015, -0.1], [-0.027, 0.02, -0.1], [0.027, 0.025, -0.1], [0.052, 0.026, -0.1],
             [0.093, 0.03, -0.1], [0.11, 0.03, -0.1], [0.16, 0.03, -0.1], [0.177, 0.025, -0.1]])

        self.foot_point_L = np.array(
            [[0.177, -0.025, -0.1], [0.16, -0.03, -0.1], [0.11, -0.03, -0.1], [0.093, -0.03, -0.1],
             [0.052, -0.026, -0.1], [0.027, -0.025, -0.1], [-0.027, -0.02, -0.1], [-0.043, -0.015, -0.1],
             [-0.05, -0.005, -0.1], [-0.055, 0.01, -0.1], [-0.048, 0.025, -0.1], [-0.038, 0.03, -0.1],
             [0.027, 0.05, -0.1], [0.1, 0.06, -0.1], [0.16, 0.05, -0.1], [0.185, 0.004, -0.1]])
        self.foot_corner_num = len(self.foot_point_R)

        offset = np.array([0.0, 0.0, 0.0])
        for point in range(0, self.foot_corner_num):
            link_pos_1 = np.array(
                [self.foot_point_R[point, 0], self.foot_point_R[point, 1], self.foot_point_R[point, 2]])
            link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, offset)
            self.foot_point_R[point, :] = link_pos_1

            link_pos_1 = rotateAndTrans(self.foot_point_L[point, :], VIEW_OFFSET_ROT, offset)
            self.foot_point_L[point, :] = link_pos_1

        self.foot_contact_point_R = np.array(
            [[0.12, 0.05, -0.08], [0.12, -0.06, -0.08], [-0.05, -0.04, -0.08], [-0.05, 0.035, -0.08]])
        self.foot_contact_point_L = np.array(
            [[0.12, 0.06, -0.08], [0.12, -0.05, -0.08], [-0.05, -0.035, -0.08], [-0.05, 0.04, -0.08]])

        self.foot_pos_r = self.foot_point_R
        self.foot_pos_l = self.foot_point_L
        self.foot_q_r = Qzero()
        self.foot_q_l = Qzero()

        self.center_pos_offset = [0, 0, 0]
        self.COG = [0, 0, 0]
        self.Counter = 0

        self.floor_level = 0.0

        self.GRP = np.array([0, 0, 0])
        self.GRF = np.array([0, 0, 0])

        self.GRP_R = np.array([0, 0, 0])
        self.GRF_R = np.array([0, 0, 0])

        self.GRP_L = np.array([0, 0, 0])
        self.GRF_L = np.array([0, 0, 0])

    # -------------------------------------------------------------------
    # initializeGL
    # -------------------------------------------------------------------
    def initializeGL(self):
        """
        initializeGL
        """
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

        GL.glClearColor(BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2], BACKGROUND_COLOR[3])
        glClearDepth(1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, 1.0, 1.0, 30.0)

        self.displaylist()

    # -------------------------------------------------------------------
    # paintGL
    # -------------------------------------------------------------------
    def paintGL(self):
        """
        paintGL
        """
        self.init_view()
        self.display_model()

    # -------------------------------------------------------------------
    # resizeGL
    # -------------------------------------------------------------------
    def resizeGL(self, w, h):
        """
        resizeGL
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    # -------------------------------------------------------------------
    # changeBackColor
    # -------------------------------------------------------------------
    def changeBackColor(self):
        """
        changeBackColor
        """
        c = self.br
        self.br = self.bg
        self.bg = self.bb
        self.bb = c
        self.updateGL()

    # -------------------------------------------------------------------
    # on_mouse_drag
    # -------------------------------------------------------------------
    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        """
        on_mouse_drag
        """
        print(button)

    # -------------------------------------------------------------------
    # mousePressEvent
    # -------------------------------------------------------------------
    def mousePressEvent(self, event):
        """
        mousePressEvent
        """

        self.lastPos = event.pos()

    # -------------------------------------------------------------------
    # mouseMoveEvent
    # -------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        """
        mouseMoveEvent
        """

        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
            # print(self.xRot + 8 * dy)

        elif event.buttons() & QtCore.Qt.RightButton:

            zoom_checked = self.zoom - dy * 0.1
            if (zoom_checked < 0.1):
                zoom_checked = 0.1
            elif (zoom_checked > 10):
                zoom_checked = 10

            self.zoom = zoom_checked
            self.updateGL()
            # print(zoom_checked)

        self.lastPos = event.pos()

    # -------------------------------------------------------------------
    # minimumSizeHint
    # -------------------------------------------------------------------
    def minimumSizeHint(self):
        """
        minimumSizeHint
        """
        return QtCore.QSize(500, 500)

    # -------------------------------------------------------------------
    # sizeHint
    # -------------------------------------------------------------------
    def sizeHint(self):
        """
        sizeHint
        """
        return QtCore.QSize(1000, 1000)

    # -------------------------------------------------------------------
    # setXRotation
    # -------------------------------------------------------------------
    def setXRotation(self, angle):
        """
        setXRotation
        """
        # angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.updateGL()

    # -------------------------------------------------------------------
    # setYRotation
    # -------------------------------------------------------------------
    def setYRotation(self, angle):
        """
        setYRotation
        """
        # angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.updateGL()

    # -------------------------------------------------------------------
    # setZRotation
    # -------------------------------------------------------------------
    def setZRotation(self, angle):
        """
        setZRotation
        """
        # angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.updateGL()

    # -------------------------------------------------------------------
    # display
    # -------------------------------------------------------------------
    def display(self):
        """
        display
        """
        # """
        gluLookAt(self.center_pos_offset[0], self.center_pos_offset[1], self.center_pos_offset[2],
                  self.center_pos_offset[0] + 1, self.center_pos_offset[1] + 1, self.center_pos_offset[2] + 2,
                  0.1, 0.1, 0.0)
        # """
        self.updateGL()

    # -------------------------------------------------------------------
    # updateData
    # -------------------------------------------------------------------
    def updateData(self, nodeID, nodeName, position, quaternion):
        """
        updateData
        """

        self.indexID[nodeID] = nodeName
        self.position[nodeID] = position
        self.quaternion[nodeID] = quaternion

    # -------------------------------------------------------------------
    # updateData
    # -------------------------------------------------------------------
    def updateForceData(self, GRP_, GRF_, GRP_R_, GRF_R_, GRP_L_, GRF_L_):
        """
        updateData
        """

        self.GRP = GRP_
        self.GRF = GRF_

        self.GRP_R = GRP_R_
        self.GRF_R = GRF_R_

        self.GRP_L = GRP_L_
        self.GRF_L = GRF_L_

    # -------------------------------------------------------------------
    # display_model
    # -------------------------------------------------------------------
    def display_model(self):
        """
        display_model
        """
        self.init_view()

        GL.glTranslated(0.0, 1.2, -2.0)  # self.center_pos_offset[0]

        GL.glRotated(90.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        # GL.glRotated(-self.zRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(-self.yRot / 16.0, 0.0, 0.0, 1.0)

        GL.glScalef(self.zoom, self.zoom, self.zoom)
        self.glDisplayFloor()
        self.display_polygon_model()
        self.glDisplayGRFs()

        # self.draw_floor()
        self.glDisplayCoordinate()
        # self.glDisplayFoot()

    # -------------------------------------------------------------------
    # display_polygon_model
    # -------------------------------------------------------------------
    def display_polygon_model(self):
        """
        display_polygon_model
        """
        self.glDisplaySegment2('Pelvis', 'L5')
        self.glDisplaySegment2('L5', 'T8')
        self.glDisplaySegment2('T8', 'Neck')
        self.glDisplaySegment2('Neck', 'Head')

        self.glDisplaySegment('RightUpperArm', 'RightForeArm')
        self.glDisplaySegment('LeftUpperArm', 'LeftForeArm')

        self.glDisplaySegment('RightForeArm', 'RightHand')
        self.glDisplaySegment('LeftForeArm', 'LeftHand')

        self.glDisplaySegment('RightUpperLeg', 'RightLowerLeg')
        self.glDisplaySegment('LeftUpperLeg', 'LeftLowerLeg')

        self.glDisplaySegment('RightLowerLeg', 'RightFoot')
        self.glDisplaySegment('LeftLowerLeg', 'LeftFoot')

        self.glDisplaySegment2('RightFoot', 'RightFoot')
        self.glDisplaySegment2('LeftFoot', 'LeftFoot')

    # -------------------------------------------------------------------
    # glDisplayCoordinate
    # -------------------------------------------------------------------
    def glDisplayCoordinate(self):
        """
        glDisplayCoordinate
        """
        link_pos_1 = np.array([0, 0, 0])
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        GL.glPushMatrix();

        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)

        GL.glBegin(GL.GL_POLYGON)
        GL.glCallList(self.index_displayList.index("Coordinate") + 1)  # self.indexID.index(indexID_1))
        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayFloor
    # -------------------------------------------------------------------
    def glDisplayFloor(self):
        """
        glDisplayFloor
        """
        link_pos_1 = np.array([0, 0, 0])
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        GL.glPushMatrix();

        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)

        GL.glBegin(GL.GL_POLYGON)
        GL.glCallList(self.index_displayList.index("Floor") + 1)  # self.indexID.index(indexID_1))
        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayGRFs
    # -------------------------------------------------------------------
    def glDisplayGRFs(self):
        """
        glDisplayGRFs
        """
        self.glDisplayGRF("GRF", self.GRF, self.GRP)
        self.glDisplayGRF("GRF_R", self.GRF_R, self.GRP_R)
        self.glDisplayGRF("GRF_L", self.GRF_L, self.GRP_L)

    # -------------------------------------------------------------------
    # glDisplayGRF
    # -------------------------------------------------------------------
    def glDisplayGRF(self, linkNodeName, GRFs_, GRPs_):
        """
        glDisplayGRF
        """
        # print(GRPs_)
        link_pos_1 = GRPs_
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        GL.glPushMatrix();

        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)

        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)

        # GL.glPushMatrix();

        force_max_ = 1000.0
        force_ = np.linalg.norm(GRFs_)
        height_ = force_ / force_max_

        if (linkNodeName == "GRF"):
            color_ = np.array([1, 0, 0])
        elif (linkNodeName == "GRF_R"):
            color_ = np.array([0, 1, 0])
        elif (linkNodeName == "GRF_L"):
            color_ = np.array([1, 1, 0])

        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, color_)
        self.draw_array(0.03, height_, 10, color_)

        GL.glPopMatrix();

        # -------------------------------------------------------------------

    # glDisplaySegment
    # -------------------------------------------------------------------
    def glDisplaySegment(self, indexID_1, indexID_2):
        """
        glDisplaySegment
        """
        # GL.glColor(self.trolltechGreen)
        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])
        link_pos_2 = rotateAndTrans(link_pos_2, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        glPushMatrix()

        if (indexID_1 == "T5"):
            link_pos_1_ = np.array(self.position[self.indexID.index("RightUpperArm"), :])
            link_pos_2_ = np.array(self.position[self.indexID.index("LeftUpperArm"), :])
            link_pos_ = [(link_pos_1_[0] + link_pos_2_[0]) / 2,
                         (link_pos_1_[1] + link_pos_2_[1]) / 2,
                         (link_pos_1_[2] + link_pos_2_[2]) / 2]
            glTranslatef(link_pos_1_[0], link_pos_1_[1], link_pos_1_[2])

        elif (indexID_1 == "RightUpperLeg" or indexID_1 == "LeftUpperLeg"):
            glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2] + 0.1)

        elif (indexID_1 == "RightLowerLeg" or indexID_1 == "LeftLowerLeg",
              indexID_1 == "RightFoot" or indexID_1 == "LeftFoot"):
            glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2] + 0.05)

        elif (indexID_1 == "RightUpperArm" or indexID_1 == "RightForeArm" or indexID_1 == "RightHand" or
              indexID_1 == "LeftUpperArm" or indexID_1 == "LeftForeArm" or indexID_1 == "LeftHand"):
            glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        else:
            glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        if (indexID_1 == "Pelvis" or indexID_1 == "T5" or indexID_1 == "Neck"):
            segment = link_pos_1 - link_pos_2
        else:
            segment = link_pos_2 - link_pos_1

        # if(segment[2]==0):
        #    segment[2]=0.001

        vec = np.sqrt(segment[0] * segment[0] + segment[1] * segment[1] + segment[2] * segment[2])
        if (vec == 0):
            vec = 0.001

        ax = 1.0 * RAD2DEG * np.arccos(segment[2] / vec)

        if (segment[2] < 0.0):
            ax = -ax

        rx = -segment[1] * segment[2]
        ry = segment[0] * segment[2]
        rz = segment[2] * segment[2]

        if (
                indexID_1 == "Neck" or indexID_1 == "T5" or indexID_1 == "Pelvis"):  # or indexID_1=="RightFoot" or indexID_1=="LeftFoot"): #indexID_1=="Pelvis" or

            # self.quaternion[self.indexID.index(indexID_1),2] = -self.quaternion[self.indexID.index(indexID_1),2]

            # GL.glMultMatrixf(QtoMatrix16(self.quaternion[self.indexID.index(indexID_1),:]))
            if (indexID_1 == "T5"):
                GL.glMultMatrixf(QtoMatrix4x4(self.quaternion[self.indexID.index("T5"), :]))
            else:
                GL.glMultMatrixf(QtoMatrix4x4(self.quaternion[self.indexID.index(indexID_1), :]))

            GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)
            GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
            GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)

        elif (indexID_1 == "RightFoot" or indexID_1 == "LeftFoot"):  # indexID_1=="Pelvis" or

            # GL.glMultMatrixf(m3d.QtoMatrix4x4(self.quaternion[self.indexID.index(indexID_1),:]))
            GL.glRotatef(ax, rx, ry, 0.0)  # rz)#0.0)

            GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)
            GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
            GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)

        else:
            GL.glRotatef(ax, rx, ry, 0.0)  # rz)#0.0)
            GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)
            GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
            GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)

        GL.glCallList(self.index_displayList.index(indexID_1) + 1)  # self.indexID.index(indexID_1))

        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplaySegment
    # -------------------------------------------------------------------
    def glDisplaySegment2(self, indexID_1, indexID_2):
        """
        glDisplaySegment
        """
        # GL.glColor(self.trolltechGreen)
        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])
        link_pos_2 = rotateAndTrans(link_pos_2, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        segment = link_pos_2 - link_pos_1
        vec = np.sqrt(segment[0] * segment[0] + segment[1] * segment[1] + segment[2] * segment[2])
        if (vec == 0):
            vec = 0.001

        ax = 1.0 * RAD2DEG * np.arccos(segment[2] / vec)

        if (segment[2] < 0.0):
            ax = -ax

        rx = -segment[1] * segment[2]
        ry = segment[0] * segment[2]
        rz = segment[2] * segment[2]

        glPushMatrix()
        glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        """
        if(0):#indexID_1=="RightFoot" or indexID_1=="LeftFoot"):
            GL.glRotatef(ax,rx,ry,0.0)#rz)#0.0)
        else:
            quaternion_temp_=self.quaternion[self.indexID.index(indexID_1),:]
            quaternion_temp_[1]=-quaternion_temp_[1]
            GL.glMultMatrixf(m3d.QtoMatrix4x4(quaternion_temp_)
            
            #GL.glMultMatrixf(m3d.QtoMatrix4x4(self.quaternion[self.indexID.index(indexID_1),:]))
        """

        quaternion_temp_ = self.quaternion[self.indexID.index(indexID_1), :]
        # quaternion_temp_[0]=-quaternion_temp_[0]
        # quaternion_temp_[1]=-quaternion_temp_[1]
        quaternion_temp_[2] = -quaternion_temp_[2]
        GL.glMultMatrixf(QtoMatrix4x4(quaternion_temp_))

        GL.glRotated((VIEW_OFFSET_ROT[2]) * RAD2DEG, 0.0, 0.0, 1.0)
        GL.glRotated((VIEW_OFFSET_ROT[0]) * RAD2DEG, 1.0, 0.0, 0.0)
        GL.glRotated((VIEW_OFFSET_ROT[1]) * RAD2DEG, 0.0, 1.0, 0.0)

        GL.glCallList(self.index_displayList.index(indexID_1) + 1)  # self.indexID.index(indexID_1))

        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # displaylist
    # -------------------------------------------------------------------
    def displaylist(self):
        """
        displaylist
        """
        GL.glGenLists(15)  # self.indexID.index(linkNodeName))

        self.set_displayList_coordinate('Coordinate')

        linkColor_1 = [0.6, 0.6, 0.6]
        linkColor_2 = [0.1, 0.2, 0.1]
        self.set_displayList_floor_p('Floor', linkColor_1, linkColor_2)

        self.obj_model = OBJ_parser()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/HEAD_SIMPLE.obj"))
        linkColor = [0.8, 0.7, 0.7]
        self.set_displayList('Neck', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/BODY_L5.obj"))
        linkColor = [0.9, 0.9, 0.9]
        self.set_displayList('L5', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/BODY_T8.obj"))
        linkColor = [0.9, 0.9, 0.9]
        self.set_displayList('T8', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/HIP.obj"))
        linkColor = [0.0, 0.0, 1.0]
        self.set_displayList('Pelvis', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/R_ARM_UPPER.obj"))
        linkColor = [0.8, 0.7, 0.7]
        self.set_displayList('RightUpperArm', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/R_ARM_HAND_F.obj"))
        linkColor = [0.8, 0.7, 0.7]
        self.set_displayList('RightForeArm', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/L_ARM_UPPER.obj"))
        linkColor = [0.8, 0.7, 0.7]
        self.set_displayList('LeftUpperArm', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/L_ARM_HAND_F.obj"))
        linkColor = [0.8, 0.7, 0.7]
        self.set_displayList('LeftForeArm', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/RL_LEG_THIGH.obj"))
        linkColor = [0.0, 0.0, 1.0]
        self.set_displayList('RightUpperLeg', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/RL_LEG_THIGH.obj"))
        linkColor = [0.0, 0.0, 1.0]
        self.set_displayList('LeftUpperLeg', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/R_LEG_LOWER.obj"))
        linkColor = [0.0, 0.0, 1.0]
        self.set_displayList('RightLowerLeg', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/L_LEG_LOWER.obj"))
        linkColor = [0.0, 0.0, 1.0]
        self.set_displayList('LeftLowerLeg', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/R_LEG_FOOT.obj"))
        linkColor = [1.0, 1.0, 0.0]
        self.set_displayList('RightFoot', DL_model, linkColor)

        self.obj_model = OBJ_parser()
        DL_model, DL_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/L_LEG_FOOT.obj"))
        linkColor = [1.0, 1.0, 0.0]
        self.set_displayList('LeftFoot', DL_model, linkColor)

    # -------------------------------------------------------------------
    # set_displayList
    # -------------------------------------------------------------------
    def set_displayList(self, linkNodeName, link_triangles, link_color):

        self.index_displayList.append(linkNodeName)
        GL.glNewList(len(self.index_displayList), GL.GL_COMPILE_AND_EXECUTE)

        GL.glBegin(GL.GL_POLYGON)  # GL_POLYGON) GL_LINES
        GL.glColor(link_color[0], link_color[1], link_color[2])
        ambient = np.array([0.2, 0.2, 0.2, 1.0])
        diffuse = np.array([1.0, 0.8, 0.0, 1.0])
        specular = np.array([1.0, 1.0, 1.0, 1.0])
        shine = 100.0
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, ambient)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, diffuse)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, specular)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, shine)

        if (linkNodeName == "Neck"):
            offset_Z = 0.1
        elif (linkNodeName == "Pelvis"):
            offset_Z = -0.1
        else:
            offset_Z = 0

        for triangle in link_triangles:
            point1 = np.array([float(triangle[0][0]), float(triangle[0][1]), float(triangle[0][2]) + offset_Z])
            point2 = np.array([float(triangle[1][0]), float(triangle[1][1]), float(triangle[1][2]) + offset_Z])
            point3 = np.array([float(triangle[2][0]), float(triangle[2][1]), float(triangle[2][2]) + offset_Z])

            array1 = point2 - point1
            array2 = point3 - point2

            array3 = np.cross(array1, array2)
            length = np.linalg.norm(array3)

            # Norm vector for a triangle defined by 3 points
            GL.glNormal3f(float(array3[0] / length), float(array3[1] / length), float(array3[2] / length))
            GL.glVertex3f(float(point1[0]), float(point1[1]), float(point1[2]))
            GL.glVertex3f(float(point2[0]), float(point2[1]), float(point2[2]))
            GL.glVertex3f(float(point3[0]), float(point3[1]), float(point3[2]))

        GL.glEnd()

        GL.glEndList()
        print("[openGLModel]: Open DisplayList", self.indexID.index(linkNodeName), len(self.index_displayList))

        # -------------------------------------------------------------------

    # set_displayList
    # -------------------------------------------------------------------
    def set_displayList_floor_p(self, linkNodeName, link_color_1, link_color_2):

        self.index_displayList.append(linkNodeName)
        GL.glNewList(len(self.index_displayList), GL.GL_COMPILE_AND_EXECUTE)

        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, link_color_1)
        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)

        pos_min = -2
        tile_num = 5
        tile_length = 0.5
        tile_num = int(-pos_min * 2 / tile_length)
        color_start = 0

        for tile_x in range(0, tile_num):
            if (color_start == 0):
                color_start = 1
            else:
                color_start = 0

            for tile_y in range(0, tile_num):

                glBegin(GL.GL_POLYGON)

                if (color_start == 1):
                    glColor4f(link_color_1[0], link_color_1[1], link_color_1[2], 1.0)
                    color_start = 0
                else:
                    glColor4f(link_color_2[0], link_color_2[1], link_color_2[2], 1.0)
                    color_start = 1

                ambient = np.array([0.2, 0.2, 0.2, 1.0])
                diffuse = np.array([1.0, 0.8, 0.0, 1.0])
                specular = np.array([1.0, 1.0, 1.0, 1.0])
                shine = 100.0
                GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, ambient)
                GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, diffuse)
                GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, specular)
                GL.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, shine)

                GL.glNormal3f(0, 0, 1)

                point1 = np.array(
                    [float(pos_min + tile_length * (tile_x + 1)), float(pos_min + tile_length * (tile_y)), 0.0])
                point2 = np.array(
                    [float(pos_min + tile_length * (tile_x + 1)), float(pos_min + tile_length * (tile_y + 1)), 0.0])
                point3 = np.array(
                    [float(pos_min + tile_length * (tile_x)), float(pos_min + tile_length * (tile_y + 1)), 0.0])
                point4 = np.array(
                    [float(pos_min + tile_length * (tile_x)), float(pos_min + tile_length * (tile_y)), 0.0])

                array1 = point2 - point1
                array2 = point3 - point1
                array3 = np.cross(array1, array2)
                length = np.linalg.norm(array3)

                # GL.glNormal3f(float(array3[0]/length),float(array3[1]/length),float(array3[2]/length))
                glVertex3f(point1[0], point1[1], point1[2])
                glVertex3f(point2[0], point2[1], point2[2])
                glVertex3f(point3[0], point3[1], point3[2])
                glVertex3f(point4[0], point4[1], point4[2])

                GL.glEnd()

        GL.glEndList()

    # -------------------------------------------------------------------
    # set_displayList
    # -------------------------------------------------------------------
    def set_displayList_coordinate(self, linkNodeName):

        self.index_displayList.append(linkNodeName)
        GL.glNewList(len(self.index_displayList), GL.GL_COMPILE_AND_EXECUTE)

        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)

        color = np.array([1, 0, 0])
        GL.glPushMatrix();
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, color)
        self.draw_array(0.01, 0.6, 10, color)
        # GL.glPopMatrix();

        color = np.array([0, 1, 0])
        # GL.glPushMatrix();
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, color)
        GL.glRotated((-90.0), 1.0, 0.0, 0.0)
        self.draw_array(0.01, 0.6, 10, color)

        color = np.array([1, 1, 0])
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, color)
        GL.glRotated((90.0), 0.0, 1.0, 0.0)
        self.draw_array(0.01, 0.6, 10, color)
        GL.glPopMatrix();

        GL.glEndList()

    # -------------------------------------------------------------------
    # set_displayList
    # -------------------------------------------------------------------
    def set_displayList_GRF(self, linkNodeName):

        self.index_displayList.append(linkNodeName)
        GL.glNewList(len(self.index_displayList), GL.GL_COMPILE_AND_EXECUTE)

        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)

        if (linkNodeName == "GRF"):
            color = np.array([1, 0, 0])
        elif (linkNodeName == "GRF"):
            color = np.array([0, 1, 0])
        elif (linkNodeName == "GRF"):
            color = np.array([0, 0, 1])

        GL.glPushMatrix();
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, color)
        self.draw_array(0.01, 0.6, 10, color)

        GL.glPopMatrix();
        GL.glEndList()

    # -------------------------------------------------------------------
    # glDisplayLine
    # -------------------------------------------------------------------
    def display_GRF(self):

        force_max = 1000.0
        length = np.linalg.norm(self.GRF)
        force_vector = self.GRF * (length * force_max) + self.GRP

        color_ = np.array([1, 0, 0])
        self.displayLine(self.GRP, force_vector, color_)

    # -------------------------------------------------------------------
    # glDisplayLine
    # -------------------------------------------------------------------
    def displayLine(self, link_pos_1, link_pos_2, line_color):
        """
        glDisplayLine
        """
        link_pos_1 = rotateAndTrans(link_pos_1, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)
        link_pos_2 = rotateAndTrans(link_pos_2, VIEW_OFFSET_ROT, VIEW_OFFSET_POS)

        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glBegin(GL.GL_LINES)
        GL.glLineWidth(2.5)
        GL.glColor3f(line_color[0], line_color[1], line_color[2])
        GL.glVertex3f(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glVertex3f(link_pos_2[0], link_pos_2[1], link_pos_2[2])
        GL.glEnd()

    # -------------------------------------------------------------------
    # draw_cylinder
    # -------------------------------------------------------------------
    def draw_array(self, radius, height, num_slices, color):
        """
        draw_cylinder
        """

        glColor(color[0], color[1], color[2])
        cylinder = gluNewQuadric()
        gluQuadricNormals(cylinder, GLU_SMOOTH)
        gluCylinder(cylinder, radius, radius, height, num_slices, num_slices)
        # glutSolidCone(1.5,1,16,8)

    # -------------------------------------------------------------------
    # draw_cylinder
    # -------------------------------------------------------------------
    def draw_cylinder(self, radius, height, num_slices):
        """
        draw_cylinder
        """

        glColor(0, 0, 1)
        cylinder = gluNewQuadric()
        gluQuadricNormals(cylinder, GLU_SMOOTH)
        gluCylinder(cylinder, radius, radius, height, num_slices, num_slices)

    # -------------------------------------------------------------------
    # draw_sphere
    # -------------------------------------------------------------------
    def draw_sphere(self, radius, num_slices):
        """
        draw_sphere
        """

        # glColor(1, 0, 0)
        sphere = gluNewQuadric()
        gluQuadricNormals(sphere, GLU_SMOOTH)
        # gluCylinder(cylinder, radius, radius, height, num_slices, num_slices)
        gluSphere(sphere, radius, 20, 20)

    # -------------------------------------------------------------------
    # updateOffset
    # -------------------------------------------------------------------
    def updateOffset(self, center_offset):
        """
        updateOffset
        """
        self.center_pos_offset = center_offset

    # -------------------------------------------------------------------
    # updateOffset
    # -------------------------------------------------------------------
    def updateCOG(self, update_COG):
        """
        updateOffset
        """
        self.COG = update_COG

    # -------------------------------------------------------------------
    # updateCounter
    # -------------------------------------------------------------------
    def updateCounter(self, update_Counter):
        """
        updateCounter
        """
        self.Counter = update_Counter

# ==============================================================================
# End Of File
# ==============================================================================
