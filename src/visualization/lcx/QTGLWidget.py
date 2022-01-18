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

import sys
import math

from OpenGL import GL, GLU
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt5 import Qt
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtOpenGL import *
from PyQt5 import QtOpenGL
from PyQt5.QtCore import QPointF, QRect, QRectF, Qt, QTimer

from PyQt5.QtWidgets import (QApplication, QWidget, QGraphicsView, QGraphicsScene,
                             QGraphicsItem, QGraphicsEllipseItem,
                             QGridLayout, QVBoxLayout, QHBoxLayout, QBoxLayout,
                             QInputDialog, QListWidget, QListWidgetItem, QCompleter,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollBar,
                             QRadioButton, qApp)

import pyqtgraph as pg
from pyqtgraph.Point import Point

import numpy as np
import csv

# from MVNX_parser import MVNX_parser
from .OBJ_parser import OBJ_parser
from .math3D import Qzero, QtoEulerAngle

# -------------------------------------------------------------------
win_width = 600
win_height = 700

PAI = 3.14159265
RAD2DEG = 180 / PAI
DEG2RAD = 1 / RAD2DEG
DELTA_T = 0.005

# light_ambient0 = [1.0, 0.5, 0.5, 1.0]
# light_diffuse0 = [1.0, 0.5, 0.5, 1.0]
# light_specular0 = [1.0, 0.5, 0.5, 1.0]
# light_position0 = [2.0, 2.0, 1.0, 1.0]

light_ambient0 = [0.2, 0.2, 0.2, 1.0]
light_diffuse0 = [0.8, 0.8, 0.8, 1.0]
light_specular0 = [0.0, 0.0, 0.0, 1.0]
light_position0 = [5.0, 5.0, -5.0, 0.0]

light_ambient1 = [0.2, 0.2, 0.2, 1.0]
light_diffuse1 = [0.8, 0.8, 0.8, 1.0]
light_specular1 = [0.0, 0.0, 0.0, 1.0]
light_position1 = [-5.0, 5.0, 5.0, 0.0]

POLYGON_MODEL_MODE = True
LIGHTING = True
TWO_LIGHTING = True


# ==============================================================================
# OpenGL Display
# ==============================================================================
class QTGLWidget(QGLWidget):
    xRotationChanged = QtCore.pyqtSignal(int)
    yRotationChanged = QtCore.pyqtSignal(int)
    zRotationChanged = QtCore.pyqtSignal(int)

    # -------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------
    def __init__(self, projectorMode, parent=None):
        super(QTGLWidget, self).__init__(parent)
        """
        __init__
        """

        self.obj_model = OBJ_parser()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.RFOOT_triangles, self.RFOOT_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/R_FOOT.obj"))
        # self.list_R_FOOT=self.create_glList(0, self.RFOOT_triangles, self.RFOOT_color)

        self.obj_model = OBJ_parser()
        self.LFOOT_triangles, self.LFOOT_color = self.obj_model.loadOBJ(os.path.join(file_dir, "MODEL/L_FOOT.obj"))
        # self.list_L_FOOT=self.create_glList(1, self.LFOOT_triangles, self.LFOOT_color)

        self.projector_mode = projectorMode

        self.object_1 = 0
        self.object_2 = 0
        self.xRot = 90
        self.yRot = 30
        self.zRot = 0
        self.zoom = 0.3

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        self.indexID = ['Pelvis', 'T8', 'Neck', 'Head',
                        'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
                        'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
                        'RightUpperArm', 'RightForeArm', 'RightHand',
                        'LeftUpperArm', 'LeftForeArm', 'LeftHand']
        self.link_node_num = 16

        self.position = np.zeros((self.link_node_num, 3))
        self.quaternion = np.zeros((self.link_node_num, 4))

        self.foot_point_R = np.array(
            [[0.176, -0.004, -0.1], [0.15, -0.03, -0.1], [0.1, -0.04, -0.1], [0.027, -0.03, -0.1],
             [-0.038, -0.03, -0.1], [-0.058, -0.025, -0.1], [-0.07, -0.009, -0.1], [-0.07, 0.011, -0.1],
             [-0.053, 0.025, -0.1], [-0.027, 0.034, -0.1], [0.027, 0.025, -0.1], [0.052, 0.026, -0.1],
             [0.093, 0.045, -0.1], [0.11, 0.048, -0.1], [0.16, 0.04, -0.1], [0.177, 0.025, -0.1]])

        self.foot_point_L = np.array(
            [[0.177, -0.025, -0.1], [0.16, -0.04, -0.1], [0.11, -0.048, -0.1], [0.093, -0.045, -0.1],
             [0.052, -0.026, -0.1], [0.027, -0.025, -0.1], [-0.027, -0.034, -0.1], [-0.053, -0.025, -0.1],
             [-0.07, -0.011, -0.1], [-0.07, 0.009, -0.1], [-0.058, 0.025, -0.1], [-0.038, 0.03, -0.1],
             [0.027, 0.03, -0.1], [0.1, 0.04, -0.1], [0.15, 0.03, -0.1], [0.176, 0.004, -0.1]])
        self.foot_pos_r = self.foot_point_R
        self.foot_pos_l = self.foot_point_L
        self.foot_q_r = Qzero()
        self.foot_q_l = Qzero()

    # -------------------------------------------------------------------
    # minimumSizeHint
    # -------------------------------------------------------------------
    def create_glList(self, linkNo, link_triangles, link_color):

        list_ID = GL.glGenLists(linkNo)
        GL.glNewList(list_ID, GL.GL_COMPILE_AND_EXECUTE)

        GL.glBegin(GL.GL_TRIANGLES)

        i = 0
        # print self.normal
        for triangle in link_triangles:
            # Set color
            GL.glColor3f(link_color[0], link_color[1], link_color[2])

            # Define 2 vectors from 3 points.
            point1 = np.array([triangle[0]])
            point2 = np.array([triangle[1]])
            point3 = np.array([triangle[2]])

            array1 = point2 - point1
            array2 = point3 - point1

            # Compute unit cross product 
            array3 = np.cross(array1, array2)
            length = np.linalg.norm(array3)
            array3 = array3 / length

            # Norm vector for a triangle defined by 3 points
            GL.glNormal3f(array3[0, 1], array3[0, 1], array3[0, 2])
            GL.glVertex3f(triangle[0][1], triangle[0][2], triangle[0][0])
            GL.glVertex3f(triangle[1][1], triangle[1][2], triangle[1][0])
            GL.glVertex3f(triangle[2][1], triangle[2][2], triangle[2][0])

            i += 1

        GL.glEnd()

        GL.glEndList()

        return list_ID

    # -------------------------------------------------------------------
    # minimumSizeHint
    # -------------------------------------------------------------------
    def draw_polygon(self, link_triangles, link_color):

        GL.glBegin(GL.GL_TRIANGLES)
        # print(link_triangles)

        i = 0
        # print self.normal
        GL.glColor4f(link_color[0], link_color[1], link_color[2], link_color[3])

        for triangle in link_triangles:
            # Set color
            # GL.glColor3f(link_color[0],link_color[1],link_color[2])

            # point1=np.array([triangle[1][:]])

            array1 = [float(triangle[1][0]) - float(triangle[0][0]), float(triangle[1][1]) - float(triangle[0][1]),
                      float(triangle[1][2]) - float(triangle[0][2])]
            array2 = [float(triangle[2][0]) - float(triangle[0][0]), float(triangle[2][1]) - float(triangle[0][1]),
                      float(triangle[2][2]) - float(triangle[0][2])]
            array3 = np.cross(array1, array2)
            length = np.linalg.norm(array3)
            array3 = array3 / length

            """
            # Define 2 vectors from 3 points.
            point1=np.array([triangle[0][0],triangle[0][1],triangle[0][2]])#np.array(triangle[0])
            point2=np.array([triangle[1][0],triangle[1][1],triangle[1][2]])#np.array(triangle[0])
            point3=np.array([triangle[2][0],triangle[2][1],triangle[2][2]])#np.array(triangle[0])
            
            print(point1)
            print(point2)
            print(point3)
            
            array1=np.subtract(point2,point1)
            array2=point3-point1
            
            # Compute unit cross product 
            array3 = np.cross(array1,array2)
            length = np.linalg.norm(array3)
            array3=array3/length
            """
            # Norm vector for a triangle defined by 3 points
            GL.glNormal3f(float(array3[1]), float(array3[2]), float(array3[0]))
            GL.glVertex3f(float(triangle[0][0]), float(triangle[0][1]), float(triangle[0][2]))
            GL.glVertex3f(float(triangle[1][0]), float(triangle[1][1]), float(triangle[1][2]))
            GL.glVertex3f(float(triangle[2][0]), float(triangle[2][1]), float(triangle[2][2]))
            # GL.glVertex3f(triangle[1][1],triangle[1][2],triangle[1][0])
            # GL.glVertex3f(triangle[2][1],triangle[2][2],triangle[2][0])

            i += 1

        GL.glEnd()

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

        angle = self.normalizeAngle(angle)
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

        angle = self.normalizeAngle(angle)
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

        angle = self.normalizeAngle(angle)
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
        self.updateGL()

    # -------------------------------------------------------------------
    # initializeGL
    # -------------------------------------------------------------------
    def initializeGL(self):
        """
        initializeGL
        """

        self.object_1 = self.makeObject(1)
        self.object_2 = self.makeObject(2)

        self.listNumber_1 = self.makeDisplayList('Pelvis', 'T8')
        self.listNumber_2 = self.makeDisplayList('T8', 'Neck')
        self.listNumber_3 = self.makeDisplayList('Neck', 'Head')

        self.listNumber_4 = self.makeDisplayList('Pelvis', 'RightUpperLeg')
        self.listNumber_5 = self.makeDisplayList('RightUpperLeg', 'RightLowerLeg')
        self.listNumber_6 = self.makeDisplayList('RightLowerLeg', 'RightFoot')

        self.listNumber_7 = self.makeDisplayList('Pelvis', 'LeftUpperLeg')
        self.listNumber_8 = self.makeDisplayList('LeftUpperLeg', 'LeftLowerLeg')
        self.listNumber_9 = self.makeDisplayList('LeftLowerLeg', 'LeftFoot')

        self.listNumber_10 = self.makeDisplayList('T8', 'RightUpperArm')
        self.listNumber_11 = self.makeDisplayList('RightUpperArm', 'RightForeArm')
        self.listNumber_12 = self.makeDisplayList('RightForeArm', 'RightHand')

        self.listNumber_13 = self.makeDisplayList('T8', 'LeftUpperArm')
        self.listNumber_14 = self.makeDisplayList('LeftUpperArm', 'LeftForeArm')
        self.listNumber_15 = self.makeDisplayList('LeftForeArm', 'LeftHand')
        self.link_num = 15

        GL.glClearColor(1.0, 1.0, 1.0, 0.0)

        GL.glShadeModel(GL.GL_FLAT)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # GL.glEnable(GL.GL_CULL_FACE)
        # GL.glFrontFace(GL.GL_CW)
        # GL.glCullFace(GL.GL_BACK)

        if (LIGHTING == True):
            GL.glEnable(GL.GL_LIGHTING)

            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, light_ambient0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, light_diffuse0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, light_specular0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position0)
            # GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glEnable(GL.GL_LIGHT0)

            if TWO_LIGHTING == True:
                GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, light_ambient1)
                GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, light_diffuse1)
                GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, light_specular1)
                GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, light_position1)
                GL.glEnable(GL.GL_LIGHT1)

    # -------------------------------------------------------------------
    # paintGL
    # -------------------------------------------------------------------
    def paintGL(self):
        """
        paintGL
        """

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        GL.glTranslated(0.0, 0.5, -5.0)
        GL.glRotated(90.0, 1.0, 0.0, 0.0)
        GL.glRotated(-self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(-self.zRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(-self.yRot / 16.0, 0.0, 0.0, 1.0)

        if (LIGHTING == True):
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position0)

            if TWO_LIGHTING == True:
                GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, light_position1)

        GL.glScalef(self.zoom, self.zoom, self.zoom)

        GL.glColor(0.8, 0.8, 1.0)
        self.glDisplayLine('Pelvis', 'T8')
        GL.glColor(0.8, 0.8, 1.0)
        self.glDisplayLine('T8', 'Neck')
        GL.glColor(0.8, 0.8, 1.0)
        self.glDisplayLine('Neck', 'Head')

        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('Pelvis', 'RightUpperLeg')
        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('RightUpperLeg', 'RightLowerLeg')
        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('RightLowerLeg', 'RightFoot')

        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('Pelvis', 'LeftUpperLeg')
        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('LeftUpperLeg', 'LeftLowerLeg')
        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('LeftLowerLeg', 'LeftFoot')

        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('T8', 'RightUpperArm')
        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('RightUpperArm', 'RightForeArm')
        GL.glColor(0.0, 0.8, 0.0)
        self.glDisplayLine('RightForeArm', 'RightHand')

        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('T8', 'LeftUpperArm')
        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('LeftUpperArm', 'LeftForeArm')
        GL.glColor(0.0, 0.0, 0.8)
        self.glDisplayLine('LeftForeArm', 'LeftHand')

        if (LIGHTING == True):
            GL.glColorMaterial(GL.GL_FRONT, GL.GL_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)

        self.glDisplayFoot()
        self.glDisplayHead()
        self.glDisplayHand()

        self.draw_floor()

        if (POLYGON_MODEL_MODE == True):
            # GL.glPushMatrix();
            GL.glColor(0.8, 0.8, 1.0)
            self.glDisplayPolygon('Pelvis', 'T8')
            GL.glColor(0.8, 0.8, 1.0)
            self.glDisplayLine('T8', 'Neck')
            GL.glColor(0.8, 0.8, 1.0)
            self.glDisplayPolygon('Neck', 'Head')

            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('Pelvis', 'RightUpperLeg')
            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('RightUpperLeg', 'RightLowerLeg')
            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('RightLowerLeg', 'RightFoot')

            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('Pelvis', 'LeftUpperLeg')
            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('LeftUpperLeg', 'LeftLowerLeg')
            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('LeftLowerLeg', 'LeftFoot')

            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('T8', 'RightUpperArm')
            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('RightUpperArm', 'RightForeArm')
            GL.glColor(0.0, 0.8, 0.0)
            self.glDisplayPolygon('RightForeArm', 'RightHand')

            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('T8', 'LeftUpperArm')
            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('LeftUpperArm', 'LeftForeArm')
            GL.glColor(0.0, 0.0, 0.8)
            self.glDisplayPolygon('LeftForeArm', 'LeftHand')

            self.RFOOT_color = [0.3, 0.0, 0.6, 0.5]
            self.displayPolygon('RightFoot', self.RFOOT_triangles, self.RFOOT_color)

            self.LFOOT_color = [0.3, 0.0, 0.6, 0.5]
            self.displayPolygon('LeftFoot', self.LFOOT_triangles, self.LFOOT_color)

            # self.glDisplayPolygonFoot()

    # -------------------------------------------------------------------
    # resizeGL
    # -------------------------------------------------------------------
    def resizeGL(self, width, height):
        """
        resizeGL
        """

        side = min(width, height)
        if side < 0:
            return

        if (self.projector_mode == True):
            GL.glClearColor(0.9, 0.9, 1.0, 0.0)

        GL.glViewport(0, 0, width, height)
        gluPerspective(45.0, (float)(width / height), 0.1, 100.0)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1, 1, 1, -0.5, 4.0, 15.0)  # (-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

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
    # makeObject
    # -------------------------------------------------------------------
    def makeObject(self, listNumber):
        """
        makeObject
        """

        genList = GL.glGenLists(listNumber)
        GL.glNewList(genList, GL.GL_COMPILE)
        GL.glPushMatrix();

        # if(listNumber==1):
        #    self.draw_Q()

        if (listNumber == 2):
            self.draw_cylinder(0.02, 0.2, 10)

        GL.glPopMatrix();
        GL.glEndList()

        return genList

    # -------------------------------------------------------------------
    # makeDisplayList
    # -------------------------------------------------------------------
    def makeDisplayList(self, indexID_1, indexID_2):
        """
        makeDisplayList
        """

        listNumber = self.indexID.index(indexID_2)
        genList = GL.glGenLists(listNumber)
        GL.glNewList(genList, GL.GL_COMPILE)

        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])
        link_gc = (link_pos_1 + link_pos_2) * 0.5
        # link_q    =np.array(self.quaternion[self.indexID.index(indexID_1),:])
        link_length = np.linalg.norm(link_pos_2 - link_pos_1)

        GL.glPushMatrix();
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glBegin(GL.GL_POLYGON)  # GL_TRIANGLE_FAN)#drawing the back circle
        # GL.glTranslatef(link_gc[0],link_gc[1],link_gc[2])
        self.draw_cylinder(0.02, link_length, 10)
        GL.glEnd()

        GL.glTranslatef(link_gc[0], link_gc[1], link_gc[2])

        GL.glPopAttrib()
        GL.glPopMatrix();
        GL.glEndList()
        return genList

    # -------------------------------------------------------------------
    # setDisplayList
    # -------------------------------------------------------------------
    def setDisplayList(self, listID, fileName):
        """
        setDisplayList
        """

        genList = GL.glGenLists(listID)
        GL.glNewList(genList, GL.GL_COMPILE)

        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])
        link_gc = (link_pos_1 + link_pos_2) * 0.5
        # link_q    =np.array(self.quaternion[self.indexID.index(indexID_1),:])
        link_length = np.linalg.norm(link_pos_2 - link_pos_1)

        GL.glPushMatrix();
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glBegin(GL.GL_POLYGON)  # GL_TRIANGLE_FAN)#drawing the back circle
        # GL.glTranslatef(link_gc[0],link_gc[1],link_gc[2])
        self.draw_cylinder(0.02, link_length, 10)
        GL.glEnd()

        GL.glTranslatef(link_gc[0], link_gc[1], link_gc[2])

        GL.glPopAttrib()
        GL.glPopMatrix();
        GL.glEndList()
        return genList

    # -------------------------------------------------------------------
    # glDisplayLine
    # -------------------------------------------------------------------
    def glDisplayLine(self, indexID_1, indexID_2):
        """
        glDisplayLine
        """
        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])

        GL.glLineWidth(2.5)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glVertex3f(link_pos_2[0], link_pos_2[1], link_pos_2[2])
        GL.glEnd()

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])
        GL.glColor(1.0, 0.0, 0.0)
        self.draw_sphere(0.02, 10)
        GL.glPopMatrix();

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_2[0], link_pos_2[1], link_pos_2[2])
        GL.glColor(1.0, 0.0, 0.0)
        self.draw_sphere(0.02, 10)
        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayPolygon
    # -------------------------------------------------------------------
    def glDisplayPolygon(self, indexID_1, indexID_2):
        """
        glDisplayPolygon
        """

        self.qglColor(self.trolltechGreen)
        link_pos_1 = np.array(self.position[self.indexID.index(indexID_1), :])
        link_pos_2 = np.array(self.position[self.indexID.index(indexID_2), :])
        # link_gc   = (link_pos_1+link_pos_2)*0.5
        # trans=link_pos_2-link_pos_1
        link_q = np.array(self.quaternion[self.indexID.index(indexID_1), :])
        link_length = np.linalg.norm(link_pos_2 - link_pos_1)
        rotX, rotY, rotZ = QtoEulerAngle(link_q)

        GL.glPushMatrix();
        GL.glRotated(rotZ * RAD2DEG, 1.0, 0.0, 0.0)
        GL.glRotated(rotY * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated(rotX * RAD2DEG, 0.0, 0.0, 1.0)

        GL.glPopMatrix();

        """
        GL.glPushMatrix();
        
        GL.glTranslatef(link_pos_1[0],link_pos_1[1],link_pos_1[2])
        self.draw_sphere(0.02,10)    
        #self.draw_cylinder(0.06,link_length,10)
        """
        segment = link_pos_1 - link_pos_2

        if (segment[2] == 0):
            segment[2] = 0.001

        # vec=segment.norm()
        vec = np.sqrt(segment[0] * segment[0] + segment[1] * segment[1] + segment[2] * segment[2])
        ax = 57.2957795 * np.arccos(segment[2] / vec)

        if (segment[2] < 0.0):
            ax = -ax

        rx = -segment[1] * segment[2]
        ry = segment[0] * segment[2]

        glPushMatrix()
        glTranslatef(link_pos_2[0], link_pos_2[1], link_pos_2[2])
        glRotatef(ax, rx, ry, 0.0)

        glColor(0.6, 0.6, 1)
        cylinder = gluNewQuadric()
        # gluQuadricNormals(cylinder, GLU_SMOOTH)

        radius = 0.05
        num_slices = 10

        gluQuadricOrientation(cylinder, GLU_OUTSIDE)
        gluCylinder(cylinder, radius, radius, vec, num_slices, num_slices)

        glColor(1, 1, 0)
        gluQuadricOrientation(cylinder, GLU_INSIDE)
        self.draw_sphere(radius + 0.01, 10)
        GL.glTranslatef(0, 0, vec)

        gluQuadricOrientation(cylinder, GLU_OUTSIDE)
        self.draw_sphere(radius + 0.01, 10)
        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayHead
    # -------------------------------------------------------------------
    def glDisplayHead(self):
        """
        glDisplayHead
        """

        link_pos_1 = np.array(self.position[self.indexID.index('Head'), :])

        GL.glLineWidth(4.0)

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        link_q = np.array(self.quaternion[self.indexID.index('Head'), :])
        rotX, rotY, rotZ = QtoEulerAngle(link_q)
        GL.glRotated(rotY * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated(rotZ * RAD2DEG, 0.0, 0.0, 1.0)
        GL.glRotated(rotX * RAD2DEG, 1.0, 0.0, 0.0)

        GL.glColor(0.8, 0.8, 1.0)
        self.draw_sphere(0.1, 10)

        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayHand
    # -------------------------------------------------------------------
    def glDisplayHand(self):
        """
        glDisplayHand
        """

        link_pos_1 = np.array(self.position[self.indexID.index('RightHand'), :])
        link_pos_2 = np.array(self.position[self.indexID.index('LeftHand'), :])
        GL.glLineWidth(4.0)

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        link_q = np.array(self.quaternion[self.indexID.index('RightHand'), :])
        rotX, rotY, rotZ = QtoEulerAngle(link_q)
        GL.glRotated(rotY * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated(rotZ * RAD2DEG, 0.0, 0.0, 1.0)
        GL.glRotated(rotX * RAD2DEG, 1.0, 0.0, 0.0)

        GL.glColor(0.0, 0.8, 0.0)
        self.draw_sphere(0.05, 10)

        GL.glPopMatrix();

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_2[0], link_pos_2[1], link_pos_2[2])

        link_q = np.array(self.quaternion[self.indexID.index('LeftHand'), :])
        rotX, rotY, rotZ = QtoEulerAngle(link_q)
        GL.glRotated(rotY * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated(rotZ * RAD2DEG, 0.0, 0.0, 1.0)
        GL.glRotated(rotX * RAD2DEG, 1.0, 0.0, 0.0)

        GL.glColor(0.0, 0.0, 0.8)
        self.draw_sphere(0.05, 10)

        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # glDisplayFoot
    # -------------------------------------------------------------------
    def glDisplayFoot(self):
        """
        glDisplayFoot
        """

        self.link_pos_r = np.array(self.position[self.indexID.index('RightFoot'), :])
        self.link_pos_l = np.array(self.position[self.indexID.index('LeftFoot'), :])

        if (POLYGON_MODEL_MODE == False):
            GL.glLineWidth(4.0)

            GL.glPushMatrix();
            GL.glTranslatef(self.link_pos_r[0], self.link_pos_r[1], self.link_pos_r[2])
            GL.glBegin(GL.GL_POLYGON)
            GL.glColor(0.8, 0, 0)
            for i in range(0, len(self.foot_pos_r)):
                GL.glVertex3f(self.foot_pos_r[i, 0], self.foot_pos_r[i, 1], self.foot_pos_r[i, 2])
            GL.glEnd()
            GL.glPopMatrix();

            GL.glPushMatrix();
            GL.glTranslatef(self.link_pos_l[0], self.link_pos_l[1], self.link_pos_l[2])
            GL.glBegin(GL.GL_POLYGON)
            GL.glColor(0.8, 0, 0)
            for i in range(0, len(self.foot_point_L)):
                GL.glVertex3f(self.foot_pos_l[i, 0], self.foot_pos_l[i, 1], self.foot_pos_l[i, 2])
            GL.glEnd()
            GL.glPopMatrix();

        if (self.link_pos_r[2] < self.link_pos_l[2]):
            self.link_pos_z = self.link_pos_r[2] - 0.11
        else:
            self.link_pos_z = self.link_pos_l[2] - 0.11

        self.link_pos_z = 0.01

        self.floor_level = self.link_pos_z - 0.005

        GL.glPushMatrix();
        GL.glTranslatef(self.link_pos_r[0], self.link_pos_r[1], self.link_pos_z)  # link_pos_r[2])
        GL.glBegin(GL.GL_POLYGON)
        GL.glColor(0.4, 0.4, 0.4)
        for i in range(0, len(self.foot_pos_r)):
            GL.glVertex3f(self.foot_pos_r[i, 0], self.foot_pos_r[i, 1], self.link_pos_z)
        GL.glEnd()
        GL.glPopMatrix();

        GL.glPushMatrix();
        GL.glTranslatef(self.link_pos_l[0], self.link_pos_l[1], self.link_pos_z)  # link_pos_l[2])
        GL.glBegin(GL.GL_POLYGON)
        GL.glColor(0.4, 0.4, 0.4)
        for i in range(0, len(self.foot_point_L)):
            GL.glVertex3f(self.foot_pos_l[i, 0], self.foot_pos_l[i, 1], self.link_pos_z)
        GL.glEnd()
        GL.glPopMatrix()

        self.draw_floor()

    # -------------------------------------------------------------------
    # glDisplayPolygon
    # -------------------------------------------------------------------
    def displayPolygon(self, linkNodeName, linkTriangles, linkColor):
        """
        glDisplayPolygonFoot
        """

        link_pos_1 = np.array(self.position[self.indexID.index(linkNodeName), :])

        GL.glPushMatrix();
        GL.glTranslatef(link_pos_1[0], link_pos_1[1], link_pos_1[2])

        link_q = np.array(self.quaternion[self.indexID.index(linkNodeName), :])
        rotX, rotY, rotZ = QtoEulerAngle(link_q)
        GL.glRotated(rotY * RAD2DEG, 0.0, 1.0, 0.0)
        GL.glRotated(rotZ * RAD2DEG, 0.0, 0.0, 1.0)
        GL.glRotated(rotX * RAD2DEG, 1.0, 0.0, 0.0)

        self.draw_polygon(linkTriangles, linkColor)

        GL.glPopMatrix();

    # -------------------------------------------------------------------
    # normalizeAngle
    # -------------------------------------------------------------------
    def draw_floor(self):

        GL.glPushMatrix()
        GL.glBegin(GL.GL_POLYGON)
        GL.glColor(0.3, 1.0, 0.3)

        floor_height_0 = -0.01
        floor_height_1 = 0.01

        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_r[0] - 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)

        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)

        GL.glVertex3f(self.link_pos_r[0] - 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_r[0] - 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)

        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)

        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_0)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_0)

        GL.glVertex3f(self.link_pos_r[0] + 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] + 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_l[0] - 1, self.link_pos_l[1] + 1, self.floor_level - floor_height_1)
        GL.glVertex3f(self.link_pos_r[0] - 1, self.link_pos_r[1] - 1, self.floor_level - floor_height_1)

        GL.glEnd()
        GL.glPopMatrix()

    # -------------------------------------------------------------------
    # normalizeAngle
    # -------------------------------------------------------------------
    def normalizeAngle(self, angle):
        """
        normalizeAngle
        """

        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

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
        gluSphere(sphere, radius, 10, 10)

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
    # updateFootData
    # -------------------------------------------------------------------
    def updateFootData(self, nodeID, position, quaternion):
        """
        updateFootData
        """

        if (nodeID == 'RightFoot'):
            self.foot_pos_r = position
            self.foot_q_r = quaternion

        elif (nodeID == 'LeftFoot'):
            self.foot_pos_l = position
            self.foot_q_l = quaternion

# ==============================================================================
# End Of File
# ==============================================================================
