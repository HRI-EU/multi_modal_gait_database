# -*- coding: utf-8 -*-
"""
#==============================================================================
File name : OBJ_parser.py
#==============================================================================

[History]
2018.03.14 : Taizo Yoshikawa : Original framework was created.

[Description]
GUI software to (1)load .obj data files. 

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

import os
import numpy as np
import csv

from PyQt5.QtCore import (QLineF, QPointF, QRectF, Qt, QTimer)
from PyQt5.QtGui import (QBrush, QColor, QPainter, QIntValidator)
from PyQt5.QtWidgets import (QApplication, QWidget)

#==============================================================================
# OBJ_parser
#==============================================================================
class OBJ_parser(QWidget):
    """
    This is a function to parse Xsense OBJ data.
    """

    def __init__(self, parent=None):
        super(OBJ_parser, self).__init__(parent) 
        
        print("[OBJ]: OBJ parser : connected")           
                        
    #-------------------------------------------------------------------
    # Open file
    #-------------------------------------------------------------------
    def loadOBJ(self,fliePath):
        
        num_vertices = 0
        numuvs = 0
        numnormals = 0
        numFaces = 0
    
        #print fliePath
    
        self.vertices = []
        self.uvs = []
        self.normals = []
        vertex_colors = []
        face_vert_IDs = []
        uvIDs = []
        normal_IDs = []
        triangles = []
        
        for line in open(fliePath, "r"):
            vals = line.split()
    
            if len(vals) == 0:
                continue
    
            if vals[0] == "v":
                #v = vals[1:4]
                v = map(float, vals[1:4])
                v=[vals[1],vals[2],vals[3]]
                self.vertices.append(v)
    
                if len(vals) == 7:
                    #vc = map(float, vals[4:7])
                    vc = [vals[4],vals[5],vals[6]]
                    vertex_colors.append(vc)
    
                num_vertices += 1
            if vals[0] == "vt":
                #vt = map(float, vals[1:3])
                vt = [vals[1],vals[2]]
                self.uvs.append(vt)
                numuvs += 1
            if vals[0] == "vn":
                #vn = map(float, vals[1:4])
                vn = [vals[1],vals[2],vals[3]]
                self.normals.append(vn)
                numnormals += 1
            if vals[0] == "f":
                fvID = []
                uvID = []
                nvID = []
                for f in vals[1:]:
                    w = f.split("/")
    
                    if num_vertices > 0:
                        fvID.append(int(w[0])-1)
    
                    if numuvs > 0:
                        uvID.append(int(w[1])-1)
    
                    if numnormals > 0:
                        nvID.append(int(w[2])-1)
    
                vertex1 = self.vertices[int(vals[1].split("/")[0])-1]
                vertex2 = self.vertices[int(vals[2].split("/")[0])-1]
                vertex3 = self.vertices[int(vals[3].split("/")[0])-1]
                
                triangles.append((vertex1,vertex2,vertex3))

                face_vert_IDs.append(fvID)
                uvIDs.append(uvID)
                normal_IDs.append(nvID)
    
                numFaces += 1
                
        self.triangles = triangles
        
        #print(self.triangles)

        return self.triangles, vertex_colors

#==============================================================================
# End Of File
#==============================================================================

    


