#Oscar Paredez 19109

import struct
import random
from collections import namedtuple
from obj import Obj

# =======================================================
# FUNCIONES MATEMATICAS

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

def sum(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element sum
    """
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element substraction
    """
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element multiplication
    """  
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Scalar with the dot product
    """
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cross(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the cross product
    """  
    return V3(
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x,
    )

def length(v0):
    """
    Input: 1 size 3 vector
    Output: Scalar with the length of the vector
    """  
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
    """
    Input: 1 size 3 vector
    Output: Size 3 vector with the normal of the vector
    """  
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

def bbox(*vertices):
    """
        Input: n size 2 vectors
        Output: 2 size 2 vectors defining the smallest bounding rectangle possible
    """  
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]
    xs.sort()
    ys.sort()

    return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

def barycentric(A, B, C, P):
    """
    Input: 3 size 2 vectors and a point
    Output: 3 barycentric coordinates of the point in relation to the triangle formed
            * returns -1, -1, -1 for degenerate triangles
    """  
    bary = cross(
        V3(C.x - A.x, B.x - A.x, A.x - P.x), 
        V3(C.y - A.y, B.y - A.y, A.y - P.y)
    )

    if abs(bary[2]) < 1:
        return -1, -1, -1   # this triangle is degenerate, return anything outside

    return (
        1 - (bary[0] + bary[1]) / bary[2], 
        bary[1] / bary[2], 
        bary[0] / bary[2]
    )

def glInit(width, height):
    return Renderer(width, height)

def char(c):
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    # short 
    return struct.pack('=h', w)

def dword(w):
    # long
    return struct.pack('=l', w)

def color(r, g, b):
    if r >=0 and r <=1 and g >=0 and g <=1 and b >=0 and b <=1:
        return bytes([round(b*255), round(g*255), round(r*255)])

BLACK = color(0, 0, 0)

class Renderer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def glCreateWindow(self, width, height):
        self.framebuffer = [[BLACK for x in range(self.width)] for y in range(self.height)]
        self.zbuffer = [[-float('inf') for x in range(self.width)]for y in range(self.height)]

    def glViewPort(self, x, y, width, height):
        self.initialViewPortX = x
        self.initialViewPortY = y
        self.viewPortX = width
        self.viewPortY = height

    def glClearColor(self, r, g, b):
        self.current_color = color(r, g, b)

    def glClear(self):
        self.framebuffer = [[self.current_color for x in range(self.width)] for y in range(self.height)]

    def glColor(self, r, g, b):
        self.vertex_color = color(r, g, b)

    def glVertex(self, x, y, color=None):
        posX = int((x+1)*(self.viewPortX/2)+self.initialViewPortX)
        posY = int((y+1)*(self.viewPortY/2)+self.initialViewPortY)
        self.framebuffer[posY][posX] = color

    def point(self, x, y, color = None):
        # 0,0 was intentionally left in the bottom left corner to mimic opengl
        try:
            self.framebuffer[y][x] = color
        except:
            #To avoid index out of range exceptions
            pass
    
    def line(self, x0, y0, x1, y1):
        # y = mx + b
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

            dy = abs(y1 - y0)
            dx = abs(x1 - x0)
        
        if x1 < x0:
            t1, t2 = x0, y0
            x0, y0 = x1, y1
            x1, y1 = t1, t2           

        offset = 0 * 2 * dx
        threshold = 0.5 * 2 * dx

        y = y0
        x = x0

        points = []
        while x <= x1:
            if steep:
                points.append((y, x))
            else:
                points.append((x, y))
                
            offset += (dy) * 2
            if offset >= threshold:
                y += 0.001 if y0 < y1 else -0.001
                threshold += 1 * 2 * dx
            x += 0.001    
        for point in points:
            # if point[0] <= 1 and point[0] >= -1 and point[1] <= 1 and point[1] >= -1:
                # self.glVertex(*point)
            # print(int((point[0]+1)*(self.viewPortX/2)+self.initialViewPortX), int((point[1]+1)*(self.viewPortY/2)+self.initialViewPortY))
            # self.glVertex(int((point[0]+1)*(self.viewPortX/2)+self.initialViewPortX), int((point[1]+1)*(self.viewPortY/2)+self.initialViewPortY))
            self.glVertex(((point[0]-self.initialViewPortX)*(2/self.viewPortX)-1), ((point[1]-self.initialViewPortY)*(2/self.viewPortY)-1))

    def draw_polygon(self, filename, scale):
        with open(filename) as f:
            lines = f.read().splitlines()
            for i in range(len(lines)):
                x0, y0 = lines[i % len(lines)].split(', ')
                x1, y1 = lines[(i + 1) % len(lines)].split(', ')
                self.line(int(x0)*scale[0], int(y0)*scale[1], int(x1)*scale[0], int(y1)*scale[1])

    def fill_polygon(self, filename):
        
        #Algoritmo que traza lineas verticales de izquierda a derecha para rellenar una figura
        verticesX = []
        verticesY = []
        insideX = []
        with open(filename) as f:
            lines = f.read().splitlines()
            for i in range(len(lines)):
                x, y = lines[i % len(lines)].split(', ')
                verticesX.append(int(x))
                verticesY.append(int(y))

            xmin, xmax, ymin, ymax = min(verticesX), max(verticesX), min(verticesY), max(verticesY)   
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    if self.framebuffer[y][x] == self.vertex_color:
                        # print(y)
                        insideX.append(x)
                for posX in range(insideX[0], insideX[-1]):
                    self.framebuffer[y][posX] = self.vertex_color
                insideX = []
            insideX = []

    def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
        # returns a vertex 3, translated and transformed
        return V3(
            (vertex[0] + translate[0]) * scale[0],
            (vertex[1] + translate[1]) * scale[1],
            (vertex[2] + translate[2]) * scale[2]
        )

    def load(self, filename, translate, scale):
        model = Obj(filename)

        light = V3(0,0,1)
    
        for face in model.faces:
            vcount = len(face)
            
            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.transform(model.vertices[f1], translate, scale)
                b = self.transform(model.vertices[f2], translate, scale)
                c = self.transform(model.vertices[f3], translate, scale)

                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)
                grey = intensity
                if grey <= 0:
                    continue
                self.triangle(a, b, c, color(grey, grey, grey))
            else:
                # assuming 4
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1   

                vertices = [
                    self.transform(model.vertices[f1], translate, scale),
                    self.transform(model.vertices[f2], translate, scale),
                    self.transform(model.vertices[f3], translate, scale),
                    self.transform(model.vertices[f4], translate, scale)
                ]

                normal = norm(cross(sub(vertices[0], vertices[1]), sub(vertices[1], vertices[2])))
                intensity = dot(normal, light)
                grey = intensity
                if grey <= 0:
                    continue
        
                A, B, C, D = vertices 
                
                self.triangle(A, B, C, color(grey, grey, grey))
                self.triangle(A, C, D, color(grey, grey, grey)) 


    def triangle(self, A, B, C, color=None):
        bbox_min, bbox_max = bbox(A, B, C)
        for x in range(int(bbox_min.x), int(bbox_max.x + 1)):
            for y in range(int(bbox_min.y), int(bbox_max.y + 1)):
                P = V2(x, y)
                w, v, u = barycentric(A, B, C, P)
                if w < 0 or v < 0 or u < 0:
                    continue
                z = A.z * w + B.z * v + C.z * u

                if z > self.zbuffer[x][y]:
                    #self.glVertex(((x-self.initialViewPortX)*(2/self.viewPortX)-1), ((y-self.initialViewPortY)*(2/self.viewPortY)-1), color)
                    self.point(x, y, color)
                    self.zbuffer[x][y] = z

    def write(self, filename):
        f = open(filename, 'bw')
        # file header
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14+40+3*(self.width*self.height)))
        f.write(dword(0))
        f.write(dword(14+40))
        # info header
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(3*(self.width*self.height)))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        # bitmap
        for y in range(self.height):
            for x in range(self.width):
                f.write(self.framebuffer[y][x])
        f.close()

    def glFinish(self):
        self.write('a.bmp')

r = glInit(2000, 2000)
r.glCreateWindow(2000, 2000)
r.glViewPort(100, 100, 1800, 1800)
r.glClearColor(0, 0, 0)
r.glClear()
r.glColor(1, 1, 1)


####################################
r.load('./models/face.obj', (25, 15, 15), (40, 40, 40))
####################################


#r.load('./models/ferrari.obj', [0, 0], [0.5, 0.5])
#r.draw_polygon('./polygons/polygon1.txt', [1, 1])
#r.fill_polygon('./polygons/polygon1.txt')
#r.glColor(0.2, 0.2, 1)
#r.draw_polygon('./polygons/polygon2.txt', [1, 1])
#r.fill_polygon('./polygons/polygon2.txt')
#r.glColor(0, 1, 1)
#r.draw_polygon('./polygons/polygon3.txt', [1, 1])
#r.fill_polygon('./polygons/polygon3.txt')
#r.glColor(0.8, 0.2, 1)
#r.draw_polygon('./polygons/polygon4.txt', [1, 1])
#r.fill_polygon('./polygons/polygon4.txt')
#r.glColor(0, 0, 0)
#r.draw_polygon('./polygons/polygon5.txt', [1, 1])
#r.fill_polygon('./polygons/polygon5.txt')

r.glFinish()
