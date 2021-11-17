# Author @xavier_barneclo
# Automated image based contact angle measuring device code

from node import node
import numpy as np

class list:
    def __init__(self,start):
        self.start = node(None,[np.Inf,np.Inf])
        self.last = node(self.start,start)
        self.start.set_next(self.last)
        self.size = 1

    def get_size(self):
        return self.size

    def get_start(self):
        return self.start.get_next()

    def find(self,node):
        s = self.start.get_next()
        best = s
        while s:
            ideal = [best.get_x(),best.get_y()]
            temp = [s.get_x(),s.get_y()]
            if self.distance(ideal,node) > self.distance(temp,node):
                best = s
            s = s.get_next()
        return best

    def get_list(self,ini,n):
        x = []
        y = []
        for i in range(n):
            x.append(ini.get_x())
            y.append(ini.get_y())
            ini = ini.get_next()
        return [x,y]

    def get_list_back(self,ini,n):
        x = []
        y = []
        for i in range(n):
            x.append(ini.get_x())
            y.append(ini.get_y())
            ini = ini.get_last()
        return [x,y]

    def distance(self,n1,n2):
        return abs(n1[0]-n2[0]) + abs(n1[1]-n2[1])

    def find_min(self,list,ref):
        min = list[0]
        for i in list:
            if self.distance(min,ref) > self.distance(i,ref):
                min = i
        return min

    def contains(self,point):
        s = self.start.get_next()
        while s:
            s_x = s.get_x()
            s_y = s.get_y()
            s = s.get_next()
            if s_x == point[0] and s_y == point[1]:
                return True
        return False
            
    
    def add(self,img,color):
        l = self.last
        col = l.x
        row = l.y

        max_r = img.shape[0]
        max_c = img.shape[1]

        pot = []

        for c in range(col-1,col+2):
            for r in range(row-1,row+2):
                if (c >= 0 and c < max_c) and (r >= 0 and r < max_r) and (c != col or r != row):
                    if img[r][c] == color and not self.contains([c,r]):
                        pot.append([c,r])

        if pot:
            new = self.find_min(pot,[col,row])
            new = node(l,new)
            l.set_next(new)

            self.last = new
            self.size += 1

            return new
        else:
            return None
