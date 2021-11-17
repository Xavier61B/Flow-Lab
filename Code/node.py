# Author @xavier_barneclo
# Automated image based contact angle measuring device code

class node:
        def __init__(self,last,coord):
            self.next = None
            self.last = last
            self.x = coord[0]
            self.y = coord[1]

        def get_next(self):
            return self.next

        def get_last(self):
            return self.last

        def set_next(self,n):
            self.next = n

        def get_x(self):
            return self.x

        def get_y(self):
            return self.y
