import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import math


# priority Queue
def perant(i):
    return int((i-1)/2)
def left(i):
    return 2*i+1
def right(i):
    return 2*i+2

class PQueue:
    def __init__(self):
        self.data = []
        self.size = 0
    


    def swap(self,i,j):
        temp = self.data[i]
        self.data[i] =self.data[j]
        self.data[j] = temp

    def heapify(self,i):
        if self.size==0:
            return
        smallest = i
        if left(i)<self.size and self.data[left(i)][0] < self.data[smallest][0]:
            smallest = left(i)
        if right(i)<self.size and self.data[right(i)][0] < self.data[smallest][0]:
            smallest = right(i)
        if smallest!=i:
            self.swap(i,smallest)
            self.heapify(smallest)
    
    def push(self,keyval):
        key,val = keyval
        self.data.append((key,val))
        i = self.size
        self.size+=1
        while i!=0 and key < self.data[perant(i)][0]:
            self.swap(i,perant(i))
            i = perant(i)
    def pop(self):
        val = self.data[0][1]
        self.swap(0,self.size-1)
        self.size-=1
        self.data.pop(len(self.data)-1)
        self.heapify(0)
        return val
    def update(self,val, key):
        i=0
        flag = False
        while i<self.size and self.data[i][1]!=val:
            i+=1
        if i==self.size:
            self.push((key,val))
        else:
            if self.data[i][0] > key:
                flag = True
                self.data[i] = (key,self.data[i][1])
                while i!=0 and key < self.data[perant(i)][0]:
                    self.swap(i,perant(i))
                    i = perant(i)
        return flag

    def isEmpty(self):
        return self.size==0
    

#maze class
class Maze:
    def __init__(self,size: int):
        
        self.size= size
        self.h_walls = np.array([True] * size*(size-1))
        self.h_walls = np.reshape(self.h_walls, (size-1,size))
        
        self.v_walls = np.array([True] * size*(size-1))
        self.v_walls = np.reshape(self.v_walls, (size,size-1))

        self.all_walls = [self.h_walls,self.v_walls]
        
        self.history = [[np.copy(self.all_walls[0]),np.copy(self.all_walls[1])]] 
        
        cells = np.array([False]* size**2)
        cells = np.reshape(cells, (size,size))

        start = rand.choice(range(size**2))
        row = int(start % size)
        col = int((start-row)/size)
        cells[row,col] = True
        cell_walls = [(0,-1,0),(0,0,0),(1,0,-1),(1,0,0)]
        list = set()
        for wall in cell_walls:
            if row+wall[1]>=0 and col+wall[2]>=0:
                list.add((0+wall[0],row+wall[1],col+wall[2]))

        while len(list)>0:
            n_wall = rand.sample(list,1)
            n_wall = n_wall[0]
            if cells[n_wall[1],n_wall[2]]==False:
                row,col = n_wall[1],n_wall[2]
                cells[row,col] = True
                self.all_walls[n_wall[0]][n_wall[1],n_wall[2]] = False
                self.history.append([np.copy(self.all_walls[0]),np.copy(self.all_walls[1])])
                for  wall in cell_walls:
                    if row+wall[1]>=0 and col+wall[2]>=0:
                        list.add((0+wall[0],row+wall[1],col+wall[2])) 
            elif n_wall[0]==0 and n_wall[1]+1 < size and cells[n_wall[1]+1,n_wall[2]]==False:
                row,col = n_wall[1]+1,n_wall[2]
                cells[row,col] = True
                self.all_walls[n_wall[0]][n_wall[1],n_wall[2]] = False
                self.history.append([np.copy(self.all_walls[0]),np.copy(self.all_walls[1])])
                for  wall in cell_walls:
                    if row+wall[1]>=0 and col+wall[2]>=0:
                        list.add((0+wall[0],row+wall[1],col+wall[2]))
            elif n_wall[0]==1 and n_wall[2]+1 < size and cells[n_wall[1],n_wall[2]+1]==False:
                row,col = n_wall[1],n_wall[2]+1
                cells[row,col] = True
                self.all_walls[n_wall[0]][n_wall[1],n_wall[2]] = False
                self.history.append([np.copy(self.all_walls[0]),np.copy(self.all_walls[1])])
                for  wall in cell_walls:
                    if row+wall[1]>=0 and col+wall[2]>=0:
                        list.add((0+wall[0],row+wall[1],col+wall[2])) 
            list.remove(n_wall)
       
        #choose start and end points - not in a wall and in a distance of at leest 2

        self.start = rand.choice(range(size**2))
        scol = int(self.start % self.size)
        srow = int((self.start-scol) / self.size)
       
        self.end = rand.choice(range(size**2))

        ecol = int(self.end % self.size)
        erow = int((self.end-ecol) / self.size)

        while abs(ecol - scol)+abs(erow - srow) <2:
            self.end = rand.choice(range(size**2))
        
            ecol = int(self.end % self.size)
            erow = int((self.end-ecol) / self.size)
        
    

    def draw_maze(self, ax):
        ax.clear()

        ax.axis([0,self.size,0,self.size])

        WALL_SIZE = 0.1
        for row in range(self.size-1):
            for col in range(self.size-1):
                ax.add_patch(patches.Rectangle((col+1-WALL_SIZE/2,row+1-WALL_SIZE/2), WALL_SIZE, WALL_SIZE, edgecolor = None, fc='b'))

        #vertical walls
        for row in range(size-1,-1,-1):
            for col in range(size-1):
                if self.all_walls[1][size-row-1,col]:
                    ax.add_patch(patches.Rectangle((col+1-WALL_SIZE/2,row+WALL_SIZE/2), WALL_SIZE, 1-WALL_SIZE, edgecolor = None, fc='b'))

        #horizontal walls
        for row in range(size,1,-1):
            for col in range(size):
                 if self.all_walls[0][size-row,col]:
                   ax.add_patch(patches.Rectangle((col+WALL_SIZE/2,row-1-WALL_SIZE/2), 1-WALL_SIZE, WALL_SIZE, edgecolor = None, fc='b'))
        
        scol = int(self.start % self.size)
        srow = int((self.start-scol) / self.size)

        ecol = int(self.end % self.size)
        erow = int((self.end-ecol) / self.size)
        ax.add_patch(patches.Circle((scol+0.5,self.size-srow-0.5), 0.2, edgecolor = None, fc='black'))
        ax.add_patch(patches.Circle((ecol+0.5,self.size-erow-0.5), 0.2, edgecolor = None, fc='g'))
      


    # a reguler A* returns a single solution
    # uses manhattan distance as huristic
    def A_star(self):
        def h(pos1, pos2):
            scol = int(pos1 % self.size)
            srow = int((pos1-scol) / self.size)

            ecol = int(pos2 % self.size)
            erow = int((pos2-scol) / self.size)

            return abs(scol - ecol)+abs(srow - erow)
    
        q = PQueue()
        explored = [False] * self.size**2
        path = [0] * self.size**2
        dist = [math.inf] * self.size**2

        dist[self.start] = 0

        scol = int(self.start % self.size)
        srow = int((self.start-scol) / self.size)

        ecol = int(self.end % self.size)
        erow = int((self.end-ecol) / self.size)

        q.push((h(self.start, self.end), self.start))

        while not q.isEmpty():
            pos = q.pop()
            col = int(pos % self.size)
            row = int((pos-col) / self.size)
            explored[pos] = True
            if pos==self.end:
                best = [self.end]
                while best[0]!=self.start:
                    best.insert(0,path[best[0]])
                return best
            if col > 0 and self.maze[row,col-1] and not explored[pos-1] and dist[pos-1] > dist[pos]+1:
                dist[pos-1] = dist[pos]+1
                q.update(pos-1,dist[pos-1] + h(pos-1, self.end))
                path[pos-1]= pos
            if col < self.size-1 and self.maze[row,col+1] and not explored[pos+1] and dist[pos+1] > dist[pos]+1:
                dist[pos+1] = dist[pos]+1
                q.update(pos+1,dist[pos+1] + h(pos+1, self.end))
                path[pos+1]= pos
            if row < self.size-1 and self.maze[row+1,col] and not explored[pos+self.size] and dist[pos+self.size] > dist[pos]+1:
                dist[pos+self.size] = dist[pos]+1
                q.update(pos+self.size,dist[pos+self.size] + h(pos+self.size, self.end))
                path[pos+self.size]= pos
            if row > 0 and self.maze[row-1,col] and not explored[pos-self.size] and dist[pos-self.size] > dist[pos]+1:
                dist[pos-self.size] = dist[pos]+1
                q.update(pos-self.size,dist[pos-self.size] + h(pos-self.size, self.end))
                path[pos-self.size]= pos
        a  = True
        return []

    # A* for animation, returns all paths checked along the way
    def ani_A_star(self):
        def h(pos1, pos2):
            scol = int(pos1 % self.size)
            srow = int((pos1-scol) / self.size)

            ecol = int(pos2 % self.size)
            erow = int((pos2-scol) / self.size)

            return abs(scol - ecol)+abs(srow - erow)
    
        q = PQueue()
        explored = [False] * self.size**2
        path = [0] * self.size**2
        dist = [math.inf] * self.size**2
        history = []
        dist[self.start] = 0

        scol = int(self.start % self.size)
        srow = int((self.start-scol) / self.size)

        ecol = int(self.end % self.size)
        erow = int((self.end-ecol) / self.size)

        q.push((h(self.start, self.end), self.start))

        while not q.isEmpty():
            pos = q.pop()

            ani_path = [pos]
            while ani_path[0]!=self.start:
                ani_path.insert(0,path[ani_path[0]])
            history.append(ani_path)
        
            col = int(pos % self.size)
            row = int((pos-col) / self.size)
            explored[pos] = True
            if pos==self.end:
                best = [self.end]
                while best[0]!=self.start:
                    best.insert(0,path[best[0]])
                history.append(best)
                return history
            if col > 0 and not self.all_walls[1][row,col-1] and not explored[pos-1] and dist[pos-1] > dist[pos]+1:
                dist[pos-1] = dist[pos]+1
                q.update(pos-1,dist[pos-1] + h(pos-1, self.end))
                path[pos-1]= pos
            if col < self.size-1 and not self.all_walls[1][row,col] and not explored[pos+1] and dist[pos+1] > dist[pos]+1:
                dist[pos+1] = dist[pos]+1
                q.update(pos+1,dist[pos+1] + h(pos+1, self.end))
                path[pos+1]= pos
            if row < self.size-1 and not self.all_walls[0][row,col] and not explored[pos+self.size] and dist[pos+self.size] > dist[pos]+1:
                dist[pos+self.size] = dist[pos]+1
                q.update(pos+self.size,dist[pos+self.size] + h(pos+self.size, self.end))
                path[pos+self.size]= pos
            if row > 0 and not self.all_walls[0][row-1,col] and not explored[pos-self.size] and dist[pos-self.size] > dist[pos]+1:
                dist[pos-self.size] = dist[pos]+1
                q.update(pos-self.size,dist[pos-self.size] + h(pos-self.size, self.end))
                path[pos-self.size]= pos
        a  = True
        return history

#main
#choose size of maze (maze is square size*size)
size = 40
    
m =  Maze(size)


paths = m.ani_A_star()

fig = plt.figure(figsize = (7.2,7.2))

ax = fig.add_subplot(111)
m.draw_maze(ax)
plt.show(block=False)

line, = ax.plot([], [], lw=3, color = 'red')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    list = paths[i % len(paths)] 
    x = [(pos % size) + 0.5 for pos in list]
    y = [size- ((pos - pos % size) / size) - 0.5 for pos in list]
    line.set_data(x, y)
    ax.set_title("Iteration number: {}\nTotal iterations: {}".format(i,len(paths)))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(paths), interval=20,repeat = False, blit=True)

anim.save('a_star.gif')
plt.show()


