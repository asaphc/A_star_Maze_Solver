import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib import animation
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
            if data[i][0] > key:
                flag = True
                data[i][0] = key
                while i!=0 and key < self.data[perant(i)][0]:
                    self.swap(i,perant(i))
                    i = perant(i)
        return flag

    def isEmpty(self):
        return self.size==0
    

#maze class
class Maze:
    def __init__(self,size: int):

        
        
        maze = np.array([False]* size**2)
        self.maze = np.reshape(maze,(size,size))

        

        #create random maze using randomized prim algorithm
        
        start = rand.choice(range(size**2))
        list = set()
        col = int(start % size)
        row = int((start -col)/size)
        self.maze[row,col] = True
        if col > 0 and not self.maze[row,col-1]:
            list.add(row*size+col-1)
        if col < size-1 and not self.maze[row,col+1]:
            list.add(row*size+col+1)
        if row < size-1 and not self.maze[row+1,col]:
            list.add((row+1)*size+col)
        if row > 0 and not self.maze[row-1,col]:
            list.add((row-1)*size+col)
        while len(list)>0:
            cell = rand.sample(list,1)
            cell = cell[0]
            list.remove(cell)
            count = 0
            col = int(cell % size)
            row = int((cell - col)/size)
            if col > 0 and self.maze[row,col-1]:
                count+=1
            if col < size-1 and self.maze[row,col+1]:
                count+=1
            if row < size-1 and self.maze[row+1,col]:
                count+=1
            if row > 0 and self.maze[row-1,col]:
                count+=1
            if count==1:
                self.maze[row,col] = True
                if col > 0 and not self.maze[row,col-1]:
                    list.add(row*size+col-1)
                if col < size-1 and not self.maze[row,col+1]:
                    list.add(row*size+col+1)
                if row < size-1 and not self.maze[row+1,col]:
                    list.add((row+1)*size+col)
                if row > 0 and not self.maze[row-1,col]:
                    list.add((row-1)*size+col)

        #choose start and end points - not in a wall and in a distance of at leest 2

        self.start = rand.choice(range(size**2))
        scol = int(self.start % len(self.maze[0]))
        srow = int((self.start-scol) / len(self.maze[0]))
        while not self.maze[srow,scol]:
            self.start = rand.choice(range(size**2))
            scol = int(self.start % len(self.maze[0]))
            srow = int((self.start-scol) / len(self.maze[0]))



        self.end = rand.choice(range(size**2))

        ecol = int(self.end % len(self.maze[0]))
        erow = int((self.end-ecol) / len(self.maze[0]))

        while abs(ecol - scol)+abs(erow - srow) <2 or not self.maze[erow,ecol]:
            self.end = rand.choice(range(size**2))
        
            ecol = int(self.end % len(self.maze[0]))
            erow = int((self.end-ecol) / len(self.maze[0]))
        
    

    def draw_maze(self, ax):
        ax.clear()

        ax.axis([0,len(self.maze[0]),0,len(self.maze)])
    
        for row in range(len(self.maze)):
            for col in range(len(self.maze[0])):
                if not self.maze[row,col]:
                    ax.add_patch(plt.Rectangle((row, col), 1, 1))

        scol = int(self.start % len(self.maze[0]))
        srow = int((self.start-scol) / len(self.maze[0]))

        ecol = int(self.end % len(self.maze[0]))
        erow = int((self.end-ecol) / len(self.maze[0]))

        ax.add_patch(plt.Rectangle((srow, scol), 1, 1, color = 'red'))
        ax.add_patch(plt.Rectangle((erow, ecol), 1, 1, color = 'black'))


    # a reguler A* returns a single solution
    # uses manhattan distance as huristic
    def A_star(self):
        def h(pos1, pos2):
            scol = int(pos1 % len(self.maze[0]))
            srow = int((pos1-scol) / len(self.maze[0]))

            ecol = int(pos2 % len(self.maze[0]))
            erow = int((pos2-scol) / len(self.maze[0]))

            return abs(scol - ecol)+abs(srow - erow)
    
        q = PQueue()
        explored = [False] * len(self.maze[0])**2
        path = [0] * len(self.maze[0])**2
        dist = [math.inf] * len(self.maze[0])**2

        dist[self.start] = 0

        scol = int(self.start % len(self.maze[0]))
        srow = int((self.start-scol) / len(self.maze[0]))

        ecol = int(self.end % len(self.maze[0]))
        erow = int((self.end-ecol) / len(self.maze[0]))

        q.push((h(self.start, self.end), self.start))

        while not q.isEmpty():
            pos = q.pop()
            col = int(pos % len(self.maze[0]))
            row = int((pos-col) / len(self.maze[0]))
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
            if col < len(self.maze[0])-1 and self.maze[row,col+1] and not explored[pos+1] and dist[pos+1] > dist[pos]+1:
                dist[pos+1] = dist[pos]+1
                q.update(pos+1,dist[pos+1] + h(pos+1, self.end))
                path[pos+1]= pos
            if row < len(self.maze)-1 and self.maze[row+1,col] and not explored[pos+len(self.maze[0])] and dist[pos+len(self.maze[0])] > dist[pos]+1:
                dist[pos+len(self.maze[0])] = dist[pos]+1
                q.update(pos+len(self.maze[0]),dist[pos+len(self.maze[0])] + h(pos+len(self.maze[0]), self.end))
                path[pos+len(self.maze[0])]= pos
            if row > 0 and self.maze[row-1,col] and not explored[pos-len(self.maze[0])] and dist[pos-len(self.maze[0])] > dist[pos]+1:
                dist[pos-len(self.maze[0])] = dist[pos]+1
                q.update(pos-len(self.maze[0]),dist[pos-len(self.maze[0])] + h(pos-len(self.maze[0]), self.end))
                path[pos-len(self.maze[0])]= pos
        a  = True
        return []

    # A* for animation, returns all paths checked along the way
    def ani_A_star(self):
        def h(pos1, pos2):
            scol = int(pos1 % len(self.maze[0]))
            srow = int((pos1-scol) / len(self.maze[0]))

            ecol = int(pos2 % len(self.maze[0]))
            erow = int((pos2-scol) / len(self.maze[0]))

            return abs(scol - ecol)+abs(srow - erow)
    
        q = PQueue()
        explored = [False] * len(self.maze[0])**2
        path = [0] * len(self.maze[0])**2
        dist = [math.inf] * len(self.maze[0])**2
        history = []
        dist[self.start] = 0

        scol = int(self.start % len(self.maze[0]))
        srow = int((self.start-scol) / len(self.maze[0]))

        ecol = int(self.end % len(self.maze[0]))
        erow = int((self.end-ecol) / len(self.maze[0]))

        q.push((h(self.start, self.end), self.start))

        while not q.isEmpty():
            pos = q.pop()

            ani_path = [pos]
            while ani_path[0]!=self.start:
                ani_path.insert(0,path[ani_path[0]])
            history.append(ani_path)
        
            col = int(pos % len(self.maze[0]))
            row = int((pos-col) / len(self.maze[0]))
            explored[pos] = True
            if pos==self.end:
                best = [self.end]
                while best[0]!=self.start:
                    best.insert(0,path[best[0]])
                history.append(best)
                return history
            if col > 0 and self.maze[row,col-1] and not explored[pos-1] and dist[pos-1] > dist[pos]+1:
                dist[pos-1] = dist[pos]+1
                q.update(pos-1,dist[pos-1] + h(pos-1, self.end))
                path[pos-1]= pos
            if col < len(self.maze[0])-1 and self.maze[row,col+1] and not explored[pos+1] and dist[pos+1] > dist[pos]+1:
                dist[pos+1] = dist[pos]+1
                q.update(pos+1,dist[pos+1] + h(pos+1, self.end))
                path[pos+1]= pos
            if row < len(self.maze)-1 and self.maze[row+1,col] and not explored[pos+len(self.maze[0])] and dist[pos+len(self.maze[0])] > dist[pos]+1:
                dist[pos+len(self.maze[0])] = dist[pos]+1
                q.update(pos+len(self.maze[0]),dist[pos+len(self.maze[0])] + h(pos+len(self.maze[0]), self.end))
                path[pos+len(self.maze[0])]= pos
            if row > 0 and self.maze[row-1,col] and not explored[pos-len(self.maze[0])] and dist[pos-len(self.maze[0])] > dist[pos]+1:
                dist[pos-len(self.maze[0])] = dist[pos]+1
                q.update(pos-len(self.maze[0]),dist[pos-len(self.maze[0])] + h(pos-len(self.maze[0]), self.end))
                path[pos-len(self.maze[0])]= pos
        a  = True
        return history

#main
#choose size of maze (maze is square size*size)
size = 20
    
m =  Maze(size)


paths = m.ani_A_star()

fig = plt.figure(figsize = (10.2,7.2))

ax = fig.add_subplot(111)
m.draw_maze(ax)
plt.show(block=False)

line, = ax.plot([], [], lw=3, color = 'green')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    list = paths[i % len(paths)] 
    y = [pos % size + 0.5 for pos in list]
    x = [(pos - pos % size) / size + 0.5 for pos in list]
    line.set_data(x, y)
    ax.set_title("Iteration number: {}\nTotal iterations: {}".format(i,len(paths)))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(paths), interval=20,repeat = False, blit=True)

#anim.save('a_star.gif')
plt.show()


