from matplotlib import pyplot as plt 
import numpy as np 
from PIL import Image, ImageDraw

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def find_path(init_p, goal_p, size_of_map, max_length, max_steps):

    plt.figure()
    plt.title("Rapidly-exploring random tree")
    plt.scatter(init_p.x, init_p.y)
    plt.scatter(goal_p.x, goal_p.y, c='r')
    plt.axis([0, size_of_map.x, 0, size_of_map.y])
    plt.ion()

    tree = np.array([[0,0,init_p.x,init_p.y]])            # order: 0 - my number, 1 - my parent, 2 - my x, 3 - my y
    nr_of_parent = 0
    max_distance = np.sqrt(size_of_map.x**2 + size_of_map.y**2)
    

    for i in range(1,max_steps):
        shot = Point(size_of_map.x*np.random.rand(), size_of_map.y*np.random.rand())
        dot = plt.scatter(shot.x, shot.y ,c='y')

        distance_to_parent = max_distance
        for k in range(len(tree)):
            distance_to_vertex = np.sqrt((tree[k,2] - shot.x)**2 + (tree[k,3]-shot.y)**2)
            if  distance_to_vertex < distance_to_parent:
                nr_of_parent = k
                distance_to_parent = distance_to_vertex

        if distance_to_parent > max_length:
            parent = Point(tree[nr_of_parent,2], tree[nr_of_parent,3])
            sin_to_parent = (shot.y - parent.y)/distance_to_parent
            cos_to_parent = (shot.x - parent.x)/distance_to_parent
            next_vertex = Point(parent.x + max_length*cos_to_parent, parent.y + max_length*sin_to_parent)
        else:
            next_vertex = Point(shot.x, shot.y)
        
        tree = np.vstack((tree, [i,nr_of_parent, next_vertex.x, next_vertex.y]))

        if i%100 == 0:
            print(f"{i} iterations are done already ")

        plt.plot([tree[nr_of_parent,2],tree[i,2]],[tree[nr_of_parent,3],tree[i,3]], 'b')
        plt.pause(0.005)
        dot.remove()
        distance_to_goal = np.sqrt((goal_p.x - next_vertex.x)**2 + (goal_p.y-next_vertex.y)**2)
        if distance_to_goal <= max_length:
            tree = np.vstack((tree, [i+1,i, goal_p.x, goal_p.y]))
            break
    else:
        print ("Path is not found. I'm sorry man :( ")
        return [],[]

    final_tree=np.array([tree[-1]])
    n = 0
    for i in reversed(range(len(tree)-1)):
        if tree[i,0] == final_tree[n,1]:
            final_tree = np.vstack((final_tree, tree[i]))
            plt.plot([final_tree[n,2],final_tree[n+1,2]],[final_tree[n,3],final_tree[n+1,3]], 'g')
            n = n + 1
    

    final_tree = final_tree[::-1]
    tree_x = final_tree[:,2]
    tree_y = final_tree[:,3]

    print('Path is found :)')
    plt.show(block=False)
    
    return tree_x, tree_y
