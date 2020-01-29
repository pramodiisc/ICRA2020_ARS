"""
A collection of functions to make it easy to deal with all the coordinate frames, points and vectors we come across in robotics
"""
import numpy as np 
from numpy import cos, sin 
class Graph():
    def __init__(self):
        self.nodes = {}
        self.ids = {}
        self.adjacency_matrix = np.zeros([100,100])
        self.id = 0

    def add_node(self, name):
        self.nodes[name] = self.id
        self.ids[self.id] = name
        self.id = self.id+1
    
    def add_connection(self, n1, n2):
        self.adjacency_matrix[self.nodes[n1],self.nodes[n2]] = 1
        self.adjacency_matrix[self.nodes[n2],self.nodes[n1]] = 1
    
    def get_shortest_path(self, n1, n2):
        """ Since all nodes in the graph have equal weights, breadth first search is optimal, no need of Djikstra"""
        nodes_visited = []
        current_node = self.nodes[n1]
        current_branch_nodes = [current_node]
        while True:
            current_node = current_branch_nodes.pop(0)
            for i in np.arange(0, self.id):
                if(self.adjacency_matrix[current_node,i] == 0):
                    continue
                if(self.adjacency_matrix[current_node,i] == 1):
                    next_node_dict =  next((node for node in nodes_visited if node['id'] == i), False)
                    if(next_node_dict == False):
                        nodes_visited.append({'id': i, 'prev_node_id': current_node})
                        current_branch_nodes.append(i)
            if self.nodes[n2] in current_branch_nodes:
                break
            if current_branch_nodes == []:
                return False
        path = []
        current_node = self.nodes[n2]
        path.append(n2)
        while(True):
            node_dict = next((node for node in nodes_visited if node['id'] == current_node), False)
            current_node = node_dict['prev_node_id']
            path.append(self.ids[current_node])
            if(current_node == self.nodes[n1]):
                break
        path.reverse()
        return path

    
    def get_adjacency_matrix(self):
        return self.adjacency_matrix[0:self.id, 0:self.id]

class TransformManager():
    def __init__(self):
        self.frames = []
        self.connections = []
        self.transforms={}
        self.G = Graph()
        self.id = 0
    def add_transform(self, from_frame, to_frame, transformation_matrix):
        if from_frame not in self.frames:
            self.frames.append(from_frame)
            self.G.add_node(from_frame)
        if to_frame not in self.frames:
            self.frames.append(to_frame)
            self.G.add_node(to_frame)
        self.G.add_connection(from_frame, to_frame)
        self.transforms[(from_frame, to_frame)] = transformation_matrix
        self.transforms[(to_frame, from_frame)] = np.linalg.inv(transformation_matrix)

    def get_transform(self, from_frame, to_frame):
        path = self.G.get_shortest_path(from_frame, to_frame)
        if path is False:
            print("No valid path exists between given transformations")
            return None
        trans_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        for i in np.arange(len(path) - 1):
            trans_matrix = self.transforms[(path[i], path[i+1])]@trans_matrix
        return trans_matrix
    def rotateEuler(self,axis, angle):
        """Returns a numpy array corresponding to transformation matrix about axis by angle radians.
        Axis is a string 'X', 'Y', 'Z' and angle is in radians """
        if(axis == 'Z'):
            return np.array([[cos(angle), -sin(angle),0,0],[sin(angle), cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
        if(axis == 'Y'):
            return np.array([[cos(angle),0,sin(angle),0],[0,1,0,0],[-sin(angle),0,cos(angle),0],[0,0,0,1]])
        if(axis == 'X'):
            return np.array([[1,0,0,0],[0,cos(angle), -sin(angle),0],[0,sin(angle), cos(angle),0],[0,0,0,1]])

    def translateEuler(self,trans):
        """ Returns a numpy array corresponding to translating by the input given"""
        return np.array([[1,0,0,trans[0]],[0,1,0,trans[1]],[0,0,1,trans[2]],[0,0,0,1]])
    def transform_point(self, point, to_frame):
        pass
    def transform_vector(self, vector, to_frame):
        pass    

class Point():
    def __init__(self, frame, transform_manager, x=0, y=0, z=0):
        self.frame = frame
        self.tf = transform_manager
        self.x = x
        self.y = y
        self.z = z
    
    def __call__(self, frame=0):
        if(frame == 0):
            return np.array([self.x, self.y, self.z])
        else:
            return (self.tf.get_transform(self.frame, frame)@np.array([self.x, self.y, self.z,1]))[:3]

class Vector():
    def __init__(self, frame, transform_manager, x=0, y=0, z=0):
        self.frame = frame
        self.tf = transform_manager
        self.x = x
        self.y = y
        self.z = z
    
    def __call__(self, frame=0):
        if(frame == 0):
            return np.array([self.x, self.y, self.z])
        else:
            return self.tf.get_transform(self.frame, frame)[:3, :3]@np.array([self.x, self.y, self.z])


