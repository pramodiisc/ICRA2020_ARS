"""
A collection of functions to make it easy to deal with all the coordinate frames, points and vectors we come across in robotics
"""
import numpy as np 
from numpy import cos, sin 
PI = np.pi
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
    
    
    def transform_matrix_from_line_segments(self, ls11,ls12,LS11,LS12):
        """ Returns a 4*4 transformation matrix (numpy array) that when multiplied with line segments in the 
        initial frame, leads to line segments in the target frame as given.
        Inputs:
        ls11 : numpy array of shape (1,3) = Line Segment 1 in initial frames initial point
        ls12 : numpy array of shape (1,3) = Line Segment 1 in initial frames final point
        LS11 : numpy array of shape (1,3) = Line Segment 1 in final frames initial point
        LS12 : numpy array of shape (1,3) = Line Segment 1 in final frames final point
          """
        norm = lambda vec: (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
        vec1 = ls12 - ls11
        vec2 = LS12 - LS11
        trans_to_origin = self.translateEuler(-ls11)
        temp = self.rotate_matrix_from_vectors(vec1, vec2)
        rot_matrix = np.zeros([4,4])
        rot_matrix[:-1, :-1] = temp
        rot_matrix[3,3] = 1
        scale_matrix = np.eye(4)*norm(vec2)/norm(vec1)            
        scale_matrix[3,3] = 1
        trans_to_point = self.translateEuler(LS11)
        return trans_to_point@scale_matrix@rot_matrix@trans_to_origin
    
    def rotate_matrix_from_vectors(self, vec1, vec2):
        """Returns a *Rotation matrix that rotates vec1 to vec2 (Thus magnitude of vectors dont matter), about the axis defined cross(vec1, vec2) 
        Uses Rodriguez Formula in its implementation"""
        norm = lambda vec: (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
        nvec1 = vec1/norm(vec1)
        nvec2 = vec2/norm(vec2)
        if(norm(nvec1 - nvec2)<= 0.001):
            return np.eye(3)
        if(norm(nvec1 + nvec2)<= 0.001):
            return np.eye(3)*-1
        cross = lambda v1, v2 : np.array([v1[1]*v2[2] - v2[1]*v1[2], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]) 
        dot = lambda v1,v2: v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
        skew_symmetric = lambda vec: np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])
        costh = dot(nvec1, nvec2)
        axis = cross(nvec1, nvec2)
        AXIS = skew_symmetric(axis)
        sinth = (axis[0]**2+ axis[1]**2 + axis[2]**2)**0.5
        rot = np.eye(3) + AXIS + ((1-costh)/(sinth**2))*(AXIS@AXIS)
        return rot    

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
def transform_points(transf_matrix, points):
    """
    Transforms a list of points by the given transformation matrix
    Inputs:
    transf_matrix: numpy (4*4) array
    points: list of numpy (1*3) arrays
    Returns:
    transformed_points: list of numpy (1*3) arrays
    """
    temp_pts = [np.array([x[0],x[1],x[2],1]) for x in points]
    newpts = []
    for pt in temp_pts:
        newpts.append((transf_matrix@pt)[:3])
    return newpts
    pass
if(__name__ == "__main__"):
    
    # ls11 = np.array([0,0,0])
    # ls12 = np.array([1,0,0])
    # LS11 = np.array([0,1,0])
    # LS12 = np.array([0,0,0])
    # tf = TransformManager()
    # # print(tf.transform_matrix_from_line_segments(ls11, ls12, LS11, LS12)@np.array([1,0,0,1]))
    # norm = lambda vec: (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5

    # isCorrect = True
    # for i in np.arange(10000):
    #     p1 = np.random.randn(3)
    #     p2 = np.random.randn(3)
    #     p3 = np.random.randn(3)
    #     p4 = np.random.randn(3)
    #     # p4 = p3 + (norm(p2 - p1)/norm(p4-p3))*(p4-p3)
    #     mat = tf.transform_matrix_from_line_segments(p1, p2, p3, p4)
    #     _p1 = np.array([p1[0], p1[1], p1[2], 1])
    #     _p2 = np.array([p2[0], p2[1], p2[2], 1])
    #     _p3 = np.array([p3[0], p3[1], p3[2], 1])
    #     _p4 = np.array([p4[0], p4[1], p4[2], 1])
    #     v1 = mat@_p1 - _p3
    #     v2 = mat@_p2 - _p4
    #     if(norm(v1)+norm(v2) >= 0.001):
    #         print(v1)
    #         isCorrect = False

    # if(isCorrect):
    #     print("passed rot+ trans test")
    # isCorrect = True
    # for i in np.arange(100000):
    #     v1 = np.random.randn(3)
    #     v2 = -v1
    #     norm = lambda vec: (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
    #     nv1 = v1/norm(v1)
    #     nv2 = v2/norm(v2)
    #     mat = tf.rotate_matrix_from_vectors(nv1,nv2)
    #     if(norm(mat@nv1 - nv2)>=  0.01):
    #         isCorrect = False
    
    # if(isCorrect):
    #     print("Passed rotation test")
    pts = [np.array([1,0,0]), np.array([0,-1,0]), np.array([0,0,1])]
    tf = TransformManager()
    new_pts = transform_points(tf.translateEuler(np.array([1,1,1])), pts)
    print(new_pts)