from dist_functions import *

def test(pointsNum=10):
  points = [[i,i,i] for i in range(pointsNum)]
  ids = [i for i in range(len(points))]
  k = len(points[0])
  distFunction = euclideanDistance
  compareFunction = float_compare
  tree = KDTree(k,distFunction,compareFunction,points,ids)
  tree.printAll()

def testNearest(pointsNum=50, target=[25,25,25], neighbours=10):
  points = [[i,i,i] for i in range(pointsNum)]
  ids = [i for i in range(pointsNum)]
  k = len(points[0])
  distFunction = euclideanDistance
  compareFunction = float_compare
  tree = KDTree(k,distFunction,compareFunction,points,ids)
  print "Finding " + str(neighbours) + " nearest neighbours"
  nbrs = tree.find_nearest_neigbours(target, neighbours)
  for n in nbrs:
    print n,
  print ""

class KDNode:
  def __init__(self, id, data, axis, left, right):
    #Keep the id to keep track
    #of the corresponding label
    self.id = id
    self.data = data
    self.axis = axis
    self.left = left
    self.right = right


class KDTree:
  def __init__(self, k, distFunction, compareFunction, points, ids):
    self.k = k

    #to compare points in nn search
    self.distFunction = distFunction

    #to compare the values of a specific dimension
    self.compareFunction = compareFunction

    #append id of each point as first element of the point
    size = len(points)
    newPoints = [] #create a list joining points with ids
    for i in range(size):
      newPoints.append([points[i],ids[i]])

    self.root = self.generate(newPoints)

  def generate(self, points, depth = 0):
    if points == []:
      return None

    #Pick the axis used to split points
    #at this level/depth of the tree
    axis = depth % self.k

    #Sort the points by comparing selected axis
    points.sort(cmp=lambda x, y: self.compareFunction(x[0][axis], y[0][axis]))

    #Pick the middle point to serve as parent
    median = len(points) / 2
    parent = KDNode(points[median][1], points[median][0], axis, None, None)

    parent.left = self.generate(points[0:median], depth + 1)
    parent.right = self.generate(points[median + 1:], depth + 1)

    return parent

  def update_nearest(self, nearest, node, target, numNearest):
    dist = self.distFunction(node.data, target)
    currentNearest = len(nearest)

    if (currentNearest < numNearest or dist < nearest[-1][0]):
      if (currentNearest >= numNearest):
        nearest.pop()

      nearest.append([dist, node.id])
      nearest.sort(cmp=lambda x,y: self.compareFunction(x[0],y[0]))

    return nearest

  def find_nearest_neigbours(self, target, numNearest):
    if (self.root == None):
      return []
    else:
      return self.nearest_neighbours([], self.root, target, numNearest, 0)

  def nearest_neighbours(self, nearest, node, target, numNearest, depth):
    axis = depth % self.k

    #if there are no children
    #just check if the parent is one of the nearest
    if node.left == None and node.right == None:
      nearest = self.update_nearest(nearest, node, target, numNearest)
      return nearest

    #check which of the children is (propably) nearer
    if node.right == None or (node.left and target[axis] <= node.data[axis]):
        nearer = node.left
        further = node.right
    else:
        nearer = node.right
        further = node.left

    nearest = self.nearest_neighbours(nearest, nearer, target, numNearest, depth+1)

    if further:
        #if we still have more near neighbours to find
        #or it is possible to find a neighbour that is nearer than
        #the less nearest neigbour in our list
        if len(nearest) < numNearest or self.maxDistance(target[axis], node.data[axis]) < nearest[-1][0]:
            nearest = self.nearest_neighbours(nearest, further, target, numNearest, depth+1)

    nearest = self.update_nearest(nearest, node, target, numNearest)
    return nearest

  def maxDistance(self, axisValue1, axisValue2):
    point1 = [axisValue1 for i in range(self.k)]
    point2 = [axisValue2 for i in range(self.k)]
    return self.distFunction(point1, point2)

  def insert(self, point, id):
    if (self.root == None):
      axis = 1
      self.root = KDNode(id, point, axis, None, None) 
    else:
      self.insertPoint(self.root, point, id)

  def insertPoint(self, parent, point, id, depth = 0):
    axis = parent.axis
    compare = self.compareFunction(point[axis], parent.data[axis])
    if (compare < 0):
      if parent.left == None:
        newaxis = (depth + 1) % self.k
        parent.left = KDNode(id, point, newaxis, None, None) 
      else:
        self.insertPoint(parent.left, point, id, depth + 1)
    else:
      if parent.right == None:
        newaxis = (depth + 1) % self.k + 1
        parent.right = KDNode(id, point, newaxis, None, None) 
      else:
        self.insertPoint(parent.right, point, id, depth + 1)


  def printAll(self):
    self.printNode(self.root)

  def printNode(self, node, depth = 0):
    if node == None:
      return

    indent = ""
    for i in range(depth):
      indent += " "
    print indent, node.data, "(Axis: " + str(node.axis) + ")"
    if (node.left != None):
      print indent, "Left"
      self.printNode(node.left, depth + 1)
    if (node.right != None):
      print indent, "Right"
      self.printNode(node.right, depth + 1)







