import math

def float_compare(x,y):
  if x < y:
    return -1
  if x == y:
    return 0
  if x > y:
    return 1

def euclideanDistance(array1, array2):
  length = len(array1)
  distance = 0
  for x in range(length):
    distance += pow((array1[x] - array2[x]), 2)
  return math.sqrt(distance)