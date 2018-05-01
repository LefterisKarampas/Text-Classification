from __future__ import division
import math
import operator
from kdtree import *
from dist_functions import *

class KNN:
  def __init__(self, k, distFunction = euclideanDistance):
    self.k = k
    self.distFunction = distFunction

  def score(self, testSet, testSetLabels):
    predictedLabels = self.predict(testSet)
    count = 0
    size = len(predictedLabels)
    for i in range(size):
      if predictedLabels[i] == testSetLabels[i]:
        count = count + 1

    return count / size


  def fit(self, trainSet, trainSetLabels):
    dims = len(trainSet[0])
    self.trainSetKDTree = KDTree(dims, self.distFunction, float_compare,
     trainSet, trainSetLabels)


  def predict(self, testSet):
    resultLabels = []

    for featuresArray in testSet:
      kneighbors = self.getNeighbors(featuresArray)
      label = self.majorityVote(kneighbors)
      resultLabels.append(label)

    return resultLabels


  def getNeighbors(self, featuresArray):
    kneighbors = self.trainSetKDTree.find_nearest_neigbours(featuresArray, self.k)

    return kneighbors


  def majorityVote(self, kneighbors):
    votes = {}

    for neighbor in kneighbors:
      label = neighbor[1]
      if label in votes:
        votes[label] += 1
      else:
        votes[label] = 1

    sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

      