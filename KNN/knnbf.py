from __future__ import division
import math
import operator
from dist_fucntions import *

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
    self.trainSet = trainSet
    self.trainSetLabels = trainSetLabels


  def predict(self, testSet):
    resultLabels = []

    for featuresArray in testSet:
      kneighbors = self.getNeighbors(featuresArray)
      label = self.majorityVote(kneighbors)
      resultLabels.append(label)

    return resultLabels


  def getNeighbors(self, featuresArray):
    distances = []

    trainSet = self.trainSet
    trainSize = len(trainSet)

    #for each element in out train set
    for i in range(trainSize):
      distance = self.distFunction(trainSet[i], featuresArray)
      indexAndDist = (i,distance)
      distances.append(indexAndDist)

    #sort according to dist
    distances.sort(key=operator.itemgetter(1))

    kneighbors = []

    for i in range(self.k):
      #only keep neighbor and his label
      index = distances[i][0] #contains the index of the neighbor
      label = self.trainSetLabels[index]
      neighbor = (trainSet[index],label)
      kneighbors.append(neighbor)

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

      