from __future__ import division
import math
import operator


#Paste this in Train_Test.py bottom to test
# myknn = knn.KNN(5)
# myknn.fit(X.toarray(),y)
# KNN_pred = myknn.predict(Y.toarray())
# predicted_categories = le.inverse_transform(KNN_pred)
# print classification_report(y_test, KNN_pred, target_names=list(le.classes_))

def euclideanDistance(array1, array2):
  length = len(array1)
  distance = 0
  for x in range(length):
    distance += pow((array1[x] - array2[x]), 2)
  return math.sqrt(distance)

class KNN:
  def __init__(self, k):
    self.k = k;

  def score(self, testSet, testSetLabels):
    predictedLabels = self.predict(testSet)
    count = 0
    size = len(predictedLabels)
    for i in range(size):
      if predictedLabels[i] == testSetLabels[i]:
        count = count + 1

    return count / size


  #trainSet must be and array
  def fit(self, trainSet, trainSetLabels):
    self.trainSet = trainSet
    self.trainSetLabels = trainSetLabels

  #testSet must be and array
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
      distance = euclideanDistance(trainSet[i], featuresArray)
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

      