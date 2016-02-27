# Calculating cross entropy of a text
# Program expects first argument to be filename of the text file to be examined.

### @@@TODO

# @AUTHOR: Ondřej Švec

import sys
import random
import operator
import functools
from math import log

TEST_DATA_COUNT = 20000  # number of words for test data
HELDOUT_DATA_COUNT = 40000  # number of words for heldout data

data = [line.strip() for line in open(sys.argv[1], 'r', encoding="iso-8859-2")]

testData = data[-TEST_DATA_COUNT:]  # get testData from data
del data[-TEST_DATA_COUNT:]  # remove them from data

heldoutData = data[-HELDOUT_DATA_COUNT:]  # get heldout data from data
del data[-HELDOUT_DATA_COUNT:]  # remove them from data

# getting statistics from the data:
trainingDataStats = {}
heldoutDataStats = {}
testDataStats = {}

for w in data:
    trainingDataStats[w] = 1 if w not in trainingDataStats else trainingDataStats[w] + 1
for w in heldoutData:
    heldoutDataStats[w] = 1 if w not in heldoutDataStats else heldoutDataStats[w] + 1
for w in testData:
    testDataStats[w] = 1 if w not in testDataStats else testDataStats[w] + 1
print('Training data stats:')
print('Words in text: ', len(data))
print('Unique words: ', len(trainingDataStats))
print('Frequency of most common: ', max(trainingDataStats.items(), key=operator.itemgetter(1))[1])
print('Number of words with 1 occurence: ',
      functools.reduce(lambda count, x: count + (x == 1), trainingDataStats.values(), 0))
print()

print('Heldout data stats:')
print('Words in text: ', len(heldoutData))
print('Unique words: ', len(heldoutDataStats))
print('Frequency of most common: ', max(heldoutDataStats.items(), key=operator.itemgetter(1))[1])
print('Number of words with 1 occurence: ',
      functools.reduce(lambda count, x: count + (x == 1), heldoutDataStats.values(), 0))
print('Coverage of unique heldout data: ',
      sum([1 if w in trainingDataStats else 0 for w in heldoutDataStats.keys()]) / len(heldoutDataStats.keys()))
print('Coverage of heldout data: ',
      sum([1 if w in trainingDataStats else 0 for w in heldoutData]) / HELDOUT_DATA_COUNT)
print()

print('Test data stats:')
print('Words in text: ', len(testData))
print('Unique words: ', len(testDataStats))
print('Frequency of most common: ', max(testDataStats.items(), key=operator.itemgetter(1))[1])
print('Number of words with 1 occurence: ',
      functools.reduce(lambda count, x: count + (x == 1), testDataStats.values(), 0))
print('Coverage of unique test data: ',
      sum([1 if w in trainingDataStats else 0 for w in testDataStats.keys()]) / len(testDataStats.keys()))
print('Coverage of test data: ',
      sum([1 if w in trainingDataStats else 0 for w in testData]) / TEST_DATA_COUNT)
print()

# dictionaries for unigram, bigram and trigram counts
unigram = {}
bigram = {}
trigram = {}

wh2 = "<<s>>"  # word history 2
wh1 = "<s>"  # word history 1
for w in data:  # iterate over data
    # +1 for the word in unigram dictionary
    unigram[w] = 1 if w not in unigram else unigram[w] + 1

    # add history to bigram if not there yet
    if wh1 not in bigram: bigram[wh1] = {}
    # +1 for the word in bigram dictionary
    bigram[wh1][w] = 1 if w not in bigram[wh1] else bigram[wh1][w] + 1

    # add history to trigram if not there yet
    if wh2 not in trigram: trigram[wh2] = {}
    if wh1 not in trigram[wh2]: trigram[wh2][wh1] = {}
    # +1 for the word in trigram dictionary
    trigram[wh2][wh1][w] = 1 if w not in trigram[wh2][wh1] else trigram[wh2][wh1][w] + 1

    # move histories
    wh2 = wh1
    wh1 = w

vocabularySize = len(unigram)  # number of unique words
textSize = len(data)  # number of all words

# Helper function: returns 0 if not in unigram dictionary, otherwise returns unigram count
def unigramCount(word):
    if word not in unigram:
        return 0

    return unigram[word]

# Helper function: returns 0 if not in bigram dictionary, otherwise returns bigram count
def bigramCount(word, h1):
    if (h1 not in bigram) or (word not in bigram[h1]):
        return 0

    return bigram[h1][word]

# Helper function: returns 0 if not in trigram dictionary, otherwise returns trigram count
def trigramCount(word, h1, h2):
    if (h2 not in trigram) or (h1 not in trigram[h2]) or (word not in trigram[h2][h1]):
        return 0

    return trigram[h2][h1][word]

# returns 0 if denominator is 0, otherwise performs division
def divisionOrZero(nominator, denominator):
    if denominator == 0: return 0
    return nominator / denominator

# Calculates uniform conditional probability (p0)
def uniformProbConditional():
    return 1 / vocabularySize


# Calculates unigram conditional probability (p1)
def unigramProbConditional(word):
    return unigramCount(word) / textSize


# Calculates bigram conditional probability (p2)
def bigramProbConditional(word, h1):
    # use uniform conditional probability when both bigram and unigram counts are zero
    # @why? because the scripts says so
    if bigramCount(word, h1) == 0:
        if unigramCount(h1) == 0:
            return uniformProbConditional()

    return divisionOrZero(bigramCount(word, h1), unigramCount(h1))


# Calculates trigram conditional probability (p3)
def trigramProbConditional(word, h1, h2):
    # use uniform conditional probability when both bigram and unigram counts are zero
    # @why? because the scripts says so
    if trigramCount(word, h1, h2) == 0:
        if bigramCount(h1, h2) == 0:
            return uniformProbConditional()

    return divisionOrZero(trigramCount(word, h1, h2), bigramCount(h1, h2))


# Calculates smoothed conditional probability
# using linear combination of lambdas and uniform, unigram, bigram and trigram conditional probabilities
def smoothedProbConditional(word, h1, h2, lambdas):
    return lambdas[0] * uniformProbConditional() \
           + lambdas[1] * unigramProbConditional(word) \
           + lambdas[2] * bigramProbConditional(word, h1) \
           + lambdas[3] * trigramProbConditional(word, h1, h2)


### EM ALGORITHM ###

# initialize lambdas
lambdas = [1 / 4] * 4
while True:
    expCounts = [0] * 4  # expected counts for lambdas
    wh2 = "<<s>>"  # word history 2
    wh1 = "<s>"  # word history 1

    # train lambdas using the heldoutData
    for w in heldoutData:
        # calculate expected counts for each lambda using uniform, unigrams, bigrams, trigrams
        expCounts[0] += lambdas[0] * uniformProbConditional() / smoothedProbConditional(w, wh1, wh2, lambdas)
        expCounts[1] += lambdas[1] * unigramProbConditional(w) / smoothedProbConditional(w, wh1, wh2, lambdas)
        expCounts[2] += lambdas[2] * bigramProbConditional(w, wh1) / smoothedProbConditional(w, wh1, wh2, lambdas)
        expCounts[3] += lambdas[3] * trigramProbConditional(w, wh1, wh2) / smoothedProbConditional(w, wh1, wh2, lambdas)

        # move histories
        wh2 = wh1
        wh1 = w

    # get new lambdas by normalizing expected counts
    newLambdas = list(map(lambda x: x / sum(expCounts), expCounts))

    # terminate if all lambdas differ by less than 0.0001
    if all([abs(x - y) < 0.0001 for x, y in zip(newLambdas, lambdas)]): break

    # use new lambdas for the next cycle
    lambdas = newLambdas


# Modifies the last lambda by difference
# and normalizes the rest of the lambdas to that they sum up to 1
def modifyLambdas(lambdas, difference):
    newLambdas = list(map(lambda x: x - difference * x / sum(lambdas[0:3]), lambdas))
    newLambdas[3] = lambdas[3] + difference
    return newLambdas


# Boosts the last lambda by percentage
def boostLambdas(lambdas, percentage):
    return modifyLambdas(lambdas, (1 - lambdas[3]) * percentage)


# Discounts the last lambda by percentage
def discountLambdas(lambdas, percentage):
    return modifyLambdas(lambdas, (-1) * lambdas[3] * (1 - percentage))


# Lists of percentages that we want to use for boosting and discounting
discountVector = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
boostingVector = [   .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]


# Calculates cross entropy on the testData using given lambdas
def calculateCrossEntropy(lambdas):
    wh2 = "<<s>>"  # word history 2
    wh1 = "<s>"  # word history 1
    crossEntropy = 0
    for w in testData:
        crossEntropy -= log(smoothedProbConditional(w, wh1, wh2, lambdas), 2)

        # move histories
        wh2 = wh1
        wh1 = w
    return crossEntropy / len(testData)  # we need to normalize by the length of the test data

# discount lambdas by some factor and calculate cross entropy
for b in discountVector:
    print('discount to ', b, 'entropy', calculateCrossEntropy(discountLambdas(lambdas, b)), 'lambdas',
          discountLambdas(lambdas, b))

# calculate cross entropy for orignal lambdas too
print('original lambdas entropy', calculateCrossEntropy(lambdas), 'lambdas', lambdas)

# boost lambdas by some factor and calculate cross entropy
for b in boostingVector:
    print('boosting by ', b, 'entropy', calculateCrossEntropy(boostLambdas(lambdas, b)), 'lambdas',
          boostLambdas(lambdas, b))
