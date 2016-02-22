import sys
import random
import operator
import functools
from math import log

TEST_DATA_COUNT = 20000
HELDOUT_DATA_COUNT = 40000
DATA_ENCODING = "iso-8859-2"

data = [line.strip() for line in open(sys.argv[1], 'r', encoding=DATA_ENCODING)]

testData = data[-TEST_DATA_COUNT:]
del data[-TEST_DATA_COUNT:]

heldoutData = data[-HELDOUT_DATA_COUNT:]
del data[-HELDOUT_DATA_COUNT:]

unigram = {}
bigram = {}
trigram = {}

wh2 = "<<s>>"  # word history 2
wh1 = "<s>"  # word history 1
for w in data:
    unigram[w] = 1 if w not in unigram else unigram[w] + 1

    if wh1 not in bigram: bigram[wh1] = {}
    bigram[wh1][w] = 1 if w not in bigram[wh1] else bigram[wh1][w] + 1

    if wh2 not in trigram: trigram[wh2] = {}
    if wh1 not in trigram[wh2]: trigram[wh2][wh1] = {}
    trigram[wh2][wh1][w] = 1 if w not in trigram[wh2][wh1] else trigram[wh2][wh1][w] + 1

    wh2 = wh1
    wh1 = w

vocabularySize = len(unigram)
textSize = len(data)

def flatProbConditional():
    return 1 / vocabularySize


def unigramProbConditional(word):
    if word not in unigram:
        return 0

    return unigram[word] / textSize


def bigramProbConditional(word, h1):
    if (h1 not in bigram) or (word not in bigram[h1]):
        return 0

    if h1 not in unigram:
        return 0

    return bigram[h1][word] / unigram[h1]


def trigramProbConditional(word, h1, h2):
    if (h2 not in trigram) or (h1 not in trigram[h2]) or (word not in trigram[h2][h1]):
        return 0

    if (h2 not in bigram) or (h1 not in bigram[h2]):
        return 0

    return trigram[h2][h1][word] / bigram[h2][h1] #sum(trigram[h2][h1].values())


def smoothedProbConditional(word, h1, h2, lambdas):
    return   lambdas[0] * flatProbConditional() \
           + lambdas[1] * unigramProbConditional(word) \
           + lambdas[2] * bigramProbConditional(word, h1) \
           + lambdas[3] * trigramProbConditional(word, h1, h2)


# print(flatProbConditional())
# print(unigramProbConditional('ale'))
# print(bigramProbConditional('ale', ','))
# print(trigramProbConditional('ale', ',', 'situace'))

# EM ALGORITHM
lambdas = [.7,.1,.1,.1] #[1/4]*4
while False:
    expCounts = [0]*4
    wh2 = "<<s>>"  # word history 2
    wh1 = "<s>"  # word history 1
    for w in data:
        # print("{:s} {:s} {:s} {:.6e} {:.6e} {:.6e}".format(wh2, wh1, w, unigramProbConditional(w), bigramProbConditional(w, wh1), trigramProbConditional(w, wh1, wh2)))
        expCounts[0] += lambdas[0]*flatProbConditional()/smoothedProbConditional(w,wh1,wh2,lambdas)
        expCounts[1] += lambdas[1]*unigramProbConditional(w)/smoothedProbConditional(w,wh1,wh2,lambdas)
        expCounts[2] += lambdas[2]*bigramProbConditional(w,wh1)/smoothedProbConditional(w,wh1,wh2,lambdas)
        expCounts[3] += lambdas[3]*trigramProbConditional(w,wh1,wh2)/smoothedProbConditional(w,wh1,wh2,lambdas)

        wh2 = wh1
        wh1 = w

    newLambdas = list(map(lambda x: x / sum(expCounts), expCounts))

    if all( [ abs(x-y) < 0.0001 for x,y in zip(newLambdas, lambdas) ] ): break

    lambdas = newLambdas

lambdasEN = [0.09843454910828085, 0.26415213029970314, 0.5077064080458099, 0.1297069125462061];
lambdas = lambdasEN

def modifyLambdas(lambdas, difference):
    newLambdas = list(map(lambda x: x - difference * x / sum(lambdas[0:3]), lambdas))
    newLambdas[3] = lambdas[3] + difference
    return newLambdas

def boostLambdas(lambdas, percentage):
    return modifyLambdas(lambdas, (1 - lambdas[3]) * percentage)

boostLambdas([0.140, 0.429, 0.245, 0.186 ], 0.1)

def discountLambdas(lambdas, percentage):
    return modifyLambdas(lambdas, (-1) * lambdas[3] * (1 - percentage))

discountLambdas([0.140, 0.429, 0.245, 0.186 ], .9)

boostingVector = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
discountVector = [.9, .8, .7, .6, .5, .4, .3, .2, .1, 0]

def calculateCrossEntropy(lambdas):
    wh2 = "<<s>>" # word history 2
    wh1 = "<s>" # word history 1
    crossEntropy = 0
    for w in testData:
        crossEntropy -= log(smoothedProbConditional(w, wh1, wh2, lambdas), 2)

        wh2 = wh1
        wh1 = w
    return crossEntropy / len(testData)

for b in boostingVector:
    print('boosting by ', b, 'entropy', calculateCrossEntropy(boostLambdas(lambdas, b)), 'lambdas', boostLambdas(lambdas, b))

for b in discountVector:
    print('discount to ', b, 'entropy', calculateCrossEntropy(discountLambdas(lambdas, b)), 'lambdas', discountLambdas(lambdas, b))
