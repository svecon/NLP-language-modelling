import sys
import random
import operator
import functools
from math import log

EXPERIMENT_COUNT = 10
FIRST_WORD_PADDING = "<s>"


def biGramProbConditional(word, h1):
    if (h1 not in biGram) or (word not in biGram[h1]):
        return 0

    if h1 not in uniGram:
        return 0

    return biGram[h1][word] / uniGram[h1]


def biGramProb(word, h1):
    if (h1 not in biGram) or (word not in biGram[h1]):
        return 0

    return biGram[h1][word] / len(data)


def calculateEntropy(dataSet):
    h1 = "<s>"
    entropy = 0
    for w in dataSet:
        entropy -= biGramProb(w, h1) * log(biGramProbConditional(w, h1), 2)
        h1 = w
    return entropy


DATA_ENCODING = "iso-8859-2"
data = [line.strip() for line in open(sys.argv[1], 'r', encoding=DATA_ENCODING)]

characters = {}
words = {}

messUpLikelihoods = [0, .00001, .0001, .001, .01, .05, .1, .25, .5]

for w in data:
    words[w] = 1 if w not in words else words[w] + 1
    for c in w:
        characters[c] = 1 if c not in characters else characters[c] + 1

wordsList = list(words.keys())
charList = list(characters.keys())

for messUpWords in [0,1]:
    print("=== Messing up characters ===" if messUpWords == 0 else "=== Messing up words ===")
    for messUpProb in messUpLikelihoods:

        entropies = []
        wordCounts = []
        characterCounts = []
        biggestFrequencies = []
        noOfFreqOne = []

        for i in range(EXPERIMENT_COUNT):
            if (messUpProb == 0) and i > 0: break # no need to repeat when probability is 0

            uniGram = {FIRST_WORD_PADDING: 1}
            biGram = {}
            newDataSet = []

            wh1 = FIRST_WORD_PADDING  # word history
            for w in data:
                newWord = w

                if messUpWords:
                    if random.random() <= messUpProb:
                        newWord = random.choice(wordsList)
                else:
                    newWord = list(newWord)
                    for il in range(len(newWord)):
                        if random.random() <= messUpProb:
                            newWord[il] = random.choice(charList)
                    newWord = ''.join(newWord)

                uniGram[newWord] = 1 if newWord not in uniGram else uniGram[newWord] + 1

                if wh1 not in biGram: biGram[wh1] = {}
                biGram[wh1][newWord] = 1 if newWord not in biGram[wh1] else biGram[wh1][newWord] + 1

                newDataSet.append(newWord)
                wh1 = newWord

            entropies.append(calculateEntropy(newDataSet))
            wordCounts.append(len(uniGram))
            characterCounts.append(sum([ len(x) for x in newDataSet ]))
            biggestFrequencies.append(max(uniGram.items(), key=operator.itemgetter(1))[1])
            noOfFreqOne.append(functools.reduce(lambda count, x: count + (x == 1), uniGram.values(), 0))


        print("Messup: ", messUpProb)
        print("Entropies: ", entropies)
        print("Word count: ", wordCounts)
        print("Number of characters: ", characterCounts)
        print("Number of characters per word: ", [ x / y for x,y in zip(characterCounts, wordCounts)])
        print("Biggest frequency: ", biggestFrequencies)
        print("Words with frequency 1: ", noOfFreqOne)
        print()
