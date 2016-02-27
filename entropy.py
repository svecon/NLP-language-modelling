# Calculating conditional entropy of a text
# Program expects first argument to be filename of the text file to be examined.

# The script iterates over the text file and messes up the words or characters.
# The script then calculates counts c(i,j), which is a number of bigrams ij.
# And stores this in biGram dictionary.

# Finally it iterates over the biGram dicrionary and calculates conditional entropy.

# @AUTHOR: Ondřej Švec

import sys
import random
import operator
import functools
from math import log

EXPERIMENT_COUNT = 10 # number of repetitions for one messup probability
FIRST_WORD_PADDING = "<s>" # word used as a beginning of a text

# P(word | h1): calculates conditional bigram probability
def biGramProbConditional(word, h1):
    if (h1 not in biGram) or (word not in biGram[h1]):
        return 0

    if h1 not in uniGram:
        return 0

    return biGram[h1][word] / uniGram[h1]

# P(h1, word): calculates joint bigram probability
def biGramProb(word, h1):
    if (h1 not in biGram) or (word not in biGram[h1]):
        return 0

    return biGram[h1][word] / len(data)

# calculates conditional entropy of dataSet using biGram dictionary
def calculateEntropy(dataSet):
    entropy = 0
    for h1 in biGram.keys():
        for w in biGram[h1].keys():
            if biGramProbConditional(w, h1) == 0: continue
            entropy -= biGramProb(w, h1) * log(biGramProbConditional(w, h1), 2)
    return entropy


# read text file and save lines as array to data
data = [line.strip() for line in open(sys.argv[1], 'r', encoding="iso-8859-2")]

characters = {} # dictionary of characters that appear in the text file
words = {} # dictionary of words that appear in the text file

# array of messup probabilities (probability with which the word or character is messed up)
messUpLikelihoods = [0, .00001, .0001, .001, .01, .05, .1, .2, .3 , .4, .5, .6, .7, .8, .9, 1]

# get all unique words and characters from the input text file
for w in data:
    words[w] = 1 if w not in words else words[w] + 1
    for c in w:
        characters[c] = 1 if c not in characters else characters[c] + 1

# converting dictionaries to lists
wordsList = list(words.keys())
charList = list(characters.keys())

# mess up characters (messUpWords==0) and words (messUpWords==1)
for messUpWords in [0,1]:
    print("=== Messing up characters ===" if messUpWords == 0 else "=== Messing up words ===")

    # iterate over all messup probabilities
    for messUpProb in messUpLikelihoods:

    	# lists in which we store calculated information from individual experiments (each messup probability is run 10 times)
        entropies = []
        wordCounts = []
        characterCounts = []
        biggestFrequencies = []
        noOfFreqOne = []

        # mess the words/characters up 10 times
        for i in range(EXPERIMENT_COUNT):
            if (messUpProb == 0) and i > 0: break # no need to repeat when probability is 0

            uniGram = {} # dictionary for unigram counts
            biGram = {} # dictionary for bigram counts
            newDataSet = []

            wh1 = FIRST_WORD_PADDING  # word history
            for w in data: # iterate over data (text file)
                newWord = w

                # get a random word from available words when the messup prob is higher than random
                if messUpWords:
                    if random.random() <= messUpProb:
                        newWord = random.choice(wordsList)
                # iterate over a word and get a random character from all available characters if the messup prob is higher then random
                else:
                    newWord = list(newWord)
                    for il in range(len(newWord)):
                        if random.random() <= messUpProb:
                            newWord[il] = random.choice(charList)
                    newWord = ''.join(newWord)

                # +1 for the word in unigram dictionary
                uniGram[newWord] = 1 if newWord not in uniGram else uniGram[newWord] + 1

                if wh1 not in biGram: biGram[wh1] = {} # add history to biGram if not there yet
                # +1 for the word in bigram dictionary
                biGram[wh1][newWord] = 1 if newWord not in biGram[wh1] else biGram[wh1][newWord] + 1

                # add the messed up word in list to save for later (calculating entropy)
                newDataSet.append(newWord)
                wh1 = newWord # move history

            # calculate conditional entropy and add it into list
            entropies.append(calculateEntropy(newDataSet))
            # number of unique words
            wordCounts.append(len(uniGram))
            # number of characters in the new (messed up) text
            characterCounts.append(sum([ len(x) for x in newDataSet ]))
            # what is a frequency of a word that is most common?
            biggestFrequencies.append(max(uniGram.items(), key=operator.itemgetter(1))[1])
            # number of words that only appear once in the text
            noOfFreqOne.append(functools.reduce(lambda count, x: count + (x == 1), uniGram.values(), 0))

        # print results of all experiments for one messup probability
        print("Messup: ", messUpProb)
        print("Entropies: ", entropies)
        print("Word count: ", wordCounts)
        print("Number of characters: ", characterCounts)
        print("Number of characters per word: ", [ x / len(newDataSet) for x in characterCounts])
        print("Biggest frequency: ", biggestFrequencies)
        print("Words with frequency 1: ", noOfFreqOne)
        print()
