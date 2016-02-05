import sys

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


# Checking if unigram, bigram and trigram have all the same data size
# checkUnigram = 0
# for v in unigram.values():
#     checkUnigram += v
#
# checkBigram = 0
# for v in bigram.values():
#     for vv in v.values():
#         checkBigram += vv
#
# checkTrigram = 0
# for v in trigram.values():
#     for vv in v.values():
#         for vvv in vv.values():
#             checkTrigram += vvv
#
# assert checkUnigram == checkBigram == checkTrigram

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
wh2 = "<<s>>"  # word history 2
wh1 = "<s>"  # word history 1
lambdas = [.7,.1,.1,.1] #[1/4]*4
expCounts = [0]*4
while True:
    for w in heldoutData:
        for i in range(len(lambdas)):
            expCounts[i] += lambdas[i]*trigramProbConditional(w,wh1,wh2)/smoothedProbConditional(w,wh1,wh2,lambdas)

        wh2 = wh1
        wh1 = w
    print(expCounts)

    newLambdas = [0]*4
    for i in range(len(lambdas)):
        newLambdas[i] = expCounts[i] / sum(expCounts)
    print(newLambdas)
    #newLambdas = map(lambda x: x / sum(expCounts), expCounts)

    print([ abs(i - j) for i,j in zip(newLambdas, lambdas) ])

    if sum([ abs(i - j) for i,j in zip(newLambdas, lambdas) ]) < 0.0001:
        break

    print(lambdas)
    lambdas = newLambdas

print(lambdas)