import json
import nltk
from rouge_score import rouge_scorer

f1 = open('user.txt', 'r')
f2 = open('gold.txt', 'r')
f3 = open('gene.txt', 'r')

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

cnt4 = 0
cnt3 = 0
cnt2 = 0
cnt1 = 0
cnt = 0

rg1_score = 0
rgl_score = 0

while True:
    l1 = f1.readline()
    l2 = f2.readline().strip()
    l3 = f3.readline().strip()

    if l2 == "":
        break

    scores = scorer.score(l2, l3)
    rg1 = scores['rouge1'].precision
    rg1_score += rg1
    rgl_score += scores['rougeL'].precision

    expected = l2.split(' ')
    actual = l3.split(' ')

    BLEUscore4 = nltk.translate.bleu_score.sentence_bleu([expected], actual, weights=(0.5, 0.3, 0.1, 0.1))
    BLEUscore3 = nltk.translate.bleu_score.sentence_bleu([expected], actual, weights=(0.5, 0.3, 0.2, 0))
    BLEUscore2 = nltk.translate.bleu_score.sentence_bleu([expected], actual, weights=(0.7, 0.3, 0, 0))
    BLEUscore1 = nltk.translate.bleu_score.sentence_bleu([expected], actual, weights=(1, 0, 0, 0))
    cnt1 += BLEUscore1
    cnt2 += BLEUscore2
    cnt3 += BLEUscore3
    cnt4 += BLEUscore4
    cnt += 1
    print(cnt)
    # print(BLEUscore4)

cnt1 /= cnt
cnt2 /= cnt
cnt3 /= cnt
cnt4 /= cnt
rg1_score /= cnt
rgl_score /= cnt
print("Bleu Score 1 : ", cnt1)
print("Bleu Score 2 : ", cnt2)
print("Bleu Score 3 : ", cnt3)
print("Bleu Score 4 : ", cnt4)
print("Rouge1 : ", rg1_score)
print("RougeL : ", rgl_score)

# print(cnt_scores)
