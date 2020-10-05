from gensim.models.fasttext import FastText as ft
from gensim.test.utils import datapath
import numpy as np
from os import listdir

#root = datapath('/Users/hanjaewon/폴더모음/학교생활/졸업과제/fasttextTest/')

questions = datapath("/Users/hanjaewon/폴더모음/학교생활/졸업과제/fasttextTest/ko_analogy.txt")
model = ft.load("/Users/hanjaewon/폴더모음/학교생활/졸업과제/models/lemma2/gan2vec_model_1500")
#print(model.corpus_total_words)
acc = model.wv.evaluate_word_analogies(questions)
count =0
print("Overall accuracy: "+str(acc[0]))
for sec in acc[1]:
#    print(sec)
    try:
        print("section: "+sec["section"])
        print("correct: "+str(len(sec["correct"]))+", "+"incorrect: "+str(len(sec["incorrect"])))
        print()
    except:
        print(str(count)+"th section error")
    count +=1
#print(acc[1])
#
#print("accuracy test result:")
#print(acc)
#len = len(acc[1])
#print(acc[1][len-1]['correct'])

