from gensim.models.fasttext import FastText as ft
from gensim.test.utils import datapath
import numpy as np
from os import listdir
import string

root = datapath('/Users/hanjaewon/폴더모음/학교생활/졸업과제/fasttextTest/')#/말뭉치모음/
model = ft(size=64)
corpus = datapath(root+"morp1.txt")
model.build_vocab(corpus_file=corpus,min_count=5)
model.train(corpus_file = corpus, epochs=model.epochs, total_examples=model.corpus_count, total_words=model.corpus_total_words)
count =0
for file in listdir(root):
    if count==18:
        break
    try:
        if file!="morp1.txt" and file[-3:]=="txt":
            count += 1
            print(file)
            corpus = datapath(root+file)
            model.build_vocab(corpus_file= corpus,update=True,min_count=5)
            model.train(corpus_file = corpus, epochs=model.epochs, total_examples=model.corpus_count, total_words=model.corpus_total_words)
    except:
        print("error occured in {}".format(file))



print("save model")
model.save("ft_model64_1")

#model = ft.load(root+"ft_model64_2")
#names = ["morp28.txt","morp29.txt","morp30-8.txt","morp30-9.txt","morp30-10.txt","morp30-11.txt"]
#for name in names:
#    try:
#        corpus = datapath(root+"말뭉치모음/"+name)
#        print("s: "+name)
#        model.build_vocab(corpus_file=corpus,update=True,min_count=5)
#        model.train(corpus_file=corpus,epochs=model.epochs,total_examples=model.corpus_count,total_words=model.corpus_total_words)
#    except:
#        print(name)
#
#print("model saved")
#model.save("ft_model64_3")

#print(model.corpus_total_words)
#corpus = datapath(root+"sentences.txt")
#model.build_vocab(corpus_file=corpus,update=True,min_count=5)
#model.train(corpus_file=corpus,epochs=model.epochs,total_examples=model.corpus_count,total_words=model.corpus_total_words)
#print("model saved")
#model.save("updated_model")

