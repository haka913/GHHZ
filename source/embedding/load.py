from gensim.models.fasttext import FastText as ft
from gensim.test.utils import datapath
import numpy as np

model = ft.load('/Users/hanjaewon/Downloads/아카이브/'+"ft_model64_2")
print("학습한 문장 수: "+str(model.corpus_count))
print("학습한 단어 수: "+str(model.corpus_total_words))


print("=========워드 임베딩과 Fasttext의 oov기능 확인=========")
print("예술이라는 단어가 학습단어에 있었는가?")
print('예술' in model.wv.vocab) #vocab에 단어가 있으면 True

print("예술 vector값은 : ")
print(model.wv.__getitem__('예술')) #경제 단어의 vector값을 보여줌

print("예슬이라는 단어가 학습단어에 있었는가?")
print('예슬' in model.wv.vocab) #vocab에 단어가 있으면 True

print("예슬 vector값은 : ")
print(model.wv.__getitem__('예슬')) #경제 단어의 vector값을 보여줌

print(" ")
print(" ")
print("=========임베딩 모델을 이용한 유사도 확인=========")
#print('경기' in model.wv.vocab)
#print(model['경기']) #DeprecationWarning
print("의자와 의지라는 단어의 유사도는?")
print(model.wv.similarity("의자", "의지")) #두 단어의 유사도를 출력함
print(model.wv.most_similar("의자"))
print("의자와 유사도가 높은 단어:")
for v in model.wv.most_similar("의자"):
    print(v)
