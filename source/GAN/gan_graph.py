from gensim.models import FastText

# 모델 로딩
#model = FastText.load('GAN2vec/gan2vec_model_0')

#print(model.similarity('주의', '되다')) # 컴퓨터의 word vector 출력
#print(model.most_similar('-', topn=30)) # 컴퓨터의 word vector 출력
import pickle
import matplotlib.pyplot as plt

# f = open('./기본기분/file.dat.2000', 'rb')
# f = open('./기본기분no update epoch2000/file.dat.2000', 'rb')
# f = open('./사정사장/file.dat.2000', 'rb')
# f = open('./사정사장no update epoch2000/file.dat.2000', 'rb')
# f = open('./의지의자/file.dat.2000', 'rb')
# f = open('./의지의자no update epoch2000/file.dat.2000', 'rb')
# f = open('./주의주위/file.dat.2000', 'rb')
# f = open('./주의주위no update epoch2000/file.dat.2000', 'rb')
# f = open('./지식자식/file.dat.2000', 'rb')
f = open('./지식자식no update epoch2000/file.dat.2000', 'rb')
d = pickle.load(f) #string 로드
x_point = d
d = pickle.load(f) #string 로드
y_point = d
d = pickle.load(f) #string 로드
y_point2 = d
d = pickle.load(f) #string 로드
sentence_7_point = d
d = pickle.load(f) #string 로드
sentence_5_point = d
d = pickle.load(f) #string 로드
sentence_3_point = d
d = pickle.load(f) #string 로드
sentence_1_point = d
f.close()

plt.ylabel('loss')
plt.xlabel('epochs')
plt.plot(x_point, y_point, label='G loss')
plt.plot(x_point, y_point2, label='D loss')
plt.legend()
plt.show()

plt.ylabel('cosine')
plt.xlabel('epochs')
plt.plot(x_point, sentence_7_point, label='word 7')
plt.plot(x_point, sentence_5_point, label='word 5')
plt.plot(x_point, sentence_3_point, label='word 3')
plt.plot(x_point, sentence_1_point, label='word 1')
plt.legend()
plt.show()
