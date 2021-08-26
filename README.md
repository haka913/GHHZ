# GHHZ

GAN을 이용한 문맥의존 철자 오류 교정

Team No: 1

Team Name: 깊은 학습 하조

Team member: 201324462 서준교, 201424419 김세원, 201724614 한재원

## Install
```bash
$ git clone https://github.com/haka913/GHHZ.git
$ cd GHHZ
$ pip install -r requirements.txt
```
### package
Deep Learning: keras, tensorflow

Word Embedding: gensim, fasttext

Korean Analyzer: pynori


## GAN을 이용한 문맥의존 철자 오류 교정

기존 언어학의 전문가가 설계한 규칙을 사용하거나 통계적인 분석방법을 사용하여 오류 교정을 시도할 수 있지만 비용 대비 성능이 높지 않다.

이 문제를 해결하기 위해 광범위한 문맥을 대상으로 DCGAN(deep convolutional generative adversarial networks)을 이용한 딥러닝과 워드 임베딩(fasttext)을 사용하여 문맥 의존 철자 오류 교정을 시도하였다.

### DCGAN Generator, Discriminator
Generator <img src=https://user-images.githubusercontent.com/43568065/130975900-124ca712-e059-4f28-a124-01a1f6b299c0.png generator width="15%" > Discriminator<img src=https://user-images.githubusercontent.com/43568065/130975961-3080e8eb-4393-490d-964e-2f4567566481.png discriminator width="15%">

### 학습 진행 순서
![image](https://user-images.githubusercontent.com/43568065/130986880-cd8f43a4-28e2-4407-a737-cb16ca084f41.png)


## Test Learning

```bash
$ cd GHHZ/source/GAN
$ python .\gan.py
or
$ python .\wgan_fasttext_m.py
```

## 결과

DCGAN으로 학습후 loss와 cosine 유사도


![image](https://user-images.githubusercontent.com/43568065/130984986-9681547b-5bc3-4a9f-85ae-7919e1fca6d7.png)


|      |  **Basline**  |               |               | **GAN-embedding**|            |               |
|:----:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|      |   precision   |     recall    |       F1      |   precision   |     recall    |       F1      |
| 기본 |     93.24%    |     85.95%    |     89.45%    |     93.13%    |     95.86%    |     89.35%    |
| 기분 |     90.65%    |     83.25%    |     86.79%    |     90.78%    |     83.45%    |     86.96%    |
| 사장 |     76.91%    |     61.34%    |     68.25%    |     77.02%    |     61.43%    |     68.35%    |
| 사정 |     91.29%    |     84.34%    |     87.68%    |     91.29%    |     84.34%    |     87.68%    |
| 의자 |     96.64%    |     92.94%    |     94.75%    |     96.64%    |     92.94%    |     94.75%    |
| 의지 |     90.50%    |     82.21%    |     86.16%    |     90.36%    |     81.88%    |     85.91%    |
| 자식 |     96.97%    |     94.86%    |     95.90%    |     96.97%    |     94.86%    |     95.90%    |
| 지식 |     66.28%    |     50.05%    |     57.03%    |     66.28%    |     50.05%    |     57.03%    |
| 주의 |     89.38%    |     81.53%    |     85.28%    |     89.38%    |     81.53%    |     85.28%    |
| 주위 |     94.02%    |     87.66%    |     90.73%    |     94.35%    |     88.06%    |     91.10%    |
| 평균 |               |               |     84.20%    |               |               |     84.23%    |
