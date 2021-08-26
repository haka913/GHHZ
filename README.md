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

|      |    **Basline ** |               |               | **GAN-embedding** |               |               |
|:----:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|      |   precision   |     recall    |       F1      |   precision   |     recall    |       F1      |
| 기본 |     87.69%    |     78.24%    |     82.70%    |     89.14%    |     80.42%    |     84.56%    |
| 기분 |     90.59%    |     82.63%    |     86.43%    |     88.28%    |     78.70%    |     83.21%    |
| 사장 |     61.69%    |     44.28%    |     51.56%    |     63.48%    |     46.37%    |     53.59%    |
| 사정 |     95.92%    |     91.96%    |     93.90%    |     95.21%    |     90.37%    |     92.73%    |
| 의자 |     99.51%    |     99.22%    |     99.36%    |     99.11%    |     98.63%    |     98.87%    |
| 의지 |     79.29%    |     65.61%    |     71.80%    |     85.86%    |     74.89%    |     80.00%    |
| 자식 |     97.89%    |     96.05%    |     96.96%    |     98.08%    |     96.15%    |     97.11%    |
| 지식 |     73.50%    |     57.83%    |     64.73%    |     73.41%    |     57.83%    |     64.70%    |
| 주의 |     83.14%    |     71.06%    |     76.62%    |     84.09%    |     72.64%    |     77.95%    |
| 주위 |     96.16%    |     92.24%    |     94.16%    |     94.07%    |     88.46%    |     91.18%    |
| 평균 |               |               |     81.82%    |               |               |     82.39%    |
