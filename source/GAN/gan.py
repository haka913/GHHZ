import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
import tensorflow as tf
# from tensorflow.contrib.distributions import RelaxedOneHotCategorical as gumbel
from gensim.models import FastText
from gensim.models.fasttext import FastText as ft
raw_data = pd.read_csv("train_data_lemma_with_da.txt", sep="\n", encoding="UTF-8")

model = ft.load('C:/Users/paul/Desktop/NLP2/embedding642/ft_model64_2')
global_dim = 64
glover_len = 7
# glover_len = 10
data = []
x_point = []
y_point = []
y_point2 = []

sentence_1_point = []
sentence_3_point = []
sentence_5_point = []
sentence_7_point = []
ft_list = list(model.wv.vocab)
# vec_size = len(list(model.wv.vocab))
vec_size = len(ft_list)
for i in raw_data["text"]:
    split_text = i.split(" ")

    if len(split_text) < glover_len:
        pass
    else:
        cnt = 0
        data_temp = []
        for j in split_text:
            data_temp.append(j)

            cnt += 1
            if cnt == 7:
            # if cnt == 10:
                cnt = 0
                data.append(data_temp)
                data_temp = []

        if cnt > 0 and cnt < 7:
        # if cnt > 0 and cnt < 10:
            for k in range(7 - cnt):
            # for k in range(10 - cnt):
                temp = np.random.randint(vec_size)
                # print(list(model.wv.vocab)[temp])
                # tmp_str = list(model.wv.vocab)[temp]
                # data_temp.append(tmp_str)
                data_temp.append(ft_list[temp])
                # tmp = np.random.rand(glover_len, global_dim)
                # data_temp.append("<pad>")

            data.append(data_temp)

# model = FastText.load_fasttext_format('raw/kor_model_100d_4min_5w.bin')
# model = ft.load('C:/Users/paul/Desktop/NLP2/embedding642/ft_model64_2')
# model = FastText(data, size = global_dim, window=5, min_count=1)

sentence_to_word_vec = np.zeros(shape=(len(data),glover_len,global_dim))
for sentence_index, i in enumerate(data):
    temp_list = np.zeros(shape=(glover_len,global_dim))
    for idx, j in enumerate(i):
        try:
            temp_list[idx] = np.array([model.wv.get_vector(j)])
        except:
            # temp_list[idx] = np.random.rand(glover_len, global_dim)
            temp_list[idx] = np.array([model.wv.get_vector(".")])
    temp_list = np.reshape(temp_list,(1,glover_len,global_dim))
    sentence_to_word_vec[sentence_index] = temp_list

sentence_to_word_vec.shape

sentence_to_word_vec = np.expand_dims(sentence_to_word_vec,axis = -1)


class GAN2vec():
    def __init__(self):
        # Input shape
        self.sentence_length = glover_len
        self.word_dimension = global_dim
        self.channels = 1
        self.sentence_shape = (self.sentence_length, self.word_dimension, self.channels)
        self.latent_dim = 100

        optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        sentence = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(sentence)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((1, 1, 512)))
        model.add(Conv2DTranspose(256, kernel_size=(3, 16), strides=2))
        model.add(Activation("relu"))
        # glover_len = 7
        model.add(Conv2DTranspose(1, kernel_size=(3, 34), strides=2))
        model.add(Reshape((7, 64, 1)))

        # glover_len=10
        # model.add(Conv2DTranspose(1, kernel_size=(4, 19), strides=3))
        # model.add(Reshape((10, 64, 1)))
        model.summary()
        model.save("generator.h5")

        noise = Input(shape=(self.latent_dim,))
        sentence = model(noise)

        return Model(noise, sentence)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(256, kernel_size=(3, 64), input_shape=self.sentence_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=(5, 1)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.save("discriminator.h5")
        sentence = Input(shape=self.sentence_shape)
        validity = model(sentence)

        return Model(sentence, validity)

    def pretrain_D(self, epochs, batch_size=128):
        X_train = sentence_to_word_vec
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1
        print("pretraining D")
        for epoch in range(epochs):
            print("{}epochs".format(epoch))
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sentences = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_sentences = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(sentences, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_sentences, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = sentence_to_word_vec

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            x_point.append(epoch)
            y_point2.append(d_loss[0])
            y_point.append(g_loss)

            # If at save interval => save generated image samples
        #    if epoch % save_interval == 0:
        #        self.show_sentence(epoch)
            self.show_sentence(epoch)

            if epoch % 500 == 0:
                model.save('gan2vec_model_' + str(epoch))

    def show_sentence(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = GAN2vec.generator.predict(noise)
        test = np.squeeze(gen_sentence)

        sum_1 = 0.0
        sum_3 = 0.0
        sum_5 = 0.0
        sum_7 = 0.0
        cnt = 0
        for i in test:
            sentence = ""
            for j in i:
                temp = model.wv.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "

                if cnt < 1:
                    sum_1 += temp[0][1]
                if cnt < 3:
                    sum_3 += temp[0][1]
                if cnt < 5:
                    sum_5 += temp[0][1]

                sum_7 += temp[0][1]
                cnt += 1
            break
        print(sentence)
        split_ = sentence.split(" ")
        data_=[]
        data_.append(split_)

        # 모델 업데이트
        model.build_vocab(data_,update=True)
        model.train(data_, total_examples=model.corpus_count, epochs= model.iter)


        sum_7 = sum_7 / 7
        sum_5 = sum_5 / 5
        sum_3 = sum_3 / 3
        sum_1 = sum_1 / 1
        sentence_7_point.append(sum_7)
        sentence_5_point.append(sum_5)
        sentence_3_point.append(sum_3)
        sentence_1_point.append(sum_1)

    def predict(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = GAN2vec.generator.predict(noise)
        test = np.squeeze(gen_sentence)
     #   print(gen_sentence)
     #   print(test)

        sentence_list = []
        for i in test:
            sentence = ""
            for j in i:
                temp = model.wv.similar_by_vector(j)
             #   print(temp)
                sentence = sentence + temp[0][0] + " "
            sentence_list.append(sentence)
        return sentence


GAN2vec = GAN2vec()
GAN2vec.pretrain_D(epochs=1)
GAN2vec.train(epochs=4001, batch_size=32, save_interval=50)
GAN2vec.generator.save('gan2vec_model')

#for i in range(10):
#    t = GAN2vec.predict()
#    print(t)

plt.ylabel('loss')
plt.xlabel('epochs')
plt.plot(x_point, y_point, label='G loss')
plt.plot(x_point, y_point2, label='D loss')
plt.legend()
plt.show()

plt.plot(x_point, sentence_7_point, label='word 7')
plt.plot(x_point, sentence_5_point, label='word 5')
plt.plot(x_point, sentence_3_point, label='word 3')
plt.plot(x_point, sentence_1_point, label='word 1')
plt.legend()
plt.show()