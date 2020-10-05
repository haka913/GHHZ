from __future__ import print_function, division
# https://github.com/robert-d-schultz/gan-word-embedding

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.layers import Add
from keras import layers
# from keras.optimizers import Adam

# error occur!
# from keras.optimizers import RMSprop
# use instead
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

import keras.backend as K

from tensorflow.keras.applications import ResNet50, ResNet50V2

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import gensim

from gensim.models.fasttext import FastText as ft
import matplotlib.pyplot as plt

from keras.engine.topology import Layer
from keras.layers import Activation, Conv2D, Add, add


def Residual(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        # merge_input = Conv1D(n_filters, 1, padding='same', activation='relu')(layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    # conv1 = Conv1D(n_filters, 3, padding='same', activation='relu')(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3, 3), padding='same', activation='linear')(conv1)
    # conv2 = Conv1D(n_filters, 3, padding='same', activation='linear')(conv1)

    conv2 = LeakyReLU(alpha=0.2)(conv2)

    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out


# Read Data
raw_data = open("./train2.txt", 'r', encoding="utf-8")
# ignore length <7 sentence,put it in length=7, split sentence 0:7 which length >7
data = []
# [[epoch], [loss]]
plt_G_loss, plt_D_loss = [], []
epoch_list = []
# for i in raw_data["text"]:
for i in raw_data:
    # split_text = i.split(" ")
    split_text = i.rstrip("\n").rstrip().split(" ")

    # print(split_text)
    if len(split_text) < 7:
        pass
    elif len(split_text) == 7:
        data.append(split_text)
        # print(split_text)
    else:
        data.append(split_text[0:7])
        # print(split_text)



# model size 64
model = ft.load('C:/Users/paul/Desktop/NLP/embedding642/ft_model64_2')
# change sentence to vector stack


sentence_to_word_vec = np.zeros(shape=(len(data), 7, 64))
for sentence_index, i in enumerate(data):
    temp_list = np.zeros(shape=(7, 64))
    for idx, j in enumerate(i):
        try:
            temp_list[idx] = np.array([model.wv.get_vector(j)])
        except:
            temp_list[idx] = np.array([model.wv.get_vector("<unk>")])
    temp_list = np.reshape(temp_list, (1, 7, 64))
    sentence_to_word_vec[sentence_index] = temp_list

# add one dimension
sentence_to_word_vec = np.expand_dims(sentence_to_word_vec, axis=-1)


class WGAN2vec():
    def __init__(self):
        self.img_rows = 7
        self.img_cols = 64
        self.channels = 1
        self.sentence_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)
        # optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-8)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        sentence = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(sentence)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        sentence1 = Dense(448)(noise)
        sentence1 = Reshape((7,64,1))(sentence1)
        sentence1 = Residual(sentence1,64)
        sentence1 = Residual(sentence1,1)
        sentence1 = LeakyReLU(alpha=0.2)(sentence1)

        model = Model(noise, sentence1)
        model.summary()
        model.save("WGAN2vec_generator.h5")

        return Model(noise, sentence1)

    def build_critic(self):
        img = Input(shape=self.sentence_shape)
        validity = Residual(img,64)
        validity = Residual(validity,128)
        validity = Residual(validity,1)
        validity = Flatten()(validity)
        validity = Dense(1, activation='sigmoid')(validity)
        model = Model(img, validity)
        model.summary()
        model.save("WGAN2vec_discriminator.h5")
        return Model(img, validity)

    def pretrain_D(self, epochs, batch_size=128):
        X_train = sentence_to_word_vec
        valid = -np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1
        print("pretraining D")
        for epoch in range(epochs):
            print("{}epochs".format(epoch))
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                # (batch size, sentence_row, sentence_col, sentecne channel)
                # (128,7,64,1)
                sentences = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_sentences = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(sentences, valid)
                d_loss_fake = self.critic.train_on_batch(gen_sentences, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = sentence_to_word_vec

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                sentences = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_sentences = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(sentences, valid)
                d_loss_fake = self.critic.train_on_batch(gen_sentences, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)
            epoch_list.append(epoch)
            plt_D_loss.append(1 - d_loss[0])
            plt_G_loss.append(1 - g_loss[0])
            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.show_sentence(epoch)

    def show_sentence(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = self.generator.predict(noise)
        test = np.squeeze(gen_sentence)
        corpus = []
        for i in test:
            sentence = ""
            for j in i:
                # temp = model.wv.similar_by_vector(j)
                temp = model.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "
            print(sentence)
            # w.write(sentence)
            # w.write("\n")

        # fasttext vector update
        model.build_vocab(sentences=corpus, update=True, min_count=5)
        model.train(sentences=corpus, epochs=model.epochs, total_examples=model.corpus_count,
                    total_words=model.corpus_total_words)

    def predict(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = self.generator.predict(noise)
        test = np.squeeze(gen_sentence)
        sentence_list = []
        for i in test:
            sentence = ""
            for j in i:
                # temp = model.wv.similar_by_vector(j)
                temp = model.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "
            sentence_list.append(sentence)
        return sentence


if __name__ == '__main__':
    wgan2vec = WGAN2vec()
    wgan2vec.pretrain_D(epochs=100)
    wgan2vec.train(epochs=4000, batch_size=64, sample_interval=50)
    for i in range(100):
        t = wgan2vec.predict()
        print(t)

    # model.save('ft_model64_3')
    # w.close()

    plt.title("loss graph")
    plt.plot(epoch_list, plt_G_loss)
    plt.plot(epoch_list, plt_D_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['G loss', 'D loss'])
    plt.show()
    # plt.savefig('GAN_loss.png')
