import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image

my_bar = st.progress(0)

count_p = 0
my_bar.progress(count_p *4)

#st.write('Hello, *World!* :sunglasses:')

NBATCH = 250  # size of mini-batch
#NBATCH = 500  # size of mini-batch
#NX = 28  # image width
#NY = 28  # image height
NX = 16  # image width
NY = 16  # image height
NC = 1   # number of channels (1=monochrome)
NZ = 32  # VAE: size of latent variable
SCALE = 6  # CPPN: zooming scale

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.random.rand(60000,NX,NY)
y_train = np.random.rand(60000,)
x_test = np.random.rand(10000,NX,NY)
y_test = np.random.rand(10000,)

x_train = x_train.reshape(-1, NY, NX, NC).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, NY, NX, NC).astype(np.float32) / 255.0

NSAMPLE = x_train.shape[0]
x_batches = tf.data.Dataset.from_tensor_slices(x_train).shuffle(NSAMPLE).batch(NBATCH)
#print(x_train.shape, x_test.shape)
count_p +=1
my_bar.progress(count_p *4)

def find_x(digit):
  ''' find testing data by digit '''
  i = -1
  while i == -1 or y_test[i] != digit:
    i = np.random.randint(y_test.shape[0])
  return x_test[i:i+1]
count_p +=1
my_bar.progress(count_p *4)

def create_coordinates(nx=NX, ny=NY, scale=SCALE, nbatch=NBATCH):
  n = (nx + ny) / 2
  nx2, ny2 = nx/n*scale, ny/n*scale
  xs, ys = np.meshgrid(np.linspace(-nx2, nx2, nx), np.linspace(-ny2, ny2, ny))
  rs = np.sqrt(xs**2 + ys**2)

  xs_repeat = np.tile(np.reshape(xs, (1, nx*ny, 1)), (nbatch, 1, 1))
  ys_repeat = np.tile(np.reshape(ys, (1, nx*ny, 1)), (nbatch, 1, 1))
  rs_repeat = np.tile(np.reshape(rs, (1, nx*ny, 1)), (nbatch, 1, 1))
  coords = np.concatenate((xs_repeat, ys_repeat, rs_repeat), axis=-1).astype(np.float32)
  return coords, xs, ys, rs
count_p +=1
my_bar.progress(count_p *4)

coords, xs, ys, rs = create_coordinates()

coords_sample, _, _, _ = create_coordinates(nbatch=5)

#print(coords.shape)


def model_vae_encoder(name='VAE-Q', nodes=32):
  return tf.keras.Sequential([
      layers.InputLayer(input_shape=(NY, NX, NC)),
      layers.Flatten(),
      layers.Dense(nodes, activation='relu'),
      layers.Dense(nodes, activation='relu'),
      layers.Dense(NZ + NZ),
  ], name=name)
count_p +=1
my_bar.progress(count_p *4)
  
def model_vae_decoder(name='VAE-P', nodes=32):
  return tf.keras.Sequential([
      layers.InputLayer(input_shape=(NZ,)),
      layers.Dense(nodes, activation='relu'),
      layers.Dense(nodes, activation='relu'),
      layers.Dense(NX*NY*NC, activation='sigmoid'),
      layers.Reshape([NY, NX, NC,]),
  ], name=name)
count_p +=1
my_bar.progress(count_p *4)


def model_dcign_encoder(name='DCIGN-Q', nodes=32):
  return tf.keras.Sequential([
      layers.InputLayer(input_shape=(NY, NX, NC)),
      layers.Conv2D(nodes*1, (3,3), strides=(2,2), padding='same', activation='relu'),
      layers.Conv2D(nodes*2, (3,3), strides=(2,2), padding='same', activation='relu'),
      layers.Flatten(),
      layers.Dense(NZ + NZ),
  ], name=name)
count_p +=1
my_bar.progress(count_p *4)
  
def model_dcign_decoder(name='DCIGN-P', nodes=32):
  return tf.keras.Sequential([
      layers.InputLayer(input_shape=(NZ,)),
      layers.Dense(7*7*nodes, activation='relu'),
      layers.Reshape((7, 7, nodes)),
      layers.Conv2DTranspose(nodes*2, (3,3), strides=(2,2), padding='same', activation='relu'),
      layers.Conv2DTranspose(nodes*1, (3,3), strides=(2,2), padding='same', activation='relu'),
      layers.Conv2DTranspose(NC, (3,3), strides=(1,1), padding='same'),
  ], name=name)
count_p +=1
my_bar.progress(count_p *4)

def param_split(z_params):
  return tf.split(z_params, 2, axis=1)
count_p +=1
my_bar.progress(count_p *4)

def sample_func(z_params, mean=0, stddev=1):
  z_mean, z_logvar = param_split(z_params)
  eps = tf.random.normal(shape=tf.shape(z_mean), mean=mean, stddev=stddev)
  return z_mean + tf.exp(z_logvar * 0.5) * eps
count_p +=1
my_bar.progress(count_p *4)

def sample(z_params, mean=0, stddev=1):
  return layers.Lambda(sample_func)(z_params)
count_p +=1
my_bar.progress(count_p *4)

def repeat_vector(inputs):
  vec_in, dim_in = inputs
  return layers.RepeatVector(K.shape(dim_in)[1])(vec_in)
count_p +=1
my_bar.progress(count_p *4)

def model_cppn_generator(name='CPPN-G', levels=4, nodes=32, stddev=1):
  normal_init = tf.keras.initializers.RandomNormal(stddev=stddev)
  inits = {'kernel_initializer':normal_init, 'bias_initializer':normal_init}

  z_in = layers.Input(shape=(NZ,))
  coord_in = layers.Input(shape=(None, 3))
  h = layers.Lambda(repeat_vector, output_shape=(None, NZ))([z_in, coord_in])
  h = layers.Concatenate()([h, coord_in])
  h = layers.Dense(nodes, activation='softplus', **inits)(h)
  for i in range(levels):
    h = layers.Dense(nodes, activation='tanh', **inits)(h)
  h = layers.Dense(NC, activation='sigmoid', **inits)(h)
  x_out = layers.Flatten()(h)
  return tf.keras.Model(inputs=[z_in, coord_in], outputs=x_out, name=name)
count_p +=1
my_bar.progress(count_p *4)

def sq(x, nx=NX, ny=NY, nc=NC):
  return tf.reshape(x, (-1, ny, nx, nc))
count_p +=1
my_bar.progress(count_p *4)

def model_dcgan_generator(name='DCGAN-G', filters=32, stddev=0.02):
#   inits = {'kernel_initializer':tf.keras.initializers.RandomNormal(stddev=stddev)}
  inits = {}

  model = tf.keras.Sequential(name=name)
  model.add(layers.InputLayer(input_shape=(NZ,)))
  model.add(layers.Dense(7*7*filters*4, **inits))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Reshape((7, 7, filters*4)))
  for f, s in zip([2,1], [1,2]):
    model.add(layers.Conv2DTranspose(filters*f, (5,5), strides=(s,s), padding='same', **inits))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  model.add(layers.Conv2DTranspose(NC, (5,5), strides=(2,2), padding='same', **inits, activation='tanh'))
  assert model.output_shape == (None, NY, NX, NC)
  return model
count_p +=1
my_bar.progress(count_p *4)

def model_dcgan_discriminator(name='DCGAN-D', filters=32, stddev=0.02):
#   inits = {'kernel_initializer':tf.keras.initializers.RandomNormal(stddev=stddev)}
  inits = {}

  model = tf.keras.Sequential(name=name)
  model.add(layers.InputLayer(input_shape=(NY, NX, NC)))
  for f in [1,2]:
    model.add(layers.Conv2D(filters*f, (5,5), strides=(2,2), padding='same', **inits))
    if f>1: model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
  model.add(layers.Flatten())
  model.add(layers.Dense(1, **inits))  # no need activation='sigmoid' if using binary_crossentropy(from_logits=True) -- https://stackoverflow.com/questions/45741878/using-binary-crossentropy-loss-in-keras-tensorflow-backend
  return model
count_p +=1
my_bar.progress(count_p *4)


def kld(z_mean, z_logvar):
  return -0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=1) / (NX*NY*NC)
count_p +=1
my_bar.progress(count_p *4)

def bce(y1, y2):
  y1_flat = tf.reshape(y1, (-1, NX*NY*NC))
  y2_flat = tf.reshape(y2, (-1, NX*NY*NC))
  return tf.keras.losses.binary_crossentropy(y1_flat, y2_flat)
count_p +=1
my_bar.progress(count_p *4)
#   return -K.sum(y1_flat * K.log(1e-10 + y2_flat) + (1-y1_flat) * K.log(1e-10 + 1 - y2_flat), axis=1) / (NX*NY*NC)

def bce_logits(b, y):
  truth = tf.ones_like(y) if b else tf.zeros_like(y)
  return tf.keras.losses.binary_crossentropy(truth, y, from_logits=True)
count_p +=1
my_bar.progress(count_p *4)



q = model_vae_encoder()
# g = model_vae_decoder(); CPPN = False
g = model_cppn_generator(levels=3, nodes=32, stddev=1); CPPN = True
d = model_dcgan_discriminator()

q_optimizer = tf.keras.optimizers.Adam(1e-3)
g_optimizer = tf.keras.optimizers.Adam(1e-3)
d_optimizer = tf.keras.optimizers.Adam(1e-4)
q_vars = q.trainable_variables
g_vars = g.trainable_variables
d_vars = d.trainable_variables

all_q_loss = []
all_g_loss = []
all_d_loss = []

#@tf.function
def train_batch_vaegan(x_batch, q_loop=1, g_loop=1, 
                       g_always=True, d_always=True, th_high=0.75, th_low=0.6):
  with tf.device('/gpu:0'):
    g_updated = d_updated = False
    for i in range(q_loop):
      with tf.GradientTape() as vae_tape:
        z_params = q(x_batch, training=True)
        z_mean, z_logvar = param_split(z_params)
        z = sample(z_params)
        if CPPN:
          x_gen = sq(g([z, coords], training=True))
        else:
          x_gen = g(z, training=True)

        q_loss = K.mean(bce(x_batch, x_gen) + kld(z_mean, z_logvar))

      q_grads = vae_tape.gradient(q_loss, q_vars + g_vars)
      q_optimizer.apply_gradients(zip(q_grads, q_vars + g_vars))

    for i in range(g_loop):
      with tf.GradientTape(persistent=True) as gan_tape:
        z_params = q(x_batch, training=True)
        z = sample(z_params)
        if CPPN:
          x_gen = sq(g([z, coords], training=True))
        else:
          x_gen = g(z, training=True)
        d_gen = d(x_gen, training=True)
        d_batch = d(x_batch, training=True)

        g_loss = K.mean(bce_logits(True, d_gen))
        d_loss = K.mean(bce_logits(True, d_batch))/2 + K.mean(bce_logits(False, d_gen))/2

      if tf.logical_or(g_always, g_loss > th_low):
        g_grads = gan_tape.gradient(g_loss, q_vars + g_vars)
        g_optimizer.apply_gradients(zip(g_grads, q_vars + g_vars))
        g_updated = True

      if i == g_loop-1:
        if tf.logical_or(d_always, tf.logical_and(g_loss < th_high, d_loss > th_low)):
          d_grads = gan_tape.gradient(d_loss, d_vars)
          d_optimizer.apply_gradients(zip(d_grads, d_vars))
          d_updated = True

      del gan_tape  # explicit delete tape if persistent=True

  return q_loss, g_loss, d_loss, g_updated, d_updated
count_p +=1
my_bar.progress(count_p *4)


XL = 50  # enlarge factor
#XL = 5  # enlarge factor
POS = 183  # test data position
#POS = 83  # test data position
choice = 3 # 1=reconstruct, 2=random latent, 3=colored
x = x_test[POS:POS+1]
# x = find_x(0)
coords_XL, _, _, _ = create_coordinates(nx=NX*XL, ny=NY*XL, nbatch=1)

if choice in [1]:
  z = sample_func(q(x), stddev=1)
  x2 = sq(g([z, coords_XL]), nx=NX*XL, ny=NY*XL)
elif choice in [2]:
  noise_z = tf.random.normal((1, NZ), stddev=1.5)
  x2 = sq(g([noise_z, coords_XL]), nx=NX*XL, ny=NY*XL)
elif choice in [3]:
  x2 = np.zeros((NY*XL, NX*XL, 3))
  for i in [0,1,2]:
    z = sample_func(q(x), stddev=1.2)
    x2c = sq(g([z, coords_XL]), nx=NX*XL, ny=NY*XL).numpy()
    x2[:,:,i] = x2c[0,:,:,0]
count_p +=1
my_bar.progress(count_p *4)

plt.figure(figsize=(8,12))
if choice in [1,2]:
  plt.imshow(tf.reshape(x2[0], (NY*XL, NX*XL)), vmin=0, vmax=1, cmap='gray')
elif choice in [3]:
  plt.imshow(x2, vmin=0, vmax=1)
plt.axis('off')
#plt.show()

#plt.plot([1,2,3], [5,7,4])
plt.savefig("2.png")
count_p +=1
my_bar.progress(count_p *4)

#st.write(count_p)

#with st.spinner('Wait for it...'):
#	time.sleep(5)
#	st.success('Done!')

image = Image.open('2.png')
#st.image(image, caption='Sunrise by the mountains', use_column_width=True)
st.image(image, use_column_width=True)
st.balloons()



#import tkinter
#import matplotlib
#matplotlib.use('TkAgg')
#plt.plot([1,2,3],[5,7,4])
#plt.show()
