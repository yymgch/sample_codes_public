# Lyapunov指数の計算を，Keras Model として実装する．
#%%
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')
mydtype = tf.float64
#%% Henon cell


class HenonCell(keras.layers.Layer):
  '''
  This class calculates one step of Henon-map,
  used as a RNN cell of keras RNN.
  For details of customizing RNN, 
  see  https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
  using several matrix operation for faster execution
  x_new = 1 - a * x^2 +y
  y_new = b x
  '''

  def __init__(self, a=1.2, b=0.4, **kwargs):

    # parameter
    self.a = tf.Variable(a, dtype=mydtype)
    self.b = tf.Variable(b, dtype=mydtype)
    # RNN custom cell must have state_size and output_size
    self.state_size = 2  # 2-dimension
    self.output_size = 2
    # extract one variable, or move one variable to another
    # leave x value in x place
    self.x_in_x = tf.constant([[1, 0], [0, 0]], dtype=mydtype)
    # move  x value into y place
    self.x_in_y = tf.constant([[0, 1], [0, 0]], dtype=mydtype)
    # move y value into x place
    self.y_in_x = tf.constant([[0, 0], [1, 0]], dtype=mydtype)
    # leave y value in y place
    self.y_in_y = tf.constant([[0, 0], [0, 1]], dtype=mydtype)
    self.one_in_x = tf.constant([1, 0], dtype=mydtype)

    super(HenonCell, self).__init__(**kwargs)

  def build(self, input_shape):
    self.batch_size = input_shape[0]

  def call(self, inputs, states):
    ''' states is a tuple, which only has a 2-dim array of the shape (batch_size, 2).
       inputs are ignored.
    '''
    #print(type(states)) # tuple
    #print(states[0].shape) # (2,)
    x_x = states[0] @ self.x_in_x
    y_x = states[0] @ self.y_in_x
    x_y = states[0] @ self.x_in_y

    x_new = self.one_in_x - self.a * x_x * x_x + y_x
    x_new += self.b * x_y

    return x_new, [x_new]  # return output and states.


# Jacobian Layer:

class Jacobian(layers.Layer):

  def __init__(self, func,  **kwargs):
    ''' input-output shapes of func should be (batch, dim) -> (batch, dim)
    '''
    self.func = func
    super(Jacobian, self).__init__(**kwargs)

  def call(self, X, t_length=None):
    # x have (batch, timestep, dim)
    if t_length == None:
      t_length = X.shape[1]
    batch_size = X.shape[0]

    X = tf.reshape(X, [batch_size*t_length, X.shape[2]])
    with tf.GradientTape() as tape:
      tape.watch(X)
      X_next, _ = self.func(None, [X])

    Jxs = tape.batch_jacobian(X_next, X)

    Jxs = tf.reshape(Jxs, [batch_size, t_length, X.shape[1], X.shape[1]])
    return Jxs

# QR cell layer


class QRDcell(layers.Layer):
  '''
  performing successive QR decomposition.
  This class can be used as a RNN cell in Keras RNN Layer
  '''

  def __init__(self, dim=2, **kwargs):
    super(QRDcell, self).__init__(**kwargs)
    self.dim = dim
    # d x d dimension (d is a dimension of dynamical systems)
    self.state_size = tf.constant([dim, dim])
    self.output_size = tf.constant(dim)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=mydtype):
    ''' return identity matrices'''

    return tf.linalg.eye(self.dim, self.dim, batch_shape=[batch_size], dtype=mydtype)

  def call(self, inputs, states):
    # inputs is  J_{n} (batch, dim, dim)
    # states is Q_{n-1} (batch, dim,dim). Q_{0} is identity matrix
    # Q_{n}R_{n} = J_{n}Q_{n-1}
    # Q_{n} is the next state. (Q_new)
    # R_{n} is the output. (R_new)

    J = inputs
    Q = states[0]
    Q_new, R_new = tf.linalg.qr(J@Q)
    return R_new, [Q_new]



def test_each_classes():
  ''' shapeをチェックして，それぞれのクラスが動いてるかテストする
  '''
  t_length = 1000
  batch_size = 10  
  dim = 2

  g1 = tf.random.Generator.from_seed(1)

  #### Henon cell###
  hc = HenonCell(a=1.4, b=0.3)

  x0 = 0.1* g1.normal(shape=(batch_size, dim), dtype=tf.float64)

  x_new, new_state = hc(None, [x0])  # one-step
  
  assert x_new.shape == (batch_size, dim)
  assert new_state[0].shape == (batch_size,dim)

  ### Henon rnn###
  henon_rnn = layers.RNN(hc, return_sequences=True)
  # henon_rnn.build(input_shape=[[batch_size, None, 2]])  
  #making dummy input, which determine sequence length.
  dummy_input = tf.zeros(shape=[batch_size, t_length, dim], dtype=mydtype)

  #initial state
  x0 = 0.1* g1.normal(shape=(batch_size, dim), dtype=tf.float64)  

  # run
  st = time.time()
  X = henon_rnn(dummy_input, initial_state=x0)
  et = time.time()
  print(f'time: {et-st}')
  assert X.shape ==(batch_size, t_length,dim)
  plt.plot(X[0, :, 0], X[0, :, 1], '.', markersize=1)

  ### Jacobian ###

  # %% Calculating Jacobian onestep
  hc = HenonCell(a=1.4, b=0.3)
  x0 = 0.1* g1.normal(shape=(batch_size, dim), dtype=tf.float64)  

  with tf.GradientTape() as tape:
    tape.watch(x0)
    x_new, _ = hc(None, [x0])  # one-step

  J0 = tape.batch_jacobian(x_new, x0)
  assert J0.shape ==(batch_size,dim,dim)
  print('J0=')
  print(J0[0, :, :])

  #  applied to sequence
  henon_rnn = layers.RNN(hc, return_sequences=True)
  dummy_input = tf.zeros(shape=[batch_size, t_length, dim], dtype=mydtype)
  X = henon_rnn(dummy_input, initial_state=x0)

  #reshape (bs, t_length,2) -> (bs*t_length,2)
  X = tf.reshape(X, [-1, 2])  # (batch_size*t_length, 2)
  st = time.time()

  with tf.GradientTape() as tape:
    tape.watch(X)
    X_next, _ = hc(None, [X])
  assert X_next.shape == (batch_size*t_length, dim)
  
  Js = tape.batch_jacobian(X_next, X)
  Js = tf.reshape(Js, [batch_size, t_length,dim,dim ])
  assert Js.shape == (batch_size, t_length, dim, dim)  
  print('J0=')
  print(Js.numpy()[0,0, :, :])
  et = time.time()
  print(f'time for calculating jacobian: {et-st}')
  
  ### Jacobian Layer ###

  hc = HenonCell(a=1.4, b=0.3)
  X = g1.normal(shape=(batch_size, t_length, dim), dtype=mydtype)
  jl = Jacobian(hc)
  Jx = jl(X)
  print(f'Jx.shape={Jx.shape}')
  assert Jx.shape == (batch_size, t_length, dim, dim)
  
  ### QR Decomposition cell

  qrdc = QRDcell(dim=dim)

  js = g1.normal(shape=(batch_size, dim, dim) ) 

  r, [q] = qrdc(js, [tf.eye(dim, dtype=mydtype)])
  assert r.shape == [batch_size, dim, dim]
  assert q.shape == [batch_size, dim, dim]


  ###  taking diagonalpart, then its absolute values and finally its logarithm ###

  log_diag_r = tf.math.log(tf.math.abs(tf.linalg.diag_part(r)))
  assert log_diag_r.shape == [batch_size, dim]

  ### qr-rnn test ###

  js = g1.normal(shape=(batch_size, t_length, dim, dim), dtype=mydtype)

  qrd_rnn = layers.RNN(QRDcell(dim=dim), return_sequences=True)

  rs = qrd_rnn(js)

  assert rs.shape == [batch_size, t_length, dim, dim]

# %%

if __name__ == '__main__':

  # GPU memory の設定
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
      tf.config.experimental.set_memory_growth(physical_devices[k], True)
      print('memory growth:', tf.config.experimental.get_memory_growth(
          physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")


#%% test each classes
  test_each_classes()

#%% making Lyapunov-Exponent calculator model by using Functional API of Keras

  # parameters
  batch_size = tf.constant(10, dtype=tf.int32)
  dim = tf.constant(2, dtype=tf.int32)
  t_length = tf.constant(10000, dtype=tf.int32)

# %% defining layers and inputs


  hc = HenonCell(a=1.4, b=0.3)
  henon_rnn = layers.RNN(hc, return_sequences=True)
  # henon_rnn.build(input_shape=[[batch_size, None, 2]])
  jl = Jacobian(hc)
  qrd_rnn = layers.RNN(QRDcell(), return_sequences=True)

  ## inputs (initial conditions and time length)
  input_initial_condition = keras.Input(shape=(dim,), dtype=mydtype) # input object for keras model
  tlength_inp = keras.Input(batch_shape=(1,), dtype=tf.int32) # passing model length of timesteps
  
#%% tracing forward calculation 

  # this tensor depends on tlength_inp
  dummy_input = tf.zeros(
      shape=[batch_size, tlength_inp[0], dim], dtype=mydtype)

  X = henon_rnn(dummy_input, initial_state=input_initial_condition) # 軌道の計算

  js = jl(X, t_length=tlength_inp[0]) # reshapeするときに必要なため，明示的にt_length_inp[0] を渡す．(これはt_lengthが決まっているtensorのときは必要ない)
  rs = qrd_rnn(js)  # qr分解しながらRを集める．
  log_diag_r = tf.math.log(tf.math.abs(
      tf.linalg.diag_part(rs)))  # Rの対角をとって絶対値のlog
  m_log_r = tf.reduce_mean(log_diag_r, axis=1)  # 平均
#%% functional APIによりモデル作成
  # 軌道計算するモデル
  gen_trj = keras.models.Model(inputs=[input_initial_condition, tlength_inp], outputs=X)
  # リアプノフ指数の計算までするモデル
  est_lyap = keras.models.Model(inputs=[input_initial_condition, tlength_inp], outputs=m_log_r)

  # 高速化する関数
  @tf.function
  def calc_lyap(x0, t_length):
    return est_lyap([x0, t_length])


# %% 動かす．
  g1 = tf.random.Generator.from_non_deterministic_state()
  x0 = 0.1*g1.normal(shape=(batch_size, dim), dtype=mydtype)
  X = gen_trj([x0,t_length])
  lyaps_batch = est_lyap([x0,t_length])
#%% 関数使って高速化
  g1 = tf.random.Generator.from_non_deterministic_state()
  x0 = 0.1*g1.normal(shape=(batch_size, dim), dtype=mydtype)

  lyaps_batch = calc_lyap(x0, t_length)

  lyap_ave = tf.reduce_mean(lyaps_batch, axis=0)
  lyap_std = tf.math.reduce_std(lyaps_batch, axis=0)  
  print(f'Estimation of lyapnov exponents (average over different trajectories +- std.):')
  for i,(av,st) in enumerate(zip(lyap_ave, lyap_std)):
    print(f'{i}: {av.numpy()} +- {st.numpy()} ')

