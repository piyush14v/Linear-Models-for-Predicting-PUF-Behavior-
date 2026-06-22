# %%
import numpy as np
import submitnew
from submitnew import my_map
from submitnew import my_fit
from submitnew import my_decode

import time as tm

# %%
import os
print(os.getcwd())

# %%
Z_trn = np.loadtxt( "secret_trn.txt" )
Z_tst = np.loadtxt( "secret_tst.txt" )


# %%

def get_model( p, q, r, s ):
  p = np.maximum( p, 0 )
  q = np.maximum( q, 0 )
  r = np.maximum( r, 0 )
  s = np.maximum( s, 0 )
  d = p - q
  c = r - s
  alpha = ( d + c ) / 2
  beta = ( d - c ) / 2
  w = np.zeros( ( len( alpha ) + 1, )  )
  w[:-1] += alpha
  w[1:] += beta
  return w

# %% [markdown]
# 

# %%

for i in [0.01, 0.1,1,10,100]:
  total_train = 0
  for itt in range(5):
    n_trials = 5

    d_size = 0
    t_train = 0
    t_map = 0
    acc = 0
    for t in range( n_trials ):
      tic = tm.perf_counter()
      w, b = my_fit( Z_trn[:, :-1], Z_trn[:,-1] , c=i)
      toc = tm.perf_counter()

      t_train += toc - tic
      w = w.reshape( -1 )

      d_size += w.shape[0]

      tic = tm.perf_counter()
      feat = my_map( Z_tst[:, :-1] )
      toc = tm.perf_counter()
      t_map += toc - tic

      scores = feat.dot( w ) + b

      pred = np.zeros_like( scores )
      pred[ scores > 0 ] = 1

      acc += np.average( Z_tst[ :, -1 ] == pred )
    d_size /= n_trials
    t_train /= n_trials
    t_map /= n_trials
    acc /= n_trials
    W = np.loadtxt( "secret_mod.txt" )
    ( n_models, dims ) = W.shape
    t_decode = 0
    m_dist = 0
    for t in range( n_trials ):
      for itr in range( n_models ):
        w = W[ itr, : ]
        tic = tm.perf_counter()
        p_hat, q_hat, r_hat, s_hat = my_decode( w )
        toc = tm.perf_counter()
        t_decode += toc - tic
        w_hat = get_model( p_hat, q_hat, r_hat, s_hat )
        m_dist += np.linalg.norm( w - w_hat )
    t_decode /= ( n_trials * n_models )
    m_dist /= ( n_trials * n_models )
    print( f"{i},{d_size},{t_train},{1 - acc}" )
    total_train+=t_train
  total_train/=5
  print(f"for c={i},avg train time={total_train},{1 - acc}")

# %% [markdown]
# 

# %%


