#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
ALL_RND_SEED = 1
random.seed(ALL_RND_SEED)

import numpy as np

np.random.seed(ALL_RND_SEED)


# In[2]:


from htmd.ui import *
from htmd.adaptive.adaptivebandit import AdaptiveBandit

from glob import glob
import os
import shutil


# In[3]:


# load file generators
for file in glob('./generators/*/run.sh'):
    os.chmod(file, 0o755)

os.makedirs('./adaptivemd', exist_ok=True)
os.makedirs('./adaptivegoal', exist_ok=True)
shutil.copytree('./generators', './adaptivemd/generators')
shutil.copytree('./generators', './adaptivegoal/generators')

os.chdir('./adaptivemd')
shutil.copy('../structure.pdb', '.')


# In[4]:


queue = LocalGPUQueue()
queue.datadir = './data'
ab = AdaptiveBandit()
ab.coorname = 'input.pdb'
ab.updateperiod = 60
ab.app = queue
ab.filter = False
ab.nmin = 0
ab.nmax = 1
#ab.nframes = 40
ab.clustmethod = MiniBatchKMeans
ab.ticadim = 2
ab.ticalag = 5
#ab.lag = 50 # 20 ps, 0.4ps per step
# Stefano says to not touch
#ab.macronum = 6
# metric
# protsel = 'protein'
#ab.projection = MetricSelfDistance(protsel)
ab.exploration = 0.01
ab.actionspace = "metric"
#ab.projection = MetricSelfDistance(protsel)
mol = Molecule('./structure.pdb')
met = MetricDihedral()
#met.fstep = 0.0004


# In[5]:


ab.projection = met
ab.nepochs = 500


# In[ ]:


ab.run()


# In[ ]:


get_ipython().system('pwd')


# In[ ]:




