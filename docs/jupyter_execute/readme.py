#!/usr/bin/env python
# coding: utf-8

# In[1]:


from latentcor import gen_data, get_tps, latentcor


# In[2]:


simdata = gen_data(n = 100, tps = ["ter", "con"])
print(simdata[0][ : 6, : ])


# In[3]:


estimate = latentcor(simdata[0], tps = ["ter", "con"])
print(estimate[0])

