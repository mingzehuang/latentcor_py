#!/usr/bin/env python
# coding: utf-8

# In[1]:


from latentcor import gen_data, latentcor


# In[2]:


print(gen_data(n = 6, tps = "con")[0])


# In[3]:


print(gen_data(n = 6, tps = "bin")[0])


# In[4]:


print(gen_data(n = 6, tps = "ter")[0])


# In[5]:


print(gen_data(n = 6, tps = "tru")[0])


# In[6]:


X = gen_data(n = 100, tps = ["con", "bin", "ter", "tru"])[0]
print(X[ :6, : ])


# In[7]:


K = latentcor(X, tps = ["con", "bin", "ter", "tru"])[3]
print(K)


# In[8]:


estimate_original = latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "original", tol = 1e-8)


# In[9]:


print(estimate_original[3])


# In[10]:


print(estimate_original[4])


# In[11]:


print(estimate_original[1])


# In[12]:


estimate_approx = latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx")
print(estimate_approx[1])


# In[13]:


print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx", ratio = 0.99)[0])
print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx", ratio = 0.4)[0])
print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "original")[0])


# In[14]:


X = gen_data(n = 6, tps = ["con", "bin", "ter", "tru"])[0]
print(latentcor(X, tps = ["con", "bin", "ter", "tru"])[1])


# In[15]:


print(latentcor(X, tps = ["con", "bin", "ter", "tru"])[0])


# In[16]:


print(latentcor(X, tps = ["con", "bin", "ter", "tru"], nu = 0.001)[0])


# In[17]:


X = gen_data(n = 100, tps = ["con", "bin", "ter", "tru"])[0]
out = latentcor(X, tps = ["con", "bin", "ter", "tru"], nu = 0.001)
print(out[1])
print(out[0])

