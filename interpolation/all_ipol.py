import os
import sys
"""sys.path.append('/scratch/user/sharkmanhmz/latentcor_py/latentcor')"""
"""sys.path.insert(0, os.path.abspath('../latentcor'))"""
import pickle
import lzma


with lzma.open(os.path.join(os.getcwd(), "ipol_10.xz"), "rb") as f:
    ipol_10 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_11.xz"), "rb") as f:
    ipol_11 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_20.xz"), "rb") as f:
    ipol_20 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_21.xz"), "rb") as f:
    ipol_21 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_22.xz"), "rb") as f:
    ipol_22 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_30.xz"), "rb") as f:
    ipol_30 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_31.xz"), "rb") as f:
    ipol_31 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_32.xz"), "rb") as f:
    ipol_32 = pickle.load(f)
with lzma.open(os.path.join(os.getcwd(), "ipol_33.xz"), "rb") as f:
    ipol_33 = pickle.load(f)

all_ipol = (ipol_10, ipol_11, ipol_20, ipol_21, ipol_22, ipol_30, ipol_31, ipol_32, ipol_33)

with lzma.open(os.path.join(os.getcwd(), "all_ipol.xz"), "wb", preset = 9) as f:
    pickle.dump(all_ipol, f)