import ctypes
from pathlib import Path
import numpy as np,cv2
lib=ctypes.CDLL(str(Path(__file__).parent/'post.so'))
arr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
lib.decode_raw.argtypes=[arr,arr,arr,ctypes.c_int,ctypes.c_int];lib.decode_raw.restype=None
lib.nms_decoded.argtypes=[arr,arr,ctypes.c_int];lib.nms_decoded.restype=ctypes.c_int
class Decoder:
 def __init__(self,h,w):
  assert h==640 and w in (384,640);self.h=h;self.w=w;self.n=sum((h//s)*(w//s) for s in (8,16,32));self.out=np.empty((1,20,self.n),np.float32);self.scoreout=self.out[:,4:].reshape(16,self.n)
 def __call__(self,raw):
  lib.decode_raw(raw[0],raw[1],self.out,self.h,self.w)
  np.negative(raw[1].reshape(16,self.n),out=self.scoreout);cv2.exp(self.scoreout,self.scoreout);np.add(self.scoreout,np.float32(1),out=self.scoreout);np.reciprocal(self.scoreout,out=self.scoreout)
  return self.out
_nmsout=np.empty((300,6),np.float32)
def fast_nms(y):
 n=lib.nms_decoded(y,_nmsout,y.shape[2]);assert 0<=n<=300
 return _nmsout[:n].copy()
