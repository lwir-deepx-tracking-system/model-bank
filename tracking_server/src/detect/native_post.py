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

if hasattr(lib, 'nms_person'):
 lib.nms_person.argtypes=[arr,arr,ctypes.c_int];lib.nms_person.restype=ctypes.c_int

 def fast_nms_person(y):
  n=lib.nms_person(y,_nmsout,y.shape[2]);assert 0<=n<=300
  return _nmsout[:n].copy()


if hasattr(lib, 'prepare_split_heads') and hasattr(lib, 'decode_split_heads'):
    head_pointers = ctypes.c_void_p * 18
    lib.prepare_split_heads.argtypes = [head_pointers, arr]
    lib.prepare_split_heads.restype = ctypes.c_int
    lib.decode_split_heads.argtypes = [head_pointers, arr, ctypes.c_float, arr, arr]
    lib.decode_split_heads.restype = ctypes.c_int

    def prepare_split_heads(heads, logits):
        pointers = head_pointers(*(head.ctypes.data for head in heads))
        invalid = lib.prepare_split_heads(pointers, logits)
        return pointers, invalid

    def decode_split_heads(pointers, logits, y_offset):
        # Use the same OpenCV float32 exp as the ordinary detector scores.
        exp_scores = cv2.exp(logits)
        rows = np.empty((300, 6), np.float32)
        points = np.empty((300, 17, 3), np.float32)
        count = lib.decode_split_heads(pointers, exp_scores, y_offset, rows, points)
        return None if count < 0 else (rows[:count], points[:count])
