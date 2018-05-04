from scipy.linalg.blas import sgemm
from scipy.linalg.blas import dgemm
import numpy as np
import time
import sys

arr_int64 = np.random.randint(0,128,(128,56000))
print("int64:\t",sys.getsizeof(arr_int64)) # 判断变量占用的空间大小

arr_int32 = arr_int64.astype(np.int32)
print("int32:\t",sys.getsizeof(arr_int32))

arr_int16 = arr_int64.astype(np.int16)
print("int16:\t",sys.getsizeof(arr_int16))

arr_int8 = arr_int64.astype(np.int8)
print("int8:\t",sys.getsizeof(arr_int8))

arr_uint8 = arr_int64.astype(np.uint8)
print("uint8:\t",sys.getsizeof(arr_uint8))

arr_float64 = arr_int64.astype(np.float64)#+np.random.rand(2000,1000)
print("float64:\t",sys.getsizeof(arr_float64))


arr_float32 = arr_int64.astype(np.float32)
print("float32:\t",sys.getsizeof(arr_float32))

arr_float16 = arr_int64.astype(np.float16)
print("float16:\t",sys.getsizeof(arr_float16))

exit(0)

s=time.clock()
np.dot(arr_float64.T,arr_float64)
print("float64:",time.clock()-s)

s=time.clock()
np.dot(arr_float32.T,arr_float32)
print("float32:",time.clock()-s)

s=time.clock()
np.dot(arr_float16.T,arr_float16)
print("float16:",time.clock()-s)

s=time.clock()
np.dot(arr_int64.T,arr_int64)
print("int64:",time.clock()-s)

s=time.clock()
np.dot(arr_int32.T,arr_int32)
print("int32:",time.clock()-s)

s=time.clock()
np.dot(arr_int16.T,arr_int16)
print("int16:",time.clock()-s)

s=time.clock()
np.dot(arr_int8.T,arr_int8)
print("int8:",time.clock()-s)

s=time.clock()
np.dot(arr_uint8.T,arr_uint8)
print("uint8:",time.clock()-s)
