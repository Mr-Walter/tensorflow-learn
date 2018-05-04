参考：

- [因特尔加速版的python](https://software.intel.com/en-us/distribution-for-python)
- https://anaconda.org/intel
- [Numpy快速处理数据--ufunc运算](https://blog.csdn.net/kezunhai/article/details/46127845)
- [为什么Python很慢](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/)
- [为什么用 Numpy 还是慢, 你用对了吗?](https://zhuanlan.zhihu.com/p/28626431)
- [Python · numba 的基本应用](https://zhuanlan.zhihu.com/p/27152060)

-----
- numpy中数据类型运算效率  float32>flaot64>int32>其它类型
- 存取效率，int8=uint8>int16>float16>int32>float32
- pickle 存储时，pickle.dump(,,protocol=4)  protocol=4 存储小，加载速度最快， protocol=3（默认） 其次

------

[toc]

------
# 1、使用numpy代替循环

```python
import numpy as np
import timeit
import time

data=np.random.random([5000,5000])
print(data.dtype) # float64

q=np.random.choice(5000,1000)

s=time.clock()
for i in range(len(q)):
    d=np.dot(data,data[q[i]])
e=time.clock()

print(e-s) # 25.646767
```

- float64-->float32  速度提升9s
- numpy中数据类型运算效率  float32>flaot64>int32>其它类型

```python
import numpy as np
import timeit
import time

data=np.random.random([5000,5000])
# print(data.dtype) # float64

data=data.astype(np.float32) # float32

q=np.random.choice(5000,1000)

s=time.clock()
for i in range(len(q)):
    d=np.dot(data,data[q[i]])
e=time.clock()

print(e-s) # 16.11548
```

- 输出放在np.dot里可以提升1s左右

```python
import numpy as np
import timeit
import time

data=np.random.random([5000,5000])
# print(data.dtype) # float64

data=data.astype(np.float32) # float32

q=np.random.choice(5000,1000)

s=time.clock()

d=np.zeros([5000,],np.float32)

for i in range(len(q)):
    np.dot(data,data[q[i]],d)
e=time.clock()

print(e-s) # 14.377855
```

- 循环转成numpy矩阵运算  提速10多秒（尽量不使用循环）

```python
import numpy as np
import timeit
import time

data=np.random.random([5000,5000])
# print(data.dtype) # float64

data=data.astype(np.float32) # float32

q=np.random.choice(5000,1000)

s=time.clock()

d=np.zeros([5000,1000],np.float32)

np.dot(data,data[q].T,d)

e=time.clock()

print(e-s) # 2.6983449
```

# 2、math，cmath，numpy
对于计算**单个元素**，则建议采用Math库里对应的函数，np.sin( )对应math.sin( )，因为对于np.sin( )为了实现数组计算，底层做了很多复杂的处理，因此对于单个元素，math库里的函数计算速度快得多；而对于数组元素，则采用numpy库里的函数计算。

对于单个元素运算，make>cmath>p
```python
import numpy as np
import timeit
import time
import cmath
import math

data=np.random.random([1])
# print(data.dtype) # float64

s=time.clock()
np.sin(data)
e=time.clock()
print(e-s) # 2.5999999999998247e-05

s=time.clock()
cmath.sin(data)
e=time.clock()
print(e-s) # 1.6999999999989246e-05

s=time.clock()
math.sin(data)
e=time.clock()
print(e-s) # 4.000000000004e-06

# 转成float32
data=data.astype(np.float32) # float32

s=time.clock()
np.sin(data)
e=time.clock()
print(e-s) # 6.0000000000060005e-06

s=time.clock()
cmath.sin(data)
e=time.clock()
print(e-s) # 3.999999999976245e-06

s=time.clock()
math.sin(data)
e=time.clock()
print(e-s) # 2.000000000002e-06
```
# 3、numpy获取元素
item不如使用下标
```python
import numpy as np
import time


data=np.random.random([5000,5000])
# print(data.dtype) # float64
data=data.astype(np.float32) # float32

s=time.clock()
print(data.item(50,100))
e=time.clock()
print(e-s) # 0.000126000000000015

s=time.clock()
print(data[50,100])
e=time.clock()
print(e-s) # 2.0000000000020002e-05
```

# 4、fromnpyfunc
通过frompyfunc( )可以将计算单个值的函数转换为一个能对数组中每个元素计算的ufunc函数
```python
import numpy as np
import time


def triangle_wave(x, c, c0, hc):
    x = x - int(x)  # 周期为1，取小数部分计算
    if x >= c:
        r = 0.0
    elif x < c0:
        r = x / c0 * hc
    else:
        r = (c - x) / (c - c0) * hc
    return r

x = np.linspace(0,2,10000)
s=time.clock()
y1 = np.array([triangle_wave(t,0.6,0.4,1.0) for t in x])
e=time.clock()
print(e-s) # 0.0074800000000000005

s=time.clock()
# ufunc是计算单个元素的函数,nin是输入参数的个数，nout是func返回值的个数
triangle_wave_ufunc = np.frompyfunc(triangle_wave, 4, 1)
y2 = triangle_wave_ufunc(x, 0.6, 0.4, 1.0)
e=time.clock()
print(e-s) # 0.0034200000000000064

# 代码简洁高效。值得注意：triangle_wave_ufunc（ ）所返回数组的元素类型是object，
# 因此还需要调用数组的astype()方法将其转换为双精度浮点数组：
```

```python
import numpy as np
import timeit
import time
import cmath
import math

data=np.random.random([50000,])
# print(data.dtype) # float64
data=data.astype(np.float32)

s=time.clock()
np.sin(data)
e=time.clock()
print(e-s)

s=time.clock()
[cmath.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
[math.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(np.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(cmath.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(math.sin,1,1)
func(data)
e=time.clock()
print(e-s)

'''
0.0003999999999999837
0.012136999999999981
0.007717000000000002
0.02742400000000003
0.008526999999999951
0.0044199999999999795
'''
```

# 5、map
通过map()可以将计算单个值的函数转换为一个能对数组中每个元素计算的ufunc函数

```python
import numpy as np
import timeit
import time
import cmath
import math

data=np.random.random([50000,])
# print(data.dtype) # float64
data=data.astype(np.float32)

s=time.clock()
np.sin(data)
e=time.clock()
print(e-s)

s=time.clock()
[cmath.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
[math.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(np.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(cmath.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(math.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
da=map(math.sin,data)
e=time.clock()
print(e-s)

da=list(da) # 取出数据

pass
'''
0.00038799999999999946
0.01245199999999999
0.007722000000000007
0.030077999999999994
0.008251000000000008
0.004479999999999984
5.0000000000050004e-06
'''
```

# 6、vectorize
使用vectorize( )可以实现和frompyfunc( )类似的功能，但他可以通过otypes参数指定返回数组的元素类型。otypes参数可以是一个表示元素类型的字符串，也可以是一个类型列表，使用列表可以描述多个返回数组的元素类型，如将上面的代码改成vectorize( )，则为：

```python
import numpy as np
import timeit
import time
import cmath
import math

data=np.random.random([500000,])
# print(data.dtype) # float64
data=data.astype(np.float32)

s=time.clock()
d=np.sin(data)
e=time.clock()
print(e-s)
# print(d[0])

s=time.clock()
[cmath.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
[math.sin(i) for i in data]
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(np.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(cmath.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
func=np.frompyfunc(math.sin,1,1)
func(data)
e=time.clock()
print(e-s)

s=time.clock()
da=map(np.sin,data)
e=time.clock()
print(e-s)

# da=list(da) # 取出数据
# print(da[0])

s=time.clock()
func=np.vectorize(math.sin,otypes=[np.float32])
func(data)
e=time.clock()
print(e-s)

'''
0.0068040000000000045
0.131255
0.07860100000000003
0.288777
0.08116699999999999
0.04541499999999998
2.9999999999752447e-06
0.05927899999999997
'''
```
# 7、线程
# 8、进程
# 9、cython
# 10、C/C++
# 11、cuda
# 12、numba

# 13、其它技巧

- order='F' 比 order='C' 性能好
```python
import numpy as np
import time

a = np.zeros((200, 200), order='C') # 以 row 为主在内存中排列
b = np.zeros((200, 200), order='F') # 以 column 为主在内存中排列

N = 2000

s=time.clock()
x=np.arange(N)
[np.concatenate((a, a), axis=0) for _ in x]
print(time.clock()-s)


s=time.clock()
[np.concatenate((a, a), axis=0) for _ in range(N)]
print(time.clock()-s)

s=time.clock()
x=np.arange(N)
[np.concatenate((b, b), axis=0) for _ in x]
print(time.clock()-s)

s=time.clock()
[np.concatenate((b, b), axis=0) for _ in range(N)]
print(time.clock()-s)

'''
0.5458940000000001
0.4799070000000001
0.47622699999999996
0.49617999999999984
'''
```

```python
import numpy as np
import time


a = np.zeros((200, 200), order='C') # 以 row 为主在内存中排列
b = np.zeros((200, 200), order='F') # 以 column 为主在内存中排列


N = 2000

s=time.clock()
x=np.arange(N)
[np.concatenate((a, a), axis=1) for _ in x]
print(time.clock()-s)


s=time.clock()
[np.concatenate((a, a), axis=1) for _ in range(N)]
print(time.clock()-s)

s=time.clock()
x=np.arange(N)
[np.concatenate((b, b), axis=1) for _ in x]
print(time.clock()-s)

s=time.clock()
[np.concatenate((b, b), axis=1) for _ in range(N)]
print(time.clock()-s)

'''
0.512697
0.480291
0.4635490000000002
0.46421200000000007
'''
```
- a*=2 比 a=a*2（a 赋值给另外一个新建的 a）速度快

## 1、 使用 `np.take()`, 替代用 index 选数据的方法
上面提到了如果用index 来选数据, 像 `a_copy1 = a[[1,4,6], [2,4,6]]`, 用 take 在大部分情况中会比这样的 `a_copy1` 要快.

```python
a = np.random.rand(1000000, 10)
N = 99
indices = np.random.randint(0, 1000000, size=10000)

def f1(a):
    for _ in range(N):
        _ = np.take(a, indices, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[indices]

print('%f' % ((t1-t0)/N))    # 0.000393
print('%f' % ((t2-t1)/N))    # 0.000569
```

## 2、使用 `np.compress()`, 替代用 mask 选数据的方法.
上面的 `a_copy2 = a[[True, True], [False, True]]` 这种就是用 TRUE, FALSE 来选择数据的. 测试如下:
```python
mask = a[:, 0] < 0.5
def f1(a):
    for _ in range(N):
        _ = np.compress(mask, a, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[mask]

print('%f' % ((t1-t0)/N))    # 0.028109
print('%f' % ((t2-t1)/N))    # 0.031013
```

## 3、非常有用的 out 参数

```python
a = a + 1         # 0.035230
a = np.add(a, 1)  # 0.032738

a += 1                 # 0.011219
np.add(a, 1, out=a)    # 0.008843
```
带有 out 的 numpy 功能都在这里: [Universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs). 所以只要是已经存在了一个 placeholder (比如 a), 我们就没有必要去再创建一个, 用 out 方便又有效.

## 4、给数据一个名字
我喜欢用 pandas, 因为 pandas 能让你给数据命名, 用名字来做 index. 在数据类型很多的时候, 名字总是比 index 好记太多了, 也好用太多了. 但是 pandas 的确比 numpy 慢. 好在我们还是有途径可以实现用名字来索引. 这就是 structured array. 下面 a/b 的结构是一样的, 只是一个是 numpy 一个是 pandas.

```python
a = np.zeros(3, dtype=[('foo', np.int32), ('bar', np.float16)])
b = pd.DataFrame(np.zeros((3, 2), dtype=np.int32), columns=['foo', 'bar'])
b['bar'] = b['bar'].astype(np.float16)

"""   
# a
array([(0,  0.), (0,  0.), (0,  0.)],
      dtype=[('foo', '<i4'), ('bar', '<f2')])

# b
   foo  bar
0    0  0.0
1    0  0.0
2    0  0.0
"""

def f1(a):
    for _ in range(N):
        a['bar'] *= a['foo']

def f2(b):
    for _ in range(N):
        b['bar'] *= b['foo']

print('%f' % ((t1-t0)/N))    # 0.000003
print('%f' % ((t2-t1)/N))    # 0.000508
```
 [Numpy Vs Pandas Performance Comparison](http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html)

