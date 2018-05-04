import numpy as np

'''结构数组'''
# 'S1' 表示字符串 1个长度， 'S5' 5个长度
persontype=np.dtype({'names':['name','age','weight'], 'formats':['S5',np.uint8, np.float32]},align=True)
arr = np.array([("Zhang",32,90),("Wang",23,92)],dtype = persontype)
print(arr)

print(arr.dtype)

print(arr[0]['name'])
print(arr[0][0])
# print(arr[0,0])

print(arr[0]['age'])
print(arr[0][1])

# 可以通过arr.tostring()或arr.tofile()方法将数组arr以二进制的方式转换为字符串或写入文件中
arr.tofile("test.bin")  #存到文件中
