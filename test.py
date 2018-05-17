# encoding:utf-8

import re
from matplotlib import pyplot as plt

with open('run.log', 'r') as f:
    lines = f.read()

# 读取字符串类型误差值
pattern = re.compile(r':0\.\d+')
result = pattern.findall(lines)

# 转化为浮点数
value = []
for i in result:
    value.append(float(i.split(':')[-1]))


plt.plot(value)
plt.show()
