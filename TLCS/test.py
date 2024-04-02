
from collections import Counter


n = 1
n1 = 3
n2 = 3
n[1] = 1
n[2] = 7
print(n)
lst = [n, n1, n2]
    # 使用Counter函数统计每个元素出现的次数
cnt = Counter(lst)
# 找到出现次数最多的元素
most_common = max(cnt, key=cnt.get)
print(most_common)
