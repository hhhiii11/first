import random

class Memory:
    def __init__(self, size_max, size_min):
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        将样本添加到内存中
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            self._samples.pop(0)


    def get_samples(self, n):
        """
        从内存中随机获取 n 个样本
        """
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())
        else:
            return random.sample(self._samples, n)


    def _size_now(self):
        """
        检查是否已满
        """
        return len(self._samples)



