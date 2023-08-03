import numpy as np              # 导入numpy库，用于处理数组和矩阵

d = 64                          # 维度：向量的维度
nb = 100000                     # 数据库大小：数据库中向量的数量
nq = 10000                      # 查询数量：要搜索的查询向量的数量
np.random.seed(1234)            # 设置随机数种子，使随机生成的数据可复现

xb = np.random.random((nb, d)).astype('float32')   # 随机生成大小为(nb, d)的数据库向量
xb[:, 0] += np.arange(nb) / 1000.                 # 修改数据库向量的第一个元素

xq = np.random.random((nq, d)).astype('float32')   # 随机生成大小为(nq, d)的查询向量
xq[:, 0] += np.arange(nq) / 1000.                 # 修改查询向量的第一个元素

import faiss                     # 导入faiss库，用于相似性搜索和聚类

index = faiss.IndexFlatL2(d)     # 创建IndexFlatL2的实例，使用L2距离度量，用于索引

print(index.is_trained)         # 检查索引是否已训练，因为使用的是预训练索引，所以会打印True

index.add(xb)                    # 将数据库向量添加到索引中

print(index.ntotal)             # 打印当前索引中的向量数量（应该等于nb）

k = 4                            # 要查找的最近邻居数量

D, I = index.search(xb[:5], k)   # 进行一次简单的检查，找到前5个数据库向量的k个最近邻居
print(I)                         # 打印每个查询向量的最近邻居的索引
print(D)                         # 打印到相应最近邻居的距离

D, I = index.search(xq, k)      # 进行实际的搜索，找到所有查询向量的k个最近邻居

print(I[:5])                    # 打印前5个查询向量的最近邻居的索引
print(I[-5:])                   # 打印最后5个查询向量的最近邻居的索引