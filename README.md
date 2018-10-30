# GpSM

implementation of ideas by Leyuan Wang to solve subgraph isomorphism problem on GPU, using Gunrock framework.

---

#### Dataset

注意：所有数据集的节点编号都必须从0开始，所有标签必须从1开始，
否则会有问题

we target at a one-to-one mapping at a time, the query graph is small(vertices less than 100), while the data graph can be very large.
(but can be placed in GPU's global memory)

目前的子图同构采用的是普通的子图的形式，而不是导出子图的形式(induced subgraph)

---

#### Paper 

Fast Parallel Subgraph Matching on the GPU, HPDC 2016 (CCF B poster)

---

#### Algorithm

Due to the hardness of Gunrock framework, we do not adopt it directly but imitate its operators by ourselves.

filter and verify framework(using enumeration)

edge as the basic unit

a little different from the original algorithm in the paper:
1. not advance from each row to form a big array due to the limit of GPU memory
2. not use intersections of two big sorted lists on the host side
3. not use atomic operations to add a new answer to the result list, which is costly and hard to implement(using atomic CAS?)

---

#### TODO

no need to count effect bits on CPU, on GPU it is faster

no need to use a long type for each flag, just a bit is ok

when running watdiv1M with triangle query, GunrockSM fails. 
We need to implement the ideas above to see whether it works or not then.

