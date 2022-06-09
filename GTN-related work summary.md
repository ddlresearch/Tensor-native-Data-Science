## GTN Related Work

### Graph
1. Tensors: An abstraction for general data processing (VLDB 2021) (The Hummingbird project) [[Paper](http://vldb.org/pvldb/vol14/p1797-koutsoukos.pdf)] [[Github](https://github.com/microsoft/hummingbird)]
> * Motivation: The implementations of GraphBLAS turn out to be suboptimal when executed over TCRs, because they rely on sparse representations of the graph, while TCRs are not efficient for sparse computations. Hence, novel implementations are required...
> * This work explores to what extent Tensor Computation Runtimes (TCRs) can support *non-ML* data processing applications, e.g, PageRANK.
2. Graph Traversal with Tensor Functionals: A Meta-Algorithm for Scalable Learning (ICLR 2021) [[Paper](https://openreview.net/forum?id=6DOZ8XNNfGN)]


### SQL
1. Query Processing on Tensor Computation Runtimes [[Paper](https://arxiv.org/pdf/2203.01877.pdf)] [[Blog](https://medium.com/syncedreview/meet-tqp-the-first-query-processor-to-run-on-tensor-computation-runtimes-delivers-up-to-20x-7d1f09d3b9f8)] [[News](https://www.marktechpost.com/2022/03/13/researchers-from-the-university-of-washington-and-uc-san-diego-introduce-tensor-query-processor-tqp-with-tensor-computation-runtimes-for-query-processing-20x-speedup/)]
> * Tensor Query Processor (TQP): a SQL query processor leveraging the tensor interface of TCRs
2. TDM: A Tensor Data Model for Logical Data Independence in Polystore Systems [[Paper](http://eric-leclercq.fr/papers/VLDB-Polystore-2018.pdf)]
> * ...polystore systems as a collection of heterogeneous data stores with multiple query interfaces

### ML Inference
1. Exploiting tensor networks for efficient machine learning (PhD thesis) [[Link](https://hub.hku.hk/handle/10722/308618)]
> * This thesis explores the tensorization and compression of machine learning models (SVMs and RBMs).
2. A Tensor Compiler for Unified Machine Learning Prediction Serving (OSDI 2020) (The Hummingbird project) [[Paper](https://web.eecs.umich.edu/~mosharaf/Readings/Hummingbird.pdf)]
> * It compiles featurization operators and traditional ML models (e.g., decision trees) into a small set of tensor operations.
3. Compiling Classical ML Pipelines into Tensor Computations for One-size-fits-all Prediction Serving (System for ML Workshop, NeurIPS 2019) (The Hummingbird project) [[Paper](http://learningsys.org/neurips19/assets/papers/27_CameraReadySubmission_Hummingbird%20(5).pdf)]
> * It compile classical ML pipelines end-to-end into tensor computations. It thereby seamlessly leverages the features provided by DNN inference systems, e.g., ease of deployment, operator optimizations and GPU support.

### TCR 
1. Learning to Optimize Tensor Programs (Nips'18 Tianqi Chen, AutoTVM) [[Paper](https://arxiv.org/pdf/1805.08166.pdf)] [[PPT](https://nips.cc/media/nips-2018/Slides/12580.pdf)]
> *  We learn domain-specific statisticalcost models to guide the search of tensor operator implementations over billions of possible program variants. We further accelerate the search using effective model transfer across workloads. Experimental results show that our framework delivers performance that is competitive with state-of-the-art hand-tuned libraries for low-power CPUs, mobile GPUs, and server-class GPUs.
2. The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding [[Paper](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/fegade-2110.10221.pdf)]
> * There is often variation in the shape and size of input data used for deep learning. In many cases, such data can be represented using tensors with non-uniform shapes, or ragged tensors ...
> *  Such techniques can, however, lead to a lot of wasted computation and therefore, a loss in performance. This paper presents CORA, a tensor compiler that allows users to easily generate efficient code for ragged tensor operators targeting a wide range of CPUs and GPUs. 
3. TVM Guides: Use Tensorize to Leverage Hardware Intrinsics[[Doc](https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html)]
4. SIMD^2: A Generalized Matrix Instruction Set for Accelerating Tensor Computation beyond GEMM [[Paper](https://arxiv.org/pdf/2205.01252.pdf)]


### Vectorized and Distributed
1. RDD-based API: vector and distributed matrix [[Doc](https://spark.apache.org/docs/latest/mllib-data-types.html)]
