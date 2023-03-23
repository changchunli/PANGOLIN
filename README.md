# PANGOLIN (CIKM2020) 
This is an official implementation of [Learning with Noisy Partial Labels by Simultaneously Leveraging Global and Local Consistencies], which is accepted by CIKM2020.

## Abstract
In real-world scenarios, the data are widespread that are annotated with a set of candidate labels but a single ground-truth label per-instance. The learning paradigm with such data, formally referred to as Partial Label (PL) learning, has recently drawn much attention. The traditional PL methods estimate the confidences being the ground-truth label of candidate labels with various regularizations and constraints, however, they only consider the local information, resulting in potentially less accurate estimations as well as worse classification performance. To alleviate this problem, we propose a novel PL method, namely PArtial label learN ing by simultaneously leveraging GlObal and Local consIsteNcies (PANGOLIN). Specifically, we design a global consistency regularization term to pull instances associated with similar labeling confidences together by minimizing the distances between instances and label prototypes, and a local consistency term to push instances marked with no same candidate labels away by maximizing their distances. We further propose a nonlinear kernel extension of PANGOLIN, and employ the Taylor approximation trick for efficient optimization. Empirical results demonstrate that PANGOLIN significantly outperforms the existing PL baseline methods.

## Prerequisite
* The requirements are in **requirements.txt**.

## Usage
Train the model by running the following command directly.

```
python pangolin.py
```

