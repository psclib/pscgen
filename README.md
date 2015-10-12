Parallel Sparse Coding Library (PSCL)
---

PSCL aims to make data-driven modeling and prediction accessible to many applications.

PSCL generates NNU output files containing pre-computed lookup tables. The tables contains the pre-computed dot products between dictionary D and all possible w-bit values of the input x (e.g., 8-bit input will have 256 possible input values).

PSCL is based on the following paper: 

HT Kung, B McDanel and S Teerapittayanon, "PNNU: Parallel Nearest-Neighbor Units for Learned Dictionaries", 28th International Workshop on Languages and Compilers for Parallel Computing, September 2015

Motivation
---

In the era of big data, we need high-performance solutions to support data-driven modeling and prediction. A lot of the computations are based on Nearest Neighbor Search (NNS), e.g., Sparse Coding. We focus on optimizing NNS and Sparse Coding.

Specification
---
Given a dictionary and device specific parameters such as memory size, the number of cores and floating-point precision, PSCL generator generates client-specific NNU files which can be loaded into an optimized multithreaded library provided in separate repositories listed below.

Format of NNU output files
---

- Binary
- JSON

Libraries
---

- C Library
- C++ Library
- JS Library
- Python Library
- MATLAB Library


