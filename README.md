# Bag of Visual Words(BoVW)

**UniTn Machine Learning Lab 2022**

The general structure of what we want to do is something like:

- Load the train and test images
- Extract some descriptors from all images. Here we are going to use an algorithm called SIFT/ORF
- Generate visual words by applying k-means to the SIFT/ORF descriptors of the train data
- Extract an histogram of the visual words from train and test data (effectively generating our features)
- Training a ML algorithm
- Apply it to test data and evaluate
#

**Implementations:**
  - [BoVW Notebook](https://github.com/khandakerrahin/BOVW/blob/main/BOVW.ipynb)
#
**Dataset sources**: 
- [MINST](http://yann.lecun.com/exdb/mnist/)
#
#
