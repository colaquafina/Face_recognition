# Face_recognition

*Dataset*: [Pubfig](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/)https://www.cs.columbia.edu/CAVE/databases/pubfig/download/, I only choose the pictures of the first 8 people. 
I download the provided pictures with the code *fig_load*, and got 408 pictures finally.

*Model*: Based on the petrained model MobilNetV2, we only add some convolution layers, and set up a value called Fine_tunning to decide which layers of the MobilNetV2 is not trainable(transfer learning).

It is a pretty easy program, but the accuracy is not very good, maybe because the training data is too small. However, it can help us learn how to write a simple deeplearning algorithm and how to use the pretrain model. 
