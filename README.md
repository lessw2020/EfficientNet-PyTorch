# EfficientNet-PyTorch
Unofficial port of Google's new EfficientNet to Pytorch and FastAI. 

Models B0-7 loading.  
Note that the code is in the notebook and assumes you have access to FastAI dev course 2 notebooks.  
Will remove that dependency and port to .py file after further testing. 

Usage: 

arch = effNet(model=5, c_out = 10)  
where model = B0-B7, and c_out = number of classes in classifier.

 Here's the official code in TF:

https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

and info about it from their blog:
https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html
