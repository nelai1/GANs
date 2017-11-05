# Generating new numbers from MNIST using DCGAN in keras
This is a slightly adapted version of the [radford 2016](https://arxiv.org/pdf/1511.06434.pdf) DCGAN implementation.
### Motivation
* gain additional experience with Keras
* implement DCGAN (with slightly different architecture then redford)

## Results
The images show the learning. The model was trained on the google cloud computing platform with 1 
K80. Every batch consists of 32 samples. Training was aborted after 12 epochs. (Every epoch consist of 60.000pictures/batch_size ~1900 iterations)

**Learning curves** 
![Loss curves after 12 epochs](examples/losses.png)

##Results:
* The network is learning
* No extended tuning was performed, there is definitely more to get out
* images below show the generated (left) and original (right) dataset after 12 epochs

**Generated Numbers**                    | **Mnist Numbers**
:---------------------------------------------------------:|:---------------------------------------------------:
![foo bar](examples/mnist_gen.png) | ![foo bar](examples/mnist.png)

The generator and discriminator model can be found in the examples folder.
```python
from keras.models import load_model
import numpy as np

gen = load_model('examples/gen.h5')
noise = np.random.normal(-1,1,size=[50, 100])
fake_images = gen.predict(noise)
```