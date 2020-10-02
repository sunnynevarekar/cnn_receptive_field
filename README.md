## Effective receptive field in Deep Convolution neural network

The receptive field in Convolutional Neural Networks (CNN) is the region of the input space that affects a particular unit of the network. Python program ```receptivefield.py``` calculates effective receptive field for a given layer in CNN.  

Usage: 
```python
python receptivefield.py [-i IMAGE_PATH] [-s1 IMG_HEIGHT] [-s2 IMG_WIDTH] [-l LAYER_NAME]
```

E.g.
```python
python receptivefield.py --img_path=resources/sat.png --img_height=512 --img_width=512 --layer_name=down5_conv2
```

![alt text](resources/receptive_viz.gif "receptive field gif")