# Algorithm_03

MODEL 1 : 3 Layers with 1 Convolution layer
---

#### 1.1. Training with Training loss

```python
model.fit(train_images, train_labels,  epochs = 5)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 25s 414us/sample - loss: 0.5652 - accuracy: 0.9405
    Epoch 2/5
    60000/60000 [==============================] - 19s 323us/sample - loss: 0.0886 - accuracy: 0.9739
    Epoch 3/5
    60000/60000 [==============================] - 20s 331us/sample - loss: 0.0713 - accuracy: 0.9788
    Epoch 4/5
    60000/60000 [==============================] - 20s 337us/sample - loss: 0.0650 - accuracy: 0.9805
    Epoch 5/5
    60000/60000 [==============================] - 20s 341us/sample - loss: 0.0557 - accuracy: 0.9837
    
    <tensorflow.python.keras.callbacks.History at 0x640665c10>

#### 1.2. Test Accuracy

```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```

    10000/1 - 1s - loss: 0.0582 - accuracy: 0.9718
    
    Test loss :  0.11564373795761203
    Test accuracy : 0.9718

#### 1.3. Images and corresponding probability that predicted Right

![output_30_0](https://user-images.githubusercontent.com/38272356/83324398-03c60e80-a2a0-11ea-8033-8264d58d9129.png)

#### 1.4. Images and corresponding probability that predicted Wrong

![output_36_0](https://user-images.githubusercontent.com/38272356/83324402-06286880-a2a0-11ea-9233-3f2230725a24.png)

MODEL 2 : 5 Layers with 2 Convolution layer
---

#### 2.1. Training with Training loss

```python
model.fit(train_images, train_labels,  epochs = 5)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 51s 856us/sample - loss: 0.2937 - accuracy: 0.9458
    Epoch 2/5
    60000/60000 [==============================] - 43s 719us/sample - loss: 0.0687 - accuracy: 0.9792
    Epoch 3/5
    60000/60000 [==============================] - 41s 686us/sample - loss: 0.0549 - accuracy: 0.9830
    Epoch 4/5
    60000/60000 [==============================] - 42s 692us/sample - loss: 0.0481 - accuracy: 0.9853
    Epoch 5/5
    60000/60000 [==============================] - 40s 673us/sample - loss: 0.0401 - accuracy: 0.9879
    
    <tensorflow.python.keras.callbacks.History at 0x646e10390>

#### 2.2. Test Accuracy

```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```

    10000/1 - 2s - loss: 0.0216 - accuracy: 0.9866
    
    Test loss :  0.04300872993719213
    Test accuracy : 0.9866

#### 2.3. Images and corresponding probability that predicted Right

![output_30_0](https://user-images.githubusercontent.com/38272356/83324546-e2b1ed80-a2a0-11ea-8b06-63e907e10b93.png)

#### 2.4. Images and corresponding probability that predicted Wrong

![output_36_0](https://user-images.githubusercontent.com/38272356/83324549-e5acde00-a2a0-11ea-89b5-d0eca095a423.png)

MODEL 3 : 7 Layers with 4 Convolution layer
---

#### 3.1. Training with Training loss

```python
model.fit(train_images, train_labels,  epochs = 5)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 74s 1ms/sample - loss: 0.1541 - accuracy: 0.9581
    Epoch 2/5
    60000/60000 [==============================] - 78s 1ms/sample - loss: 0.0545 - accuracy: 0.9838
    Epoch 3/5
    60000/60000 [==============================] - 73s 1ms/sample - loss: 0.0429 - accuracy: 0.9870
    Epoch 4/5
    60000/60000 [==============================] - 76s 1ms/sample - loss: 0.0361 - accuracy: 0.9888
    Epoch 5/5
    60000/60000 [==============================] - 92s 2ms/sample - loss: 0.0331 - accuracy: 0.9898

    <tensorflow.python.keras.callbacks.History at 0x63cfff250>

#### 3.2. Test Accuracy

```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```

    10000/1 - 3s - loss: 0.0212 - accuracy: 0.9876
    
    Test loss :  0.041838526410068154
    Test accuracy : 0.9876

#### 3.3. Images and corresponding probability that predicted Right

![output_30_0](https://user-images.githubusercontent.com/38272356/83324741-5b657980-a2a2-11ea-94fb-a05a2b39ca0e.png)

#### 3.4. Images and corresponding probability that predicted Wrong

![output_36_0](https://user-images.githubusercontent.com/38272356/83324746-5e606a00-a2a2-11ea-933c-3f32d0314d02.png)
