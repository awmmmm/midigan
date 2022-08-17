# midigan
we provided [dataset](https://drive.google.com/file/d/1Oo4e2WgLg2s6xL97jp61GQfPAxR2ElmC/view?usp=sharing)  mentioned in our paper in google drive.

you can see it using python with numpy package.

```
train_data = np.load("X_train.npy").squeeze(2)
print(train_data)
print(train_data.shape)
```

```
[[ 0.92631579  0.92982456  0.92631579 ...  0.78947368  0.03508772
   0.47017544]
 [ 0.92982456  0.92631579  0.92982456 ...  0.03508772  0.47017544
   0.92982456]
 [ 0.92631579  0.92982456  0.92982456 ...  0.47017544  0.92982456
   0.98596491]
 ...
 [-0.86315789 -0.85614035 -0.87719298 ... -0.87719298  0.32631579
  -0.57192982]
 [-0.85614035 -0.87719298  0.32631579 ...  0.32631579 -0.57192982
  -0.87719298]
 [-0.87719298  0.32631579 -0.03157895 ... -0.57192982 -0.87719298
   0.32631579]]
```



```
(158373, 100)
```

as we can see , the first output the specific content ,the second output show the  format ,the dataset is composed of 158373 piece of 100 dim data,each data range in (-1,1).



we also provided the chord dict we used in our paper in dictionary './dataset'  that the first column is the index of the chord and the second column is the chord itself.
