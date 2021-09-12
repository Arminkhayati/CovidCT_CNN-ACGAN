#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/Arminkhayati/CovidCT_CNN-')

    


# In[ ]:


import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB7, ResNet50
# from classification_models.keras import Classifiers


# In[ ]:


# https://keras.io/api/preprocessing/image/#flowfromdirectory-method

batch_size = 16

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=[0.7, 1.0],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
        cval=0.0,)

test_datagen = ImageDataGenerator(
    rescale=1./255
    )





train_data = train_datagen.flow_from_directory(
        './CovidCT_CNN-/data/train',
        target_size=(256, 256),
        color_mode="rgb",
        class_mode="categorical",
        shuffle=True,    
        batch_size=batch_size
)

test_data = train_datagen.flow_from_directory(
        './CovidCT_CNN-/data/test',
        target_size=(256, 256),
        color_mode="rgb",
        class_mode="categorical",
        shuffle=False,
        batch_size=batch_size
)




t_x, t_y = train_data.__getitem__(0)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
      c_ax.imshow(np.clip(c_x * 255, 0, 255).astype('int'))
      c_ax.set_title('Severity {}'.format(c_y))
      c_ax.axis('off')

t_x.shape[1:]


# In[ ]:


input_shape = (256, 256, 3)
# img_input = Input(shape=input_shape)
base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
base_model.trainable = True
base_model_input = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
model = Dense(2, activation='softmax')(x)
model = Model(base_model_input, model)
model.compile(optimizer = Adam(), loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('covid')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)
callbacks_list = [checkpoint, reduceLROnPlat]

history = model.fit(train_data, 
                    batch_size=batch_size,
                    validation_data = test_data,
                    epochs = 50, 
                    callbacks = callbacks_list,)
                    # workers=2,
                    # use_multiprocessing=True)
model.load_weights(weight_path)
model.save('full_covid_model.h5')


# In[ ]:


def plotmodel(history,name):
    
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    lrs = history.history['lr']
    epochs = range(1, len(acc) + 1) 
    
    plt.figure(1)                  
    plt.plot(epochs,acc)#mooth_curve(acc))
    plt.plot(epochs,val_acc)#smooth_curve(val_acc))
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.savefig('acc_'+name+'.png')
    
    plt.figure(2)
    plt.plot(epochs, loss)#smooth_curve(loss))
    plt.plot(epochs,val_loss)#smooth_curve(val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig('loss_'+name+'.png')
    
    plt.figure(3)
    plt.plot(epochs, lrs)#,smooth_curve(lrs))
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper right')
    plt.savefig('lr_'+name+'.png')

plotmodel(history,'history')


# In[ ]:


from IPython.display import FileLink
FileLink('./full_covid_model.h5')


# In[ ]:


# Plot Results


# In[ ]:


model = load_model("/content/drive/My Drive/nn-prject/resnet50-3-dense.h5")
pred= model.predict(test_data, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = test_data.classes[0:len(predicted_class_indices)]
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(labels, predicted_class_indices)))
print(classification_report(labels, predicted_class_indices))


# In[ ]:


# Confusio Matrix

sns.heatmap(confusion_matrix(labels, predicted_class_indices), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues)


# Roc curve and Average precision recall
sick_vec = labels>0
sick_score = np.sum(pred[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')

n_classes=2
# For each class
precision = dict()
recall = dict()
average_precision = dict()
level_cat = to_categorical(test_data.classes).astype('int')[:len(labels)]
# level_cat = np.array([l.flatten() for l in level_cat])
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(level_cat[:, i],
                                                        pred[:, i])
    average_precision[i] = average_precision_score(level_cat[:, i], pred[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(level_cat.ravel(),
    pred.ravel())
average_precision["micro"] = average_precision_score(level_cat, pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}\n\n\n\n\n'
      .format(average_precision["micro"]))


plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# Multi class precision recall
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(level_cat[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(level_cat.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw = 2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Multi class ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# 
