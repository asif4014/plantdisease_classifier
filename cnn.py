
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Tomato/train_set',target_size = (60, 60),batch_size = 10)

test_set = test_datagen.flow_from_directory('Tomato/test_set',target_size = (60, 60),batch_size = 10)

model = classifier.fit_generator(training_set,steps_per_epoch = 250,epochs = 20,validation_data = test_set,callbacks=[es_cb],validation_steps = 50)

classifier.save("tomato_classifier.h5")
print("Saved model to disk")


# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Tomato/test_set/Healthy/2(healthy).JPG', target_size = (60, 60))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] == 1:
    prediction = 'Bacterial_spot'
elif result[0][1] == 1:
    prediction = 'Early_blight'
else:
    prediction = 'Healthy'
print(prediction)