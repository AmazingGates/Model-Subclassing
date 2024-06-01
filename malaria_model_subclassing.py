import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import Accuracy, RootMeanSquaredError
from keras.optimizers import Adam


dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, shuffle_files=True, 
                                  split=["train"])


for data in dataset[0].take(4):
    print(data)


def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))

    val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))

    test_dataset = val_test_dataset.skip(int(VAL_RATIO*dataset_size))

    return train_dataset, val_dataset, test_dataset


TRAIN_RATIO = 0.8
VAL_RATIO= 0.1
TEST_RATIO = 0.1


train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)  

print(list(train_dataset.take(1).as_numpy_iterator()), list(val_dataset.take(1).as_numpy_iterator()), 
      list(test_dataset.take(1).as_numpy_iterator()))


for i, (image, label) in enumerate(train_dataset.take(16)):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image)
    plt.title(dataset_info.features["label"].int2str(label))
    plt.show()
    print(plt.imshow(image))


IM_SIZE = 224

def resizing(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)), label


train_dataset = train_dataset.map(resizing)
print(train_dataset)

print(resizing(image, label))


for image, label in train_dataset.take(1):
    print(image, label) 


def resize_rescale(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image") 

x = Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(func_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

x = Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)


x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

#x = Dense(1, activation = "sigmoid")(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model")

lenet_model_func.summary()

#################################################################################

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image") 

x = Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(func_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

x = Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(x)
x = BatchNormalization()(x)
output = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

feature_extractor_model = Model(func_input, output, name = "Feature_Extractor")

feature_extractor_model.summary()

#################################################################################

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_extractor_model(func_input)

x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

#x = Dense(1, activation = "sigmoid")(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model")

lenet_model_func.summary()

###########################################################################################

y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
#print(bce(y_true, y_pred))

#lenet_model.compile(optimizer = Adam(learning_rate=0.01),
#              loss = BinaryCrossentropy(),
#              metrics = "accuracy")

#history = lenet_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

###################################################################################################


# It's important to note that model subclassing permits us to create recursively composable layers and models.

# Now, what does that mean?

# This means that we could create a layer where its attributes are other layers, and this layer tracks the weights
#and biases of the sub layers.

# Before making an example, let's get this import.

# We're going to import layer from layers (See end of line 6)

# Now we can create our model using the Model subclassing

# With all that done, now we can start with our feature extractor function.

# This will inherit from layer.

# So it will inherit from tensorflow Layer

# Next we will have an init method, followed by a call method.

# Next, we can copy the feature extractor that we recently created and paste in to our subclass model, minus the
#model = Model and the model.Summary()

# Next we will add a super() and pass in as a parameters FeatureExtractor and self, then we will add a dot init method.

# Now that we have defined this, we can goahead use our layers as the attribute for this feature extractor layer.

# We will use the self.conv_1, which will be equal to Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", 
#activation = "relu")

# From there we have self.batch_1, which is going to be our first Batchnormalizer()

# Next we will have the self.pool_1, which will get MaxPool2D(pool_size=2, strides= 2, padding = "valid")

# And then we just repeat this process so we could have the same thing for self.conv_2

# Now we can start building our call method, which takes an input two parameters, (self, x).

# This permits us to call each and every layer defined in our init method.

# We'll start with x equals self.conv_1(x), which looks similar to our function API.

# Then we will do the same for the rest of self's.

# From here we just return x

# Also note that we are going to pass in an additional parameter to our call method. call(self, x, training).

# The training argument can tell us whether to use a given layer or not during the training process.

# For now, all this is going to be used for the training.

# Now we can take off the rest of the code we pasted from the functional API section.

class FeatureExtractor(Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, pool_size):
        super(FeatureExtractor, self).__init__()
        self.conv_1 = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, 
                             activation = activation)
        self.batch_1 = BatchNormalization()
        self.pool_1 = MaxPool2D(pool_size = pool_size, strides = 2*strides)

        self.conv_2 = Conv2D(filters = filters*2, kernel_size = kernel_size, strides = strides, padding = padding, 
                             activation = activation)
        self.batch_2 = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size = pool_size, strides = 2*strides)

#        self.conv_1 = Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")
#        self.batch_1 = BatchNormalization()
#        self.pool_1 = MaxPool2D(pool_size=2, strides= 2, padding = "valid")

#        self.conv_2 = Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")
#        self.batch_1 = BatchNormalization()
#        self.pool_1 = MaxPool2D(pool_size=2, strides= 2, padding = "valid")

    def call(self, x, training):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.pool_2(x)

        return x
      




# Now we will be able to build our model.

# We will be using the Lenet model first.

# We'll take the feature extractor function version that we created.

# But before copying and pasteing, we have to ensure that we have created the feature_sub_classed, which is the 
#FeatureExtractor from our subclass model we created.

# We could always pass in the parameters, like the number of filters or the kernel size via this def __init__()
#that we created in in our Model subclass model.

# we could specify filters and the kernel size like this def __init__(self, filters, kernel_size, strides, padding,
#activation, ). See line 220

# Once we pass these in as parameters, we no longer need to specify them in our self.'s. See lines 223 - 236 for
#comparisons.

# Note: MaxPool2D keeps its strides = 2 parameters and it gets modified to strides = 2*strides. See lines 224 and 228

# Now that we have defined all of that we are ready to pass in these values.

# So we simply copy all the values from the def__init__(), minus the self, and then we pass them as parameters in our
#FeatureExtractor(). See line 275

# Once we have our parameters copied and pasted we can specify their values.

# Note: These values will actually replace the words and stand alone as parameters.

# Now that everything is defined we will ensure that we have our times two specified for our Conv2D filters 
#parameter. See line 226

# Also note, to avoid a value error, we have to modify our Conv2D_1, Conv2D_2 and pool_size. See line 222,225,227,230

# Now we can run our program.

feature_sub_classed = FeatureExtractor(8, 3, 1, "valid", "relu", 2)


func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_extractor_model(func_input)

x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

#x = Dense(1, activation = "sigmoid")(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Subclass_Model")

lenet_model_func.summary()

# Now we have successfully created our feature Subclass layer.

# We will now be able to use this in the model we have.

# We'll start by copying and pasteing our model information.

# Then we will modify x = feature ectractor model from our pasted information to x = feature subclassed (see line 322 
#for the modification) (see line 323 for commented out original)

# Now we can comfortably run this new model.

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_sub_classed(func_input)
#x = feature_extractor_model(func_input)

x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)


func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model_Example")

lenet_model_func.summary()


# One last thing we could do, instead of doing it this way, we're going to create a model using the model subclassing
#method.

# First we will copy the class FeatureExtractor()

# And instead of having layer, we're going to to pass in a parameter of Model into our FeatureExtractor()

# And then we're going to define a Feature Extractor.

# We will do this changing the self.conv_1 to a self.feature_extractor.

# And this self.feature_extractor will be equal to the FeatureExtractor() we had for our feature sub classed.

# Note: The FeatureExtractor(8, 3, 1, "valid", "relu", 2) replaces all of the other self.'s

# Next we're going to modify our call() by changing the x = self.conv_1(x) to x = self.feature_extractor(x).

# Once we are done that everything else can be removed, up until the return x.

# Now that we're done with the feature extraction, we can get the other parts that make up the model, like the 
#flatten(), the Dense(), and the Batchnormalization().

# Also note that we're going to modify our class from FeatureExtractor to class LenetModel

# We will also modify our super() to now take LenetModel as a parameter instead of the FeatureExtractor

# Next we will have a self.flatten equals Flatten

# Next is the self.dense_1 equal to Dense(1000, activation = "relu")

# Now the self.batch_1 equal to BatchNormalization() is next

# We can do the samething for dense_2 and batch_2

# Note: dense_2 should match the number that it's copied from, in this case, dense_2 is 100.

# And finally we have our last Dense() layer.

# This last Dense() doesn't get a batch, and it's activation is a sigmoid.

# Everything esle can now be removed.

# Now we'll get into our call() method.

# In our call() we're basically going to call all of our layers.

# So after the feature extraction we'll create a class.

# And this class makes use of the feature extractor, which was also created using the same model subclassing method.

# self.flatten = Flatten() is first and it gets modified to x = self.flatten(x) and it goes after the 
#x = self.feature_extractor(x).

# Next the rest of the self.'s get modified and then added.

# Now we have our Lenet Model

# We will name it the lenet_sub_classed and it's going to be equalled to the LenetModel

# Our model is now complete.

# To run our model of course we'll use the lenet_sub_classed.summary(), but before we can run our model, we have to
#call our model on a batch of data.

# We can do that by doing this, lenet_sub_classed(tf.zeros([1, 224, 224, 3])).

# This lenet_sub_classed(tf.zeros([1, 224, 224, 3])), will go between the LenetModel() and the .summary()

# Now our model can be run


class LenetModel(Model):
    def __init__(self, filters, kernel_size, strides, padding, activation, pool_size):
        super(LenetModel, self).__init__()

        self.feature_extractor = FeatureExtractor(8, 3, 1, "valid", "relu", 2)

        self.flatten = Flatten()

        self.dense_1 = Dense(1000, activation = "relu")
        self.batch_1 = BatchNormalization()

        self.dense_2 = Dense(100, activation = "relu")
        self.batch_2 = BatchNormalization()

        self.dense_3 = Dense(1, activation = "sigmoid")


    def call(self, x, training):
        x = self.feature_extractor(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.batch_1(x)
        x = self.dense_2(x)
        x = self.batch_2(x)
        x = self.dense_3(x)

        return x

lenet_sub_classed = LenetModel(8, 3, 1, "valid", "relu", 2)
lenet_sub_classed(tf.zeros([1, 224, 224, 3]))
lenet_sub_classed.summary()

# Here we have our summary, now we can compile.

# First we will change the lenet_model to lenet_sub_classed to compile our code.

# And then we're going to fit the lenet_sub_classed to get the history and everything should run fine.


lenet_sub_classed.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = "accuracy")

history = lenet_sub_classed.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

# Note: We had to modify our dense_3 from 10 t0 1 in order to avoid an error. This is the error we get when trying
#to run our code in its original form. (See line 464)

# ValueError: `logits` and `labels` must have the same shape, received ((None, 10) vs (None, 1)).

# With the dense layer modified, we can see that we are getting a similar result when we compile and run our 
#model.fit, when we compare this to what we have with a functional API and sequential API.
