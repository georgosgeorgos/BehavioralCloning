from keras.models import load_model
from sklearn.model_selection import train_test_split

from NVIDIA_model import Net as nvidia
from utils import generate_labels_track1, data_generator


# preprocess dataset for track1
labels = generate_labels_track1()
# split dataset in train and validation set
train_samples, validation_samples = train_test_split(labels, test_size=0.2)
# build generators
train_generator = data_generator(train_samples)
validation_generator = data_generator(validation_samples)
# build model
model = nvidia(input_shape=(160,320,3), crop=((50,20), (0,0)))
model.compile(loss='mse', optimizer="adam")
model.summary()

train_size = (len(train_samples) // 32) * 32
valid_size = (len(validation_samples) // 32) * 32

# fit model using generators
h = model.fit_generator(train_generator, 
	samples_per_epoch=train_size, 
	validation_data=validation_generator, 
	nb_val_samples=valid_size, 
	nb_epoch=40, 
	verbose=1)

# save model
#model.save('NVIDIA_model.h5')