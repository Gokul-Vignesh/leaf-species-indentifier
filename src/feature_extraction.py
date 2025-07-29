from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataset(data_dir, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train = datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, subset='training')
    val = datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, subset='validation')
    return train, val

