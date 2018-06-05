import numpy as np
import keras
import cv2

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=16, dim=(16,128,128), n_channels=3,
                 n_classes=200, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.source = '/datahdd/Dataset/Moments_in_Time_Mini/processed/'
        self.txt_category = "/datahdd/Dataset/Moments_in_Time_Mini/moments_categories.txt"
        self.category_name = np.loadtxt(self.txt_category, delimiter=',', usecols=0, dtype=str, encoding='utf-8')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, 16,128,128 ))
        y = np.empty((self.batch_size), dtype=int)
        image_stack = []


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            random_int = np.random.randint(150)

            # Store sample
            for frame_number in range(16):

                current_frame = int((frame_number +random_int)/2)
                if current_frame < 10 :
                    str_current_frame = '0'+str(current_frame)
                else :
                    str_current_frame = str(current_frame)

                image = cv2.imread(self.source + '/' + ID + '/' + str_current_frame + '.jpg')
                resized_image = cv2.resize(image, dsize = (128, 128))
                image_stack.append(resized_image)

            image_volume = np.array(image_stack)
            image_volume = np.reshape(image_volume, (3, 16, 128, 128))

            X[i,] = image_volume
            image_stack = []

            # Store class
            current_order = np.where(self.list_IDs == ID)
            current_class = self.labels[current_order]
            current_classnumber = np.where(self.category_name == current_class)
            current_category = current_classnumber[0]
            y[i] = current_category

        print (y)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)