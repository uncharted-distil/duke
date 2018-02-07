import os

import numpy as numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense

from utils import normalize_text
import numpy.random as rn


def main(m=1000, n=100):

    # from keras.models import Model

    # # This returns a tensor
    #     inputs = Input(shape=(n,))

    #     # a layer instance is callable on a tensor, and returns a tensor
    #     x = Dense(64, activation='relu')(inputs)
    #     x = Dense(64, activation='relu')(x)
    #     predictions = Dense(1, activation='linear')(x)
    #     # predictions = Dense(10, activation='softmax')(x)

    #     # This creates a model that includes
    #     # the Input layer and two Dense layers
    #     model = Model(inputs=inputs, outputs=predictions)
    #     # model.compile(optimizer='rmsprop',
    #     model.compile(optimizer='sgd',
    #                   loss='mean_squared_error',
    #                   metrics=['accuracy'])
    #     model.fit(data, labels)

    x_train = rn.randn(m, n)
    y_train = rn.randn(m)
    x_test = rn.randn(m, n)
    y_test = rn.randn(m)

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=n))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  #   metrics=['mean_squared_error'],
                  )
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    # model.add(Dense(units=10, activation='softmax'))

    model.fit(x_train, y_train, epochs=5, batch_size=32)
    # model.train_on_batch(x_batch, y_batch)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss:', loss_and_metrics)

    prediction = model.predict(x_test, batch_size=128)

    # define / compile model
    # load dataset
    # optimize
    # evaluate


def preprocess_ckan_data(data_dir='data/ckan'):

    # create top-level preprocessed dir if it doesn't exist
    preprocessed_path = data_dir + 'preprocessed'
    if not os.path.isdir(preprocessed_path):
        os.mkdir(preprocessed_path)

    dataset_dirs = glob.glob('{data_dir}/original/*'.format(data_dir=data_dir))
    for ds_path in dataset_dirs:

        ds_name = ds_path.split('/')[-1]
        # create dataset dir in preprocessed dir if it doesn't exist
        preprocessed_dir = '{0}/preprocessed/{1}'.format(data_dir, ds_name)
        if not os.path.isdir(preprocessed_dir):
            os.mkdir(preprocessed_dir)

        # get metadata
        metadata_fname = '{0}/{1}_metadata.json'.format(ds_path, ds_name)
        with open(metadata_fname) as md_file:
            metadata = json.load(md_file)

        # extract tags from metadata and write to file
        tags = [normalize_text(tag['name']) for tag in metadata['tags']]  # TODO sufficient normalization for tags?
        tag_fname = '{0}/{1}_tags.json'.format(preprocessed_dir, ds_path)
        with open(tag_fname, 'w') as tag_file:
            json.dump(tag_file)

        data_files = glob.glob('{0}/original/*'.format(ds_path))
        for data_file in data_files:
            file_type = data_file.split('.')[-1].lower()
            if file_type in ['xls', 'xlsx']:
                # TODO convert excel to csv
                pass
            elif file_type is 'csv':
                data_frame = pd.read_csv(data_file)
                # TODO clean / normalize
            else:
                print('invalid file type:', file_type)

            file_name = data_file.split('/')[-1]
            data_frame.to_csv('{0}/{1}.csv'.format(preprocessed_dir, file_name), index=False)


if __name__ == '__main__':
    main()
