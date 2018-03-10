import os

import numpy as numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, GRU, LSTM

from utils import normalize_text
import numpy.random as rn
import numpy as np

from keras.utils import to_categorical


# define / compile model
# load dataset
# drop missing / clean / format / normalize
# optimize
# evaluate

def extract_tags(metadata):
    # extract tags from metadata and write to file
    return [normalize_text(tag['name']) for tag in metadata['tags']]  # TODO sufficient normalization for tags?


def dataset_classifier(data_path='data/example'):

    data = pd.read_csv(data_path, header=0)
    print(data)

    # lowercase : boolean, default=True
    #     binary : boolean, default=False.
    # def load_dataset(self, dataset, columns=None, drop_nan=True, reset_data=True):
    # self.vprint('loading dataset {0}'.format(dataset if isinstance(dataset, str) else 'from pandas DataFrame'))

    # read csv assuming first line has header text. TODO handle files w/o headers

    # headers = dataset.columns.values
    # if columns:
    #     text_df = dataset[columns]
    # else:
    #     text_df = dataset.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
    # # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')

    # dtype_dropped = get_dropped(headers, text_df.columns.values)
    # self.vprint('\ndropped non-text columns: {0}'.format(list(dtype_dropped)))

    # TODO fill_nan
    # if drop_nan:  # drop columns if there are any missing values
    #     # TODO handle missing values w/o dropping whole column
    #     text_df = text_df.dropna(axis=1, how='any')
    #     nan_dropped = get_dropped(headers, text_df.columns.values)
    #     nan_dropped = nan_dropped.difference(dtype_dropped)
    #     if nan_dropped:
    #         self.vprint('\ndropped columns with missing values: {0}'.format(list(nan_dropped)))

    # self.data = {}
    # self.data['headers'] = self.format_data(headers)

    # for col in text_df.columns.values:
    #     self.vprint('normalizing column: {0}'.format(normalize_text(col, to_list=False)))
    #     formatted_data = self.format_data(text_df[col].values)
    #     if len(formatted_data) > 0:  # if not all values were removed for being out of vocab, add col to data
    #         self.data[normalize_text(col, to_list=False)] = formatted_data
    #     else:
    #         self.vprint('\nwarning: all rows removed by formatting (out of vocab?) in column', col, '\n')

    # unique_words = set()
    # for key in self.data:
    #     for row in self.data[key]:
    #         for term in row:
    #             unique_words.add(term)
    # num_terms = sum([len(self.data[key]) for key in self.data.keys()])
    # percent_unique = float(len(unique_words)) / float(num_terms)
    # self.metadata = {
    #     'num_columns': len(self.data),
    #     'num_terms': num_terms,
    #     'unique_terms': len(unique_words),
    #     'percent_unique': percent_unique
    # }

    # return self.data


def dynamic_net(n_samples=1024, dim=64, n_steps=32):

    A = rn.randn(dim, dim)
    b = rn.randn(dim)
    x = [rn.randn(dim, n_samples)]
    y = rn.randint(2, size=n_samples)
    y = to_categorical(y)

    for s in range(rn.randint(n_steps)):
        x.append(np.dot(A, x[-1]))

    x = np.array(x)
    x = np.rollaxis(x, 2)  # change shape from n_steps, dim, n_samples -> n_samples, n_steps, dim
    n_train = n_samples//2

    print('\nbuilding model')
    model = Sequential()
    model.add(LSTM(units=64,
                   activation="tanh",
                   recurrent_activation="hard_sigmoid",
                   input_shape=(None, dim),
                   return_sequences=True
                   )
              )
    model.add(LSTM(units=64,
                   activation="tanh",
                   recurrent_activation="hard_sigmoid",
                   )
              )
    model.add(Dense(units=2, activation='softmax'))

    print('\ncompiling model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'],
                  )

    print('\ncompiled')

    x_train = x[:n_train, :, :]
    x_test = x[n_train:, :, :]
    y_train = y[:n_train]
    y_test = y[n_train:]
    print('fitting')
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    print('evaluation')
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)
    print('predicting')
    # n_samples, n_steps, dim
    prediction = model.predict(x_test[:, :n_steps//3, :], batch_size=128)

    print('prediction', prediction)
    print('prediction shape', prediction.shape)

# keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
#                  activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)


def basic_net(m=1000, n=100):

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
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    # model.train_on_batch(x_batch, y_batch)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss:', loss_and_metrics)

    prediction = model.predict(x_test, batch_size=128)


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
        extract_tags(metadata)
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
    # main()

    # dynamic_net()
    dataset_classifier()
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

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    # model.add(Dense(units=10, activation='softmax'))
