import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from keras.layers import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.saving import save
from sklearn.preprocessing import MinMaxScaler


class Intersection:

    def __init__(self, id, connections=[]) -> None:
        self.id = id
        self.connections = connections

    def find_intersection(self, connection):
        connections.get(connection)


connections = {
    # 2825: Intersection(2825, [4030]),
    # 2827: Intersection(2827, [4051]),
    # 2820: Intersection(2820, [3662, 4321]),
    # 4030: Intersection(4030, [4321]),
    # 4063: Intersection(4063, [4057, 2200, 3127, 4034])
    # 970: ,
    # 2000: ,
    # 2200: ,
    # 2820: ,
    # 2825: ,
    # 2827: ,
    # 2846: ,
    # 3001: ,
    # 3002: ,
    # 3120: ,
    # 3122: ,
    # 3126: ,
    # 3127: ,
    # 3180: ,
    # 3662: ,
    # 3682: ,
    # 3685: ,
    # 3804: ,
    # 3812: ,
    # 4030: ,
    # 4032: ,
    # 4034: ,
    # 4035: ,
    # 4040: ,
    # 4043: ,
    # 4051: ,
    # 4057: ,
    # 4063: ,
    # 4262: ,
    # 4263: ,
    # 4264: ,
    # 4266: ,
    # 4270: ,
    # 4272: ,
    # 4273: ,
    # 4321: ,
    # 4324: ,
    # 4335: ,
    # 4812: ,
    # 4821
}


# def get_connection(scats_number: int, connection_number: int) -> Intersection:
#     intersection = connections[scats_number]
#     if intersection == None:
#         return None

#     index = intersection[1].index(connection_number)
#     if index < 0:
#         return None

#     return connections[intersection[1][index]]


# print(get_connection(2825, 4030))

def load_data():
    # Read data
    sites = pd.read_csv('data/scats-sites.csv')
    data = pd.read_csv('data/scats-data.csv')

    # Filter interseciton sites
    sites = sites[sites['Site Type'].eq('INT')]
    data = data[data['SCATS Number'].isin(sites['Site Number'])]

    # Filter out data at (0,0)
    # data = data[(data['NB_LATITUDE'] != 0) & (data['NB_LONGITUDE'] != 0)]

    # Offset positions to align with map
    data['NB_LATITUDE'] = data['NB_LATITUDE'].add(0.0015)
    data['NB_LONGITUDE'] = data['NB_LONGITUDE'].add(0.0013)

    # Assign unique ID to connections
    prev = None
    index = -1
    col = []

    for i, row in data.iterrows():
        if row['Location'] != prev:
            prev = row['Location']
            index += 1
        col.append(index)

    data.insert(0, 'id', col)

    return data, sites


def process_data(data, lags):
    # read data
    site_data = data[data['id'] == 0].iloc[:, 11:].iloc[0].to_numpy().reshape(-1, 1)

    # normalize data
    scaler = MinMaxScaler((0, 1)).fit(site_data)
    flow1 = scaler.transform(site_data).reshape(1, -1)[0]
    flow2 = scaler.transform(site_data).reshape(1, -1)[0]

    # group data into arrays of 12 elements (defined by lags variable)
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    # shuffle training data
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    # separate label (y_...) from data (X_...)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    # return scalar so it can be unscaled using scaler.inverse_transform()
    return X_train, y_train, X_test, y_test, scaler


def train_model():
    X_train, y_train, _, _, _ = process_data(data, lag)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model, name = get_gru([lag, 64, 64, 1])

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config['batch'],
        epochs=config['epochs'],
        validation_split=0.05)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def get_gru(units):
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model, 'gru'


def test_model():
    # load the model
    model = save.load_model('model/gru.h5')

    # process the data
    _, _, X_test, y_test, scaler = process_data(data, lag)

    # unscale the test labels
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # reshape the test data so it works with the model
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # predict using the model
    predicted = model.predict(X_test)

    # unscale predicted data
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # plot results!
    plot_results(y_test, predicted, 'gru')


def plot_results(y_true, y_pred, name):
    d = '2016-10-1 ' + str(lag / 4) + ':00'
    x = pd.date_range(d, periods=len(y_true), freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


lag = 8
config = {'batch': 50, 'epochs': 50}
data, sites = load_data()

train_model()
test_model()

# Show sites on map
# fig = px.scatter_mapbox(data, lat='NB_LATITUDE', lon='NB_LONGITUDE', hover_name='Location', hover_data=['SCATS Number', 'id'],
#                         color_discrete_sequence=['fuchsia'], zoom=8)
# fig.update_layout(mapbox_style='open-street-map')
# fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
# fig.show()
