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


connections = {
    970: [2846, 3685],
    2000: [4043, 3685, 3682],
    2200: [4063],
    2820: [3662, 4321],
    2825: [4030],
    2827: [4051],
    2846: [970],
    3001: [3002, 4262, 3662],
    3002: [3001, 3662, 4263, 4035],
    3120: [3122, 4035, 4040],
    3122: [3127, 3120, 3804],
    3126: [3127, 3682],
    3127: [3126, 4063, 3122],
    3180: [4051, 4057],
    3662: [2820, 4335, 4324, 3002, 3001],
    3682: [3126, 3804, 2000],
    3685: [970, 2000],
    3804: [3812, 4040, 3122, 3682],
    3812: [3804],
    4030: [4321, 4051, 2825, 4032],
    4032: [4321, 4030, 4034, 4057],
    4034: [4035, 4032, 4324, 4063],
    4035: [3120, 3002, 4034],
    4040: [4043, 4272, 4266, 3120, 3804],
    4043: [2000, 4273, 4040],
    4051: [4030, 3180, 2827],
    4057: [3180, 4032, 4063],
    4063: [4057, 4034, 3127, 2200],
    4262: [4263, 3001],
    4263: [4262, 3002, 4264],
    4264: [4263, 4324, 4266, 4270],
    4266: [4264, 4040],
    4270: [4264, 4812, 4263, 4272],
    4272: [4273, 4040, 4270],
    4273: [4272, 4043],
    4321: [4335, 2820, 4030, 4032],
    4324: [3662, 4264, 4034],
    4335: [3662, 4321],
    4812: [4270, 4263],
    4821: [3001]
}

intersections = []


def load_data():
    # Read data
    sites = pd.read_csv('data/scats-sites.csv')
    data = pd.read_csv('data/scats-data-test.csv')

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
    flattened_data = data.iloc[:, 11:].to_numpy().flatten().reshape(-1, 1)
    scaler = MinMaxScaler((0, 1)).fit(flattened_data)

    train, test, validation = split_data(data)

    arr_x_train, arr_y_train = [], []

    test_arr = [], []

    arr_x_train, arr_y_train = process_datapool(train, lags, scaler, True)
    arr_x_test, arr_y_test = process_datapool(test, lags, scaler, False)
    arr_x_valid, arr_y_valid = process_datapool(validation, lags, scaler, False)

    # Convert to numpy arrays
    arr_x_train = np.array(arr_x_train)
    arr_y_train = np.array(arr_y_train)
    arr_x_test = np.array(arr_x_test)
    arr_y_test = np.array(arr_y_test)
    arr_x_valid = np.array(arr_x_valid)
    arr_y_valid = np.array(arr_y_valid)

    return arr_x_train, arr_y_train, arr_x_test, arr_y_test, arr_x_valid, arr_y_valid, scaler


def process_datapool(data, lags, scaler, shuffle):
    x = []
    y = []

    for index, row in data.iterrows():
        # read data
        id = row['SCATS Number']
        site_data = row.iloc[11:].to_numpy().reshape(-1, 1)

        # normalize data
        flow1 = scaler.transform(site_data).reshape(1, -1)[0]
        flow2 = scaler.transform(site_data).reshape(1, -1)[0]

        flow1_copy = np.append(flow1, flow1)
        flow2_copy = np.append(flow2, flow2)

        # group data into arrays of 8 elements (defined by lags variable)
        container = []
        for i in range(len(flow1), len(flow1_copy)):
            arr = flow1_copy[i - lags: i + 1]
            np.insert(arr, 0, id)
            container.append(arr)

        # shuffle training data
        container = np.array(container)
        if (shuffle):
            np.random.shuffle(container)

        # separate label (y_...) from data (X_...)
        X = container[:, :-1]
        Y = container[:, -1]

        x.extend(X)
        y.extend(Y)

    return x, y


def split_data(data):
    count = 0
    prev_id = data.iloc[0].iloc[0]

    train = pd.DataFrame()
    test = pd.DataFrame()
    validation = pd.DataFrame()

    for i, row in data.iterrows():
        id = row.iloc[0]
        if id != prev_id:
            prev_id = id
            count = 0

        if count < 7:
            test = test.append(row, ignore_index=True)
        elif count < 9:
            validation = validation.append(row, ignore_index=True)
        else:
            train = train.append(row, ignore_index=True)

    return train, test, validation


def train_model():
    X_train, y_train, _, _, X_valid, y_valid, _ = process_data(data, lag)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model, name = get_gru([lag, 64, 64, 1])

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config['batch'],
        epochs=config['epochs'],
        validation_data=(X_valid, y_valid))

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


def test_model(id):
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
    day = 1
    data_range = range(day * 96, (day + 1) * 96)

    d = '2016-10-1 00:00'
    x = pd.date_range(d, periods=len(y_true[data_range]), freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true[data_range], label='True Data')
    ax.plot(x, y_pred[data_range], label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


lag = 8
config = {'batch': 50, 'epochs': 20}
data, sites = load_data()

unique_connections = data.drop_duplicates("id")
scats_numbers = unique_connections["SCATS Number"].unique()

for id in scats_numbers:
    connections = unique_connections[unique_connections["SCATS Number"] == id]
    mean_latitude = connections["NB_LATITUDE"].mean()
    mean_longitude = connections["NB_LONGITUDE"].mean()
    intersections.append((id, mean_latitude, mean_longitude))

train_model()
test_model(4034)

# Show sites on map
fig = px.scatter_mapbox(data, lat=[x[1] for x in intersections], lon=[x[2] for x in intersections], hover_name=[x[0] for x in intersections],
                        color_discrete_sequence=['fuchsia'], zoom=8)
fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
fig.show()
