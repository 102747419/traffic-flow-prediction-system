import math
import os
import sys
import tkinter as tk

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.engine.training import Model
from keras.layers import Dense, Dropout
from keras.layers.core import Activation
from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential
from keras.saving import save
from sklearn.preprocessing import MinMaxScaler

import astar

lag = 8
train_file = "data/train-data.csv"
test_file = "data/test-data.csv"
shaped_models = ["lstm", "gru", "gru2"]
config = {"batch": 40, "epochs": 4}

graph = {
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


def load_data():
    """
    Loads the data from the csv file and returns a dataframe of the data.
    """

    # Read data from csv files
    sites = pd.read_csv("data/scats-sites.csv")
    data = pd.read_csv("data/scats-data.csv")

    # Filter interseciton sites
    sites = sites[sites["Site Type"].eq("INT")]
    data = data[data["SCATS Number"].isin(sites["Site Number"])]

    # Filter out data at (0,0)
    data = data[(data["NB_LATITUDE"] != 0) & (data["NB_LONGITUDE"] != 0)]

    # Offset positions to align with map
    data["NB_LATITUDE"] = data["NB_LATITUDE"].add(0.0015)
    data["NB_LONGITUDE"] = data["NB_LONGITUDE"].add(0.0013)

    # Assign unique ID to connections
    n_prev, lat_prev, lon_prev = None, None, None
    index = -1
    col = []

    for i, row in data.iterrows():
        if row["Location"] != n_prev or row["NB_LATITUDE"] != lat_prev or row["NB_LONGITUDE"] != lon_prev:
            n_prev = row["Location"]
            lat_prev = row["NB_LATITUDE"]
            lon_prev = row["NB_LONGITUDE"]
            index += 1
        col.append(index)

    data.insert(0, "id", col)

    return data


def generate_intersections(data):
    """
    Generates a dataframe of intersections with the traffic volume data averaged for each day.
    """

    test = pd.DataFrame(columns=["SCATS Number",
                                 "V00", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09",
                                 "V10", "V11", "V12",
                                 "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22",
                                 "V23", "V24", "V25",
                                 "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35",
                                 "V36", "V37", "V38",
                                 "V39", "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48",
                                 "V49", "V50", "V51",
                                 "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61",
                                 "V62", "V63", "V64",
                                 "V65", "V66", "V67", "V68", "V69", "V70", "V71", "V72", "V73", "V74",
                                 "V75", "V76", "V77",
                                 "V78", "V79", "V80", "V81", "V82", "V83", "V84", "V85", "V86", "V87",
                                 "V88", "V89", "V90",
                                 "V91", "V92", "V93", "V94", "V95"])

    train = pd.DataFrame(columns=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE", "Date",
                                  "V00", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12",
                                  "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
                                  "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38",
                                  "V39", "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49", "V50", "V51",
                                  "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64",
                                  "V65", "V66", "V67", "V68", "V69", "V70", "V71", "V72", "V73", "V74", "V75", "V76", "V77",
                                  "V78", "V79", "V80", "V81", "V82", "V83", "V84", "V85", "V86", "V87", "V88", "V89", "V90",
                                  "V91", "V92", "V93", "V94", "V95"])

    connection_arrays = [[[0 for volume in range(96)] for day in range(31)] for sensor in range(10)]

    arr_index = 0
    prev_id = 0
    prev_site = data.iloc[0]["SCATS Number"]
    day = 0
    lat = 0
    long = 0
    for i, row in data.iterrows():
        id = row["id"]
        site = row["SCATS Number"]
        if site != prev_site:
            # Calculate averages and read to DF
            lat = lat / arr_index
            long = long / arr_index
            temp_list = [[0]*96]*31
            # Average volumes
            for d in range(0, 31):
                for v in range(0, 96):
                    temp_list[d][v] = 0
                    for sensor in range(0, arr_index):
                        temp_list[d][v] += connection_arrays[sensor][d][v]
                    temp_list[d][v] /= arr_index
                # Read into DataFrame
                date = f"10/{(d+1)}/2016"
                if d+1 == 3:
                    new_row = {"SCATS Number": prev_site,
                               "V00": temp_list[d][0], "V01": temp_list[d][1], "V02": temp_list[d][2],
                               "V03": temp_list[d][3], "V04": temp_list[d][4], "V05": temp_list[d][5],
                               "V06": temp_list[d][6], "V07": temp_list[d][7], "V08": temp_list[d][8],
                               "V09": temp_list[d][9], "V10": temp_list[d][10], "V11": temp_list[d][11],
                               "V12": temp_list[d][12],
                               "V13": temp_list[d][13], "V14": temp_list[d][14], "V15": temp_list[d][15],
                               "V16": temp_list[d][16], "V17": temp_list[d][17], "V18": temp_list[d][18],
                               "V19": temp_list[d][19], "V20": temp_list[d][20], "V21": temp_list[d][21],
                               "V22": temp_list[d][22], "V23": temp_list[d][23], "V24": temp_list[d][24],
                               "V25": temp_list[d][25],
                               "V26": temp_list[d][26], "V27": temp_list[d][27], "V28": temp_list[d][28],
                               "V29": temp_list[d][29], "V30": temp_list[d][30], "V31": temp_list[d][31],
                               "V32": temp_list[d][32], "V33": temp_list[d][33], "V34": temp_list[d][34],
                               "V35": temp_list[d][35], "V36": temp_list[d][36], "V37": temp_list[d][37],
                               "V38": temp_list[d][38],
                               "V39": temp_list[d][39], "V40": temp_list[d][40], "V41": temp_list[d][41],
                               "V42": temp_list[d][42], "V43": temp_list[d][43], "V44": temp_list[d][44],
                               "V45": temp_list[d][45], "V46": temp_list[d][46], "V47": temp_list[d][47],
                               "V48": temp_list[d][48], "V49": temp_list[d][49], "V50": temp_list[d][50],
                               "V51": temp_list[d][51],
                               "V52": temp_list[d][52], "V53": temp_list[d][53], "V54": temp_list[d][54],
                               "V55": temp_list[d][55], "V56": temp_list[d][56], "V57": temp_list[d][57],
                               "V58": temp_list[d][58], "V59": temp_list[d][59], "V60": temp_list[d][60],
                               "V61": temp_list[d][61], "V62": temp_list[d][62], "V63": temp_list[d][63],
                               "V64": temp_list[d][64],
                               "V65": temp_list[d][65], "V66": temp_list[d][66], "V67": temp_list[d][67],
                               "V68": temp_list[d][68], "V69": temp_list[d][69], "V70": temp_list[d][70],
                               "V71": temp_list[d][71], "V72": temp_list[d][72], "V73": temp_list[d][73],
                               "V74": temp_list[d][74], "V75": temp_list[d][75], "V76": temp_list[d][76],
                               "V77": temp_list[d][77],
                               "V78": temp_list[d][78], "V79": temp_list[d][79], "V80": temp_list[d][80],
                               "V81": temp_list[d][81], "V82": temp_list[d][82], "V83": temp_list[d][83],
                               "V84": temp_list[d][84], "V85": temp_list[d][85], "V86": temp_list[d][86],
                               "V87": temp_list[d][87], "V88": temp_list[d][88], "V89": temp_list[d][89],
                               "V90": temp_list[d][90],
                               "V91": temp_list[d][91], "V92": temp_list[d][92], "V93": temp_list[d][93],
                               "V94": temp_list[d][94], "V95": temp_list[d][95]}
                    test = test.append(new_row, ignore_index=True)
                else:
                    new_row = {"SCATS Number": prev_site, "NB_LATITUDE": lat, "NB_LONGITUDE": long, "Date": date,
                               "V00": temp_list[d][0], "V01": temp_list[d][1], "V02": temp_list[d][2],
                               "V03": temp_list[d][3], "V04": temp_list[d][4], "V05": temp_list[d][5],
                               "V06": temp_list[d][6], "V07": temp_list[d][7], "V08": temp_list[d][8],
                               "V09": temp_list[d][9], "V10": temp_list[d][10], "V11": temp_list[d][11],
                               "V12": temp_list[d][12],
                               "V13": temp_list[d][13], "V14": temp_list[d][14], "V15": temp_list[d][15],
                               "V16": temp_list[d][16], "V17": temp_list[d][17], "V18": temp_list[d][18],
                               "V19": temp_list[d][19], "V20": temp_list[d][20], "V21": temp_list[d][21],
                               "V22": temp_list[d][22], "V23": temp_list[d][23], "V24": temp_list[d][24],
                               "V25": temp_list[d][25],
                               "V26": temp_list[d][26], "V27": temp_list[d][27], "V28": temp_list[d][28],
                               "V29": temp_list[d][29], "V30": temp_list[d][30], "V31": temp_list[d][31],
                               "V32": temp_list[d][32], "V33": temp_list[d][33], "V34": temp_list[d][34],
                               "V35": temp_list[d][35], "V36": temp_list[d][36], "V37": temp_list[d][37],
                               "V38": temp_list[d][38],
                               "V39": temp_list[d][39], "V40": temp_list[d][40], "V41": temp_list[d][41],
                               "V42": temp_list[d][42], "V43": temp_list[d][43], "V44": temp_list[d][44],
                               "V45": temp_list[d][45], "V46": temp_list[d][46], "V47": temp_list[d][47],
                               "V48": temp_list[d][48], "V49": temp_list[d][49], "V50": temp_list[d][50],
                               "V51": temp_list[d][51],
                               "V52": temp_list[d][52], "V53": temp_list[d][53], "V54": temp_list[d][54],
                               "V55": temp_list[d][55], "V56": temp_list[d][56], "V57": temp_list[d][57],
                               "V58": temp_list[d][58], "V59": temp_list[d][59], "V60": temp_list[d][60],
                               "V61": temp_list[d][61], "V62": temp_list[d][62], "V63": temp_list[d][63],
                               "V64": temp_list[d][64],
                               "V65": temp_list[d][65], "V66": temp_list[d][66], "V67": temp_list[d][67],
                               "V68": temp_list[d][68], "V69": temp_list[d][69], "V70": temp_list[d][70],
                               "V71": temp_list[d][71], "V72": temp_list[d][72], "V73": temp_list[d][73],
                               "V74": temp_list[d][74], "V75": temp_list[d][75], "V76": temp_list[d][76],
                               "V77": temp_list[d][77],
                               "V78": temp_list[d][78], "V79": temp_list[d][79], "V80": temp_list[d][80],
                               "V81": temp_list[d][81], "V82": temp_list[d][82], "V83": temp_list[d][83],
                               "V84": temp_list[d][84], "V85": temp_list[d][85], "V86": temp_list[d][86],
                               "V87": temp_list[d][87], "V88": temp_list[d][88], "V89": temp_list[d][89],
                               "V90": temp_list[d][90],
                               "V91": temp_list[d][91], "V92": temp_list[d][92], "V93": temp_list[d][93],
                               "V94": temp_list[d][94], "V95": temp_list[d][95]}
                    train = train.append(new_row, ignore_index=True)
            prev_site = site
            arr_index = 0
            lat = 0
            long = 0

        if id != prev_id:
            prev_id = id
            arr_index += 1
            day = 0
            lat += row["NB_LATITUDE"]
            long += row["NB_LONGITUDE"]
        # Read volumes into array
        for v in range(0, 96):
            connection_arrays[arr_index][day][v] = row[v+11]

        day += 1

    return train, test


def process_data(train_path, test_path, lags):
    """
    Processes the data so it can be used for the model.
    """

    print("Begin processing data...")

    train_df = pd.read_csv(train_path, encoding="utf-8").fillna(0)
    test_df = pd.read_csv(test_path, encoding="utf-8").fillna(0)

    flattened_data = train_df.iloc[:, 4:].to_numpy().flatten().reshape(-1, 1)
    scaler = MinMaxScaler((0, 1)).fit(flattened_data)

    arr_X_train = []
    arr_y_train = []
    arr_X_test = []
    arr_y_test = []

    arr_X_train, arr_y_train = handle_data(train_df, scaler, False)
    arr_X_test, arr_y_test = handle_data(test_df, scaler, True)

    print("Finished processing data")

    return arr_X_train, arr_y_train, arr_X_test, arr_y_test, scaler


def handle_data(data, scaler, test):
    """
    Groups data into chunks of lags and creates a list of arrays.
    """

    arr_X, arr_y = [], []

    # Normalize data
    if test:
        flow1 = scaler.transform(data.iloc[:, 1:].values.reshape(-1, 1)).reshape(1, -1)[0]
    else:
        flow1 = scaler.transform(data.iloc[:, 4:].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Group data into arrays of 8 elements (defined by lags variable)
    container = []
    for i in range(lag, len(flow1)):
        line = math.floor(i/96)
        id = data["SCATS Number"].iloc[line]
        tuple = flow1[i - lag: i + 1]
        tuple = np.insert(tuple, 0, id)
        container.append(tuple)

    # Shuffle training data
    container = np.array(container)
    if not test:
        np.random.shuffle(container)

    # Separate label (y_...) from data (X_...)
    X_train = container[:, :-1]
    y_train = container[:, -1]

    # Add to the rest of the data
    arr_X.extend(X_train)
    arr_y.extend(y_train)

    # Convert to numpy arrays
    arr_X = np.array(arr_X)
    arr_y = np.array(arr_y)

    return arr_X, arr_y


def train(model_name):
    """
    Train the model.
    """

    X_train, y_train, _, _, _ = process_data(train_file, test_file, lag)

    model, train_func, name = get_model(model_name)

    if model_name in shaped_models:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    print(f"Training {name}...")
    train_func(model, X_train, y_train, name, config)
    print("Training complete!")


def train_model(model, X_train, y_train, name, config):
    """
    Train most models.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save("model/" + name + ".h5")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("model/" + name + " loss.csv", encoding="utf-8", index=False)


def get_model(name):
    """
    Get the model by its name.
    """

    if name == "lstm":
        return get_lstm([lag + 1, 64, 64, 1])
    if name == "relu":
        return get_relu([lag + 1, 100, 50, 75, 100, 1], name)
    if name == "thick":
        return get_relu([lag + 1, 400, 400, 400, 400, 1], name)
    if name == "saes":
        return get_saes([lag + 1, 400, 1, 400, 1, 400, 1, 400, 1])
    if name == "gru2":
        return get_gru2([lag + 1, 400, 400, 1])

    # Return gru by default
    return get_gru([lag + 1, 64, 64, 1])


def get_gru(layers):
    """
    Get the GRU model with the given layers.
    """

    model = Sequential()
    model.add(GRU(layers[1], input_shape=(layers[0], 1), return_sequences=True))
    model.add(GRU(layers[2]))
    model.add(Dropout(0.2))
    model.add(Dense(layers[3], activation="sigmoid"))

    return model, train_model, "gru"


def get_gru2(layers):
    """
    Get the GRU model with the given layers.
    """

    model = Sequential()
    model.add(GRU(layers[1], input_shape=(layers[0], 1), return_sequences=True))
    model.add(GRU(layers[2]))
    model.add(Dropout(0.2))
    model.add(Dense(layers[3], activation="sigmoid"))

    return model, train_model, "gru2"


def get_relu(layers, name):
    """
    Get the ReLU model with the given layers.
    """

    model = Sequential([
        Dense(1, input_shape=(layers[0],)),
        Dense(layers[1], activation='relu'),
        Dense(layers[2], activation='relu'),
        Dense(layers[3], activation='relu'),
        Dense(layers[4], activation='relu'),
        Dense(layers[5])])

    return model, train_model, name


def get_saes(layers):
    """
    Get the SAES model with the given layers.
    """

    model = Sequential([
        Dense(1, input_shape=(layers[0],)),
        Dense(layers[1], activation="relu"),
        Dense(layers[2], activation="relu"),
        Dense(layers[3], activation="relu"),
        Dense(layers[4], activation="relu"),
        Dense(layers[5], activation="relu"),
        Dense(layers[6], activation="relu"),
        Dense(layers[7])])

    return model, train_model, "saes"


def get_lstm(units):
    """
    Get the LSTM model with the given layers.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation="sigmoid"))

    return model, train_model, "lstm"


def test(model_name):
    """
    Test the model.
    """

    # Load the model
    model = save.load_model(f"model/{model_name}.h5")

    # Process the data
    _, _, X_test, y_test, scaler = process_data(train_file, test_file, lag)

    # Unscale the test labels
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # Reshape the test data so it works with the model
    if model_name in shaped_models:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # Predict using the model
    predicted = model.predict(X_test)

    # Unscale predicted data
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # Plot results!
    plot_results(y_test, predicted, model_name)


def plot_results(y_true, y_pred, name):
    """
    Plot the results on a graph.
    """

    day = 1
    data_range = range(day * 96, (day + 1) * 96)

    d = "2016-10-1 00:00"
    x = pd.date_range(d, periods=len(y_true[data_range]), freq="15min")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true[data_range], label="True Data")
    ax.plot(x, y_pred[data_range], label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel("Flow")

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


# https://stackoverflow.com/a/8922151/10456572
# Replaced with A*
def dfs(start_id, dest_id):
    """
    Depth-first search to find the shortest path between two intersections.
    """

    # Maintain a queue of paths
    queue = []

    # Push the first path into the queue
    queue.append([start_id])

    while queue:
        # Get the first path from the queue
        path = queue.pop(0)

        # Get the last node from the path
        node = path[-1]

        # Path found
        if node == dest_id:
            return path

        # Enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

    return None


def a_star(start_id, dest_id, start_time_minutes, visited):
    """
    A* search to find the shortest path between two intersections.
    """

    def neighbors(n):
        for n1 in graph[n]:
            yield n1

    def distance(n1, n2, time_minutes):
        added_cost = 0
        if n2 in visited:
            added_cost = math.log2(visited[n2] / 60 + 1)

        return travel_time_mins(n1, n2, time_minutes) + added_cost

    path = list(astar.find_path(
        start_id, dest_id, start_time_minutes,
        neighbors_fnct=neighbors,
        heuristic_cost_estimate_fnct=distance_km,
        distance_between_fnct=distance))

    return path


def a_star_multiple(start_id, dest_id, start_time_minutes, routes=5, tries=500):
    """
    A* search to find multiple shortest paths between two intersections, sorted by travel time.
    """

    solutions = []
    visited = {}

    for i in range(tries):
        route = a_star(start_id, dest_id, start_time_minutes, visited)

        # Only add solution if it is unique
        if route not in solutions:
            solutions.append(route)

        # Increment visited count
        for id in route:
            if id not in visited:
                visited[id] = 1
            else:
                visited[id] += 1

        # Ensure max routes isnt exceeded
        if len(solutions) == routes:
            break

    # Sort solutions so they are in the correct order
    solutions = sorted(solutions, key=lambda x: total_travel_time_mins(x, start_time_minutes))

    return solutions


def distance_km(a_id, b_id):
    """
    Get the distance between two intersections.
    """

    a = intersections[intersections["SCATS Number"] == a_id].iloc[0]
    b = intersections[intersections["SCATS Number"] == b_id].iloc[0]

    delta_x = a["NB_LONGITUDE"] - b["NB_LONGITUDE"]
    delta_y = a["NB_LATITUDE"] - b["NB_LATITUDE"]

    # 1 degree latitude/longitude = 111km
    return math.sqrt(delta_x ** 2 + delta_y ** 2) * 111


def total_distance_km(route):
    """
    Get the total distance of a route.
    """

    dist = 0

    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        dist += distance_km(a, b)

    return dist


def predict_traffic_volume(site_id, time_index):
    """
    Predict the traffic volume at a site at a given time.
    """

    unique_sites = intersections.drop_duplicates("SCATS Number")
    scats_numbers = list(unique_sites["SCATS Number"].values)
    site_index = scats_numbers.index(site_id) * 96 + time_index

    return REGRESSION[site_index]


def get_interpolated_traffic_volume(site_id, time_minutes):
    """
    Get the interpolated traffic volume at a site at a given time.
    """

    # Calculate the time indices
    index1 = math.floor(time_minutes / 15) % 96
    index2 = math.ceil(time_minutes / 15) % 96

    # Calculate how much to interpolate
    t = (time_minutes % 15) / 15

    # Get the traffic volume at each time
    volume1 = predict_traffic_volume(site_id, index1)
    volume2 = predict_traffic_volume(site_id, index2)

    # Interpolate between the two traffic volumes
    return (1 - t) * volume1 + t * volume2


def get_traffic_volume(a_id, b_id, time_minutes):
    """
    Get the averaged interpolated traffic volume between two intersections at a given time.
    """

    # Calculate the duration of the journey between these two connected sites
    # Assuming we are going 60km/h, `time_h = distance_km / speed_kmh` simplifies to `time_m = distance_km`
    duration = distance_km(a_id, b_id)

    # Calculate the time at each site
    a_time = time_minutes
    b_time = a_time + duration

    # Calculate the interpolated traffic volume for each site
    a_volume = get_interpolated_traffic_volume(a_id, a_time)
    b_volume = get_interpolated_traffic_volume(b_id, b_time)

    # Average the two sites' traffic volumes
    return (a_volume + b_volume) / 2


def travel_time_mins(a_id, b_id, time_minutes):
    """
    Get the travel time between two intersections at a given time.
    """

    # Every 120 cars adds 1 minute to the travel time
    return distance_km(a_id, b_id) + get_traffic_volume(a_id, b_id, time_minutes) / 120


def total_travel_time_mins(route, start_time_minutes):
    """
    Get the total travel time of a route when starting at a given time.
    """

    time = start_time_minutes

    for i in range(len(route) - 1):
        time += travel_time_mins(route[i], route[i + 1], time)

    return time - start_time_minutes


def military_to_minutes(military):
    """
    Convert a military time string to minutes.
    Doesn't need to be rounded to the nearest 15 minutes.
    Format: HHMM
    Example: "0730"
    """

    hour = int(military[:2])
    minutes = int(military[2:])
    return hour * 60 + minutes


def format_duration(minutes):
    """
    Format a duration in minutes to a string.
    Format: HH:MM:SS
    Example: "00:30:20"
    """

    mins = math.floor(minutes)
    hours = math.floor(mins / 60)
    mins %= 60
    seconds = math.floor(minutes % 1 * 60)

    # Format as HH:MM:SS
    return f"{str(hours).zfill(2)}:{str(mins).zfill(2)}:{str(seconds).zfill(2)}"


def print_routes(routes, start_time_minutes):
    """
    Print the routes to the console with the intersections along the route, the total travel time, and the total distance.
    """

    for i, route in enumerate(routes):
        distance = total_distance_km(route)
        travel_time = total_travel_time_mins(route, start_time_minutes)

        print(f"===== Route {i + 1} =====")
        print(f"Route: {' → '.join(map(str, route))}")
        print(f"Distance: {round(distance, 2)}km")
        print(f"Duration: {format_duration(travel_time)}")


def show_routes_on_map(routes, start_time_minutes):
    """
    Show the routes on a map in a new browser window.
    """

    unique_sites = intersections.drop_duplicates("SCATS Number")
    scats_numbers = unique_sites["SCATS Number"].values
    latitudes = unique_sites["NB_LATITUDE"]
    longitudes = unique_sites["NB_LONGITUDE"]

    fig = go.Figure(go.Scattermapbox(
        name="Intersections",
        mode="markers",
        hovertext=[f"SCATS Number: {x}" for x in scats_numbers],
        lon=[x for x in longitudes],
        lat=[x for x in latitudes],
        marker={"size": 10}))

    # Enumerate over routes
    for i, route in enumerate(routes):
        travel_time = total_travel_time_mins(route, start_time_minutes)

        # Add route to map
        fig.add_trace(go.Scattermapbox(
            name=f"Route {i + 1} {format_duration(travel_time)}",
            mode="markers+lines",
            hovertext=[x for x in route],
            lon=[unique_sites.loc[unique_sites["SCATS Number"] == x, "NB_LONGITUDE"].values[0] for x in route],
            lat=[unique_sites.loc[unique_sites["SCATS Number"] == x, "NB_LATITUDE"].values[0] for x in route],
            marker={"size": 10}, line={"width": 4}))

    # Configure map
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Zoom in on the routes
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        mapbox=dict(
            bearing=0,
            center=dict(
                lat=latitudes.mean(),
                lon=longitudes.mean()
            ),
            pitch=0,
            zoom=12
        ),
    )

    # Open map in browser
    fig.show()


def calc_routes(start_id, dest_id, start_time_minutes):
    """
    Calculate the routes from a start and destination intersection at a given time.
    """

    if start_id == dest_id:
        print("Please specify different start and destination locations.")
        return

    print("Begin calculating routes...")
    # Get best routes
    routes = a_star_multiple(start_id, dest_id, start_time_minutes)

    print("Finished calculating routes")

    # Print routes to console
    print_routes(routes, start_time_minutes)

    # Show on map
    show_routes_on_map(routes, start_time_minutes)


def init(model_name):
    """
    Initialise stuff.
    """

    REGRESSION = None

    if not os.path.isfile(test_file) or not os.path.isfile(train_file):
        DATA = load_data()
        DATA, TEST_DATA = generate_intersections(DATA)
        DATA.to_csv(train_file, index=False)
        TEST_DATA.to_csv(test_file, index=False)
        DATA.to_csv(train_file, index=False)

    if os.path.isfile(f"model/{model_name}.h5"):
        MODEL = save.load_model(f"model/{model_name}.h5")
        _, _, test_x, _, g_scaler = process_data("data/train-data.csv", "data/test-data.csv", lag)
        if model_name in shaped_models:
            X_test = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        else:
            X_test = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        REGRESSION = MODEL.predict(X_test)
        REGRESSION = g_scaler.inverse_transform(REGRESSION.reshape(-1, 1)).reshape(1, -1)[0]
    else:
        print("Please train the model.")
        sys.exit()

    return REGRESSION, pd.read_csv(train_file)


def show_gui():
    """
    Show the GUI.
    """

    # Declare canvas dimensions
    width = 400
    height = 300

    # Create a window and canvas
    window = tk.Tk()
    window.title("Traffic Flow Prediction System")
    canvas = tk.Canvas(window, width=width, height=height)

    # Get SCATS numbers
    unique_sites = intersections.drop_duplicates("SCATS Number")
    scats_numbers = unique_sites["SCATS Number"].values

    # Generate values for dropdown menus
    hours_values = [x for x in range(24)]
    mins_values = [0, 15, 30, 45]

    # Create time dropdown menus
    hours = tk.IntVar(window)
    hours.set(hours_values[0])
    minutes = tk.IntVar(window)
    minutes.set(mins_values[0])

    # Create start and destination dropdown menus
    start = tk.StringVar(window)
    start.set(scats_numbers[0])
    dest = tk.StringVar(window)
    dest.set(scats_numbers[0])

    # Create labels
    lbl_heading = tk.Label(window, text="Traffic Flow Prediction System")
    lbl_heading.config(font=("helvetica", 20))
    lbl_time = tk.Label(window, text="Departure Time ", anchor="e", width=50)
    lbl_start = tk.Label(window, text="Starting SCATS site ", anchor="e", width=50)
    lbl_end = tk.Label(window, text="Destination SCATS site ", anchor="e", width=50)

    # Create dropdown menus
    drp_hour = tk.OptionMenu(window, hours, *hours_values)
    drp_minute = tk.OptionMenu(window, minutes, *mins_values)
    drp_start = tk.OptionMenu(window, start, *scats_numbers)
    drp_end = tk.OptionMenu(window, dest, *scats_numbers)

    # Configure dropdown menus
    drp_hour.config(width=3)
    drp_minute.config(width=3)
    drp_start.config(width=3)
    drp_end.config(width=3)

    # Create buttons
    btn_exit = tk.Button(window, text="Exit", width=14, command=window.destroy)
    btn_calculate = tk.Button(window, text="Calculate Routes", width=14, command=lambda: calc_routes(int(start.get()), int(dest.get()), int(hours.get() * 60 + minutes.get())))

    # Attach labels to canvas
    canvas.create_window(width / 2, 40, window=lbl_heading)
    canvas.create_window(20, 105, window=lbl_time)
    canvas.create_window(20, 145, window=lbl_start)
    canvas.create_window(20, 185, window=lbl_end)

    # Attach dropdown menus to canvas
    canvas.create_window(width / 2 + 40, 105, window=drp_hour)
    canvas.create_window(width / 2 + 105, 105, window=drp_minute)
    canvas.create_window(width / 2 + 40, 145, window=drp_start)
    canvas.create_window(width / 2 + 40, 185, window=drp_end)

    # Attach buttons to canvas
    canvas.create_window(width / 2 - 60, 260, window=btn_calculate)
    canvas.create_window(width / 2 + 60, 260, window=btn_exit)

    canvas.pack()
    window.mainloop()


if __name__ == "__main__":
    # MODEL TEST - TEMP
    train("gru2")
    # input("\nPress Enter to continue...")
    test("gru2")
    input("\nPress Enter to continue...")
    # train("relu")
    # train("thick")
    # train("saes")
    # input("\nPress Enter to continue...")
    # test("relu")
    # test("thick")
    # test("saes")
    # input("\nPress Enter to continue...")


    if False:
        ###### Command-line implementation ######

        start_id = int(sys.argv[1])
        dest_id = int(sys.argv[2])
        start_time_minutes = military_to_minutes(sys.argv[3])
        model_name = sys.argv[4].lower() if len(sys.argv) > 4 else "gru"

        REGRESSION, intersections = init(model_name)

        calc_routes(start_id, dest_id, start_time_minutes)
    else:
        ###### GUI implementation ######

        model_name = sys.argv[1].lower() if len(sys.argv) > 1 else "gru"

        REGRESSION, intersections = init(model_name)

        show_gui()
