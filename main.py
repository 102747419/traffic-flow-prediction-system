import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.layers import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.saving import save
from sklearn.preprocessing import MinMaxScaler

import astar

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

intersections = {}


def load_data():
    # Read data
    sites = pd.read_csv("data/scats-sites.csv")
    data = pd.read_csv("data/scats-data.csv")

    # Filter interseciton sites
    sites = sites[sites["Site Type"].eq("INT")]
    data = data[data["SCATS Number"].isin(sites["Site Number"])]

    # Filter out data at (0,0)
    # data = data[(data["NB_LATITUDE"] != 0) & (data["NB_LONGITUDE"] != 0)]

    # Offset positions to align with map
    data["NB_LATITUDE"] = data["NB_LATITUDE"].add(0.0015)
    data["NB_LONGITUDE"] = data["NB_LONGITUDE"].add(0.0013)

    # Assign unique ID to connections
    prev = None
    index = -1
    col = []

    for i, row in data.iterrows():
        if row["Location"] != prev:
            prev = row["Location"]
            index += 1
        col.append(index)

    data.insert(0, "id", col)

    return data, sites


def process_data(data, lags):
    flattened_data = data.iloc[:, 11:].to_numpy().flatten().reshape(-1, 1)
    scaler = MinMaxScaler((0, 1)).fit(flattened_data)

    arr_X_train = []
    arr_y_train = []
    arr_X_test = []
    arr_y_test = []

    for index, row in data.iterrows():
        # TEMPORARY SO ITS FASTER TO TEST
        if index > 100:
            break

        # read data
        id = row["SCATS Number"]
        site_data = row.iloc[11:].to_numpy().reshape(-1, 1)

        # normalize data
        flow1 = scaler.transform(site_data).reshape(1, -1)[0]
        flow2 = scaler.transform(site_data).reshape(1, -1)[0]

        flow1_copy = np.append(flow1, flow1)
        flow2_copy = np.append(flow2, flow2)

        # group data into arrays of 8 elements (defined by lags variable)
        train, test = [], []
        for i in range(len(flow1), len(flow1_copy)):
            arr = flow1_copy[i - lags: i + 1]
            np.insert(arr, 0, id)
            train.append(arr)
        for i in range(len(flow2), len(flow2_copy)):
            arr = flow2_copy[i - lags: i + 1]
            np.insert(arr, 0, id)
            test.append(arr)

        # shuffle training data
        train = np.array(train)
        test = np.array(test)
        np.random.shuffle(train)

        # separate label (y_...) from data (X_...)
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = test[:, :-1]
        y_test = test[:, -1]

        arr_X_train.extend(X_train)
        arr_y_train.extend(y_train)
        arr_X_test.extend(X_test)
        arr_y_test.extend(y_test)

    arr_X_train = np.array(arr_X_train)
    arr_y_train = np.array(arr_y_train)
    arr_X_test = np.array(arr_X_test)
    arr_y_test = np.array(arr_y_test)

    return arr_X_train, arr_y_train, arr_X_test, arr_y_test, scaler

    # # read data
    # site_data = data[data["id"] == 0].iloc[:, 11:].iloc[0].to_numpy().reshape(-1, 1)

    # # normalize data
    # scaler = MinMaxScaler((0, 1)).fit(site_data)
    # flow1 = scaler.transform(site_data).reshape(1, -1)[0]
    # flow2 = scaler.transform(site_data).reshape(1, -1)[0]

    # flow1_copy = np.append(flow1, flow1)
    # flow2_copy = np.append(flow2, flow2)

    # # group data into arrays of 8 elements (defined by lags variable)
    # train, test = [], []
    # for i in range(len(flow1), len(flow1_copy)):
    #     arr = flow1_copy[i - lags: i + 1]
    #     # np.insert(arr, 0, 0)
    #     train.append(arr)
    # for i in range(len(flow2), len(flow2_copy)):
    #     arr = flow2_copy[i - lags: i + 1]
    #     # np.insert(arr, 0, 0)
    #     test.append(arr)

    # # shuffle training data
    # train = np.array(train)
    # test = np.array(test)
    # np.random.shuffle(train)

    # # separate label (y_...) from data (X_...)
    # X_train = train[:, :-1]
    # y_train = train[:, -1]
    # X_test = test[:, :-1]
    # y_test = test[:, -1]

    # # return scalar so it can be unscaled using scaler.inverse_transform()
    # return X_train, y_train, X_test, y_test, scaler


def train_model():
    X_train, y_train, _, _, _ = process_data(data, lag)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model, name = get_gru([lag, 64, 64, 1])

    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save("model/" + name + ".h5")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("model/" + name + " loss.csv", encoding="utf-8", index=False)


def get_gru(units):
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation="sigmoid"))

    return model, "gru"


def test_model(id):
    # load the model
    model = save.load_model("model/gru.h5")

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
    plot_results(y_test, predicted, "gru")


def plot_results(y_true, y_pred, name):
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
def dfs(start_id, dest_id):
    # maintain a queue of paths
    queue = []

    # push the first path into the queue
    queue.append([start_id])

    while queue:
        # get the first path from the queue
        path = queue.pop(0)

        # get the last node from the path
        node = path[-1]

        # path found
        if node == dest_id:
            return path

        # enumerate all adjacent nodes, construct a
        # new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

    return None


def a_star(start_id, dest_id, visited):
    def neighbors(n):
        for n1 in graph[n]:
            yield n1

    def distance(n1, n2):
        added_cost = 0
        if n2 in visited:
            added_cost = math.log2(visited[n2] / 60 + 1)

        return travel_time_mins(n1, n2) + added_cost

    path = list(astar.find_path(
        start_id, dest_id,
        neighbors_fnct=neighbors,
        heuristic_cost_estimate_fnct=travel_time_mins,
        distance_between_fnct=distance))

    return path


def a_star_multiple(start_id, dest_id, routes=5, tries=500):
    solutions = []
    visited = {}

    for i in range(tries):
        route = a_star(start_id, dest_id, visited)

        # only add solution if it is unique
        if route not in solutions:
            solutions.append(route)

        for id in route:
            if id not in visited:
                visited[id] = 1
            else:
                visited[id] += 1

        # ensure max routes isnt exceeded
        if len(solutions) == routes:
            break

    return solutions


def sort_routes(routes):
    time = map(total_travel_time_mins, routes)
    return [x for _, x in sorted(zip(time, routes))]


def distance_km(a_id, b_id):
    a = intersections[a_id]
    b = intersections[b_id]

    a_x = a[2]
    a_y = a[1]

    b_x = b[2]
    b_y = b[1]

    diff_x = a_x - b_x
    diff_y = a_y - b_y

    return math.sqrt(diff_x ** 2 + diff_y ** 2) * 111


def total_distance_km(route):
    dist = 0

    for index, id in enumerate(route[:-1]):
        a = route[index]
        b = route[index + 1]
        dist += distance_km(a, b)

    return dist


def travel_time_mins(a_id, b_id):
    return distance_km(a_id, b_id) + 0.5


def total_travel_time_mins(route):
    return total_distance_km(route) + (len(route) - 1) * 0.5


def format_time(minutes):
    mins = math.floor(minutes)
    seconds = round((minutes - mins) * 60)

    text = f"{mins}m"

    if seconds > 0:
        text += f" {seconds}s"

    return text


lag = 8
config = {"batch": 50, "epochs": 20}
data, sites = load_data()

unique_connections = data.drop_duplicates("id")
scats_numbers = unique_connections["SCATS Number"].unique()

for id in scats_numbers:
    connections = unique_connections[unique_connections["SCATS Number"] == id]
    mean_latitude = connections["NB_LATITUDE"].mean()
    mean_longitude = connections["NB_LONGITUDE"].mean()
    intersections[id] = (id, mean_latitude, mean_longitude)

# train_model()
# test_model(4034)
routes = a_star_multiple(int(sys.argv[1]), int(sys.argv[2]))
routes = sort_routes(routes)

# Show sites on map
intersection_values = list(intersections.values())
fig = go.Figure(go.Scattermapbox(
    name="Intersections",
    mode="markers",
    lon=[x[2] for x in intersection_values],
    lat=[x[1] for x in intersection_values],
    marker={"size": 10}))

for i, route in enumerate(routes):
    distance = total_distance_km(route)
    travel_time = total_travel_time_mins(route)

    print("===== Route " + str(i + 1) + " =====")
    print("Route:", route)
    print("Distance (km):  ", distance)
    print("Duration (mins):", travel_time)

    fig.add_trace(go.Scattermapbox(
        name=f"Route {i + 1} {format_time(travel_time)}",
        mode="markers+lines",
        lon=[intersections[x][2] for x in route],
        lat=[intersections[x][1] for x in route],
        marker={"size": 10}, line={"width": 4}))


fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
