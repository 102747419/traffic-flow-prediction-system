import math
import sys

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
    # Read data from csv files
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
        # Read data
        id = row["SCATS Number"]
        site_data = row.iloc[11:].to_numpy().reshape(-1, 1)

        # Normalize data
        flow1 = scaler.transform(site_data).reshape(1, -1)[0]
        flow2 = scaler.transform(site_data).reshape(1, -1)[0]

        flow1_copy = np.append(flow1, flow1)
        flow2_copy = np.append(flow2, flow2)

        # Group data into arrays of 8 elements (defined by lags variable)
        train, test = [], []
        for i in range(len(flow1), len(flow1_copy)):
            arr = flow1_copy[i - lags: i + 1]
            np.insert(arr, 0, id)
            train.append(arr)
        for i in range(len(flow2), len(flow2_copy)):
            arr = flow2_copy[i - lags: i + 1]
            np.insert(arr, 0, id)
            test.append(arr)

        # Shuffle training data
        train = np.array(train)
        test = np.array(test)
        np.random.shuffle(train)

        # Separate label (y_...) from data (X_...)
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = test[:, :-1]
        y_test = test[:, -1]

        # Add to the rest of the data
        arr_X_train.extend(X_train)
        arr_y_train.extend(y_train)
        arr_X_test.extend(X_test)
        arr_y_test.extend(y_test)

    # Convert to numpy arrays
    arr_X_train = np.array(arr_X_train)
    arr_y_train = np.array(arr_y_train)
    arr_X_test = np.array(arr_X_test)
    arr_y_test = np.array(arr_y_test)

    return arr_X_train, arr_y_train, arr_X_test, arr_y_test, scaler


def train(model_name):
    X_train, y_train, _, _, _ = process_data(data, lag)

    model, train_func, name = get_model(model_name)

    if name == "saes":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(f"Training {name}...")
    train_func(model, X_train, y_train, name, config)
    print("Training complete!")


def train_model(model, X_train, y_train, name, config):
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save("model/" + name + ".h5")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("model/" + name + " loss.csv", encoding="utf-8", index=False)


def train_saes(models, X_train, y_train, name, config):
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer("hidden").output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer("hidden").get_weights()
        saes.get_layer(f"hidden{i + 1}").set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def get_model(name):
    if name == "saes":
        return get_saes([lag, 400, 400, 400, 1])
    if name == "lstm":
        return get_lstm([lag, 64, 64, 1])

    # Return gru by default
    return get_gru([lag, 64, 64, 1])


def get_gru(layers):
    model = Sequential()
    model.add(GRU(layers[1], input_shape=(layers[0], 1), return_sequences=True))
    model.add(GRU(layers[2]))
    model.add(Dropout(0.2))
    model.add(Dense(layers[3], activation="sigmoid"))

    return model, train_model, "gru"


def get_lstm(units):
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model, train_model, "lstm"


def get_sae(inputs, hidden, output):
    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name="hidden"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation="sigmoid"))

    return model, train_model, "sae"


def get_saes(layers):
    sae1 = get_sae(layers[0], layers[1], layers[-1])
    sae2 = get_sae(layers[1], layers[2], layers[-1])
    sae3 = get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name="hidden1"))
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers[2], name="hidden2"))
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers[3], name="hidden3"))
    saes.add(Activation("sigmoid"))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation="sigmoid"))

    models = [sae1, sae2, sae3, saes]

    return models, train_saes, "saes"


def test_model(model_name):
    # Load the model
    model = save.load_model(f"model/{model_name}.h5")

    # Process the data
    _, _, X_test, y_test, scaler = process_data(data, lag)

    # Unscale the test labels
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # Reshape the test data so it works with the model
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict using the model
    predicted = model.predict(X_test)

    # Unscale predicted data
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # Plot results!
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

    return solutions


def sort_routes(routes, start_time_minutes):
    return sorted(routes, key=lambda x: total_travel_time_mins(x, start_time_minutes))


def distance_km(a_id, b_id):
    a = intersections[a_id]
    b = intersections[b_id]

    delta_x = a[2] - b[2]
    delta_y = a[1] - b[1]

    # 1 degree latitude/longitude = 111km
    return math.sqrt(delta_x ** 2 + delta_y ** 2) * 111


def total_distance_km(route):
    dist = 0

    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        dist += distance_km(a, b)

    return dist


def predict_traffic_volume(site_id, time_index):
    # # Load the model
    # model = save.load_model(f"model/{model_name}.h5")

    # # Process the data
    # _, _, X_test, y_test, scaler = process_data(data, lag)

    # # Unscale the test labels
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # # Reshape the test data so it works with the model
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # # Predict using the model
    # predicted = model.predict(X_test)

    # # Unscale predicted data
    # predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # return predicted[time_index]
    return intersections[site_id][3][time_index]


def get_interpolated_traffic_volume(site_id, time_minutes):
    # Calculate the time indices
    index1 = math.floor(time_minutes / 15)
    index2 = math.ceil(time_minutes / 15)

    # Calculate how much to interpolate
    t = (time_minutes % 15) / 15

    # Get the traffic volume at each time
    volume1 = predict_traffic_volume(site_id, index1)
    volume2 = predict_traffic_volume(site_id, index2)

    # Interpolate between the two traffic volumes
    return (1 - t) * volume1 + t * volume2


def get_traffic_volume(a_id, b_id, time_minutes):
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
    # Every 120 cars adds 1 minute to the travel time
    return distance_km(a_id, b_id) + get_traffic_volume(a_id, b_id, time_minutes) / 120


def total_travel_time_mins(route, start_time_minutes):
    time = start_time_minutes

    for i in range(len(route) - 1):
        time += travel_time_mins(route[i], route[i + 1], time)

    return time - start_time_minutes


def military_to_minutes(military):
    hour = int(military[:2])
    minutes = int(military[2:])
    return hour * 60 + minutes


def format_duration(minutes):
    mins = math.floor(minutes)
    hours = math.floor(mins / 60)
    mins %= 60
    seconds = round((minutes - mins) * 60)

    # Format as HH:MM:SS
    return f"{str(hours).zfill(2)}:{str(mins).zfill(2)}:{str(seconds).zfill(2)}"


lag = 8
config = {"batch": 50, "epochs": 20}
data, sites = load_data()

unique_connections = data.drop_duplicates("id")
scats_numbers = unique_connections["SCATS Number"].unique()

for id in scats_numbers:
    # Find a row for each connection at this site
    connections = unique_connections[unique_connections["SCATS Number"] == id]

    # Find the mean position of all connections at this site
    mean_latitude = connections["NB_LATITUDE"].mean()
    mean_longitude = connections["NB_LONGITUDE"].mean()

    # Average the volume data for all connections at this site
    times = connections.iloc[:, 11:].to_numpy()
    avg_times = [np.mean(k) for k in zip(*times)]

    # Save the intersection to the dictionary
    intersections[id] = (id, mean_latitude, mean_longitude, avg_times)

model_name = sys.argv[4].lower() if len(sys.argv) > 4 else "gru"

# train(model_name)
# test_model()

start_time_minutes = military_to_minutes(sys.argv[3])
routes = a_star_multiple(int(sys.argv[1]), int(sys.argv[2]), start_time_minutes)
routes = sort_routes(routes, start_time_minutes)

# Show map
intersection_values = list(intersections.values())
fig = go.Figure(go.Scattermapbox(
    name="Intersections",
    mode="markers",
    hovertext=[f"SCATS Number: {x[0]}" for x in intersection_values],
    lon=[x[2] for x in intersection_values],
    lat=[x[1] for x in intersection_values],
    marker={"size": 10}))

# Enumerate over routes
for i, route in enumerate(routes):
    distance = total_distance_km(route)
    travel_time = total_travel_time_mins(route, start_time_minutes)

    # Print out route information
    print(f"===== Route {i + 1} =====")
    print(f"Route: {' → '.join(map(str, route))}")
    print(f"Distance: {round(distance, 2)}km")
    print(f"Duration: {format_duration(travel_time)}")

    # Add route to map
    fig.add_trace(go.Scattermapbox(
        name=f"Route {i + 1} {format_duration(travel_time)}",
        mode="markers+lines",
        hovertext=[intersections[x][0] for x in route],
        lon=[intersections[x][2] for x in route],
        lat=[intersections[x][1] for x in route],
        marker={"size": 10}, line={"width": 4}))


fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
