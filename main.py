import numpy as np
import pandas as pd
import plotly.express as px


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


# Read data
sites = pd.read_csv('data/scats-sites.csv')
data = pd.read_csv('data/scats-data.csv')

print('sites', sites.size)
print('data ', data.size)

# Filter interseciton sites
sites = sites[sites['Site Type'].eq('INT')]
data = data[data['SCATS Number'].isin(sites['Site Number'])]

# Filter out data at (0,0)
# data = data[(data['NB_LATITUDE'] != 0) & (data['NB_LONGITUDE'] != 0)]

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

print(data['id'].head(200))

print('sites filtered', sites.size)
print('data filtered ', data.size)
print('unique sites  ', len(np.unique(data['SCATS Number'])))

# Offset positions to align with map
data['NB_LATITUDE'] = data['NB_LATITUDE'].add(0.0015)
data['NB_LONGITUDE'] = data['NB_LONGITUDE'].add(0.0013)

# Show sites on map
fig = px.scatter_mapbox(data, lat='NB_LATITUDE', lon='NB_LONGITUDE', hover_name='Location', hover_data=['SCATS Number', 'id'],
                        color_discrete_sequence=['fuchsia'], zoom=8)
fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
fig.show()
