import pandas as pd
import plotly.express as px

# Read data
sites = pd.read_csv('data/scats-sites.csv')
data = pd.read_csv('data/scats-data.csv')

print('sites', sites.size)
print('data ', data.size)

# Filter interseciton sites
sites = sites[sites['Site Type'].eq('INT')]
data = data[data['SCATS Number'].isin(sites['Site Number'].to_numpy())]

# Filter out data at (0,0)
# data = data[(data['NB_LATITUDE'] != 0) & (data['NB_LONGITUDE'] != 0)]

print('sites filtered', sites.size)
print('data filtered ', data.size)

# Offset positions to align with map
data['NB_LATITUDE'] = data['NB_LATITUDE'].add(0.0015)
data['NB_LONGITUDE'] = data['NB_LONGITUDE'].add(0.0013)

# Show sites on map
fig = px.scatter_mapbox(data, lat='NB_LATITUDE', lon='NB_LONGITUDE', hover_name='Location', hover_data=['SCATS Number'],
                        color_discrete_sequence=['fuchsia'], zoom=8)
fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
fig.show()
