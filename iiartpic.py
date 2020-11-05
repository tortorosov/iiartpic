import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import time

#st.title("Example Project")

orig_url='https://drive.google.com/file/d/1NrMfNIJpBbF5_yHp9bVIXLsWehYaoiM2/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id


def get_data():
	url = requests.get(dwn_url).text
	csv_raw = StringIO(url)

	z_data = pd.read_csv(csv_raw)

	#st.dataframe(z_data)

	z = z_data.values
	sh_0, sh_1 = z.shape
	x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
	
	return x, y, z

x, y, z = get_data()
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Example Project', autosize=False,
				  width=800, height=800,
				  margin=dict(l=40, r=40, b=40, t=40))

my_fig = st.plotly_chart(fig)

# while True:
	# x, y, z = get_data()
	# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
	# fig.update_layout(title='Example Project', autosize=False,
					  # width=800, height=800,
					  # margin=dict(l=40, r=40, b=40, t=40))

	# my_fig.plotly_chart(fig)
	
	# time.sleep(1.1)
