import streamlit as st

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO

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
	#x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
	
	return sh_0, sh_1, z

sh_0, sh_1, z = get_data()

#st.write(sh_0)
#st.write(sh_1)
#st.write(z)

#x = ['ПОЛЬЗОВАТЕЛЬ', 'АДМИНИСТРИРОВАНИЕ', 'ПРОГРАММИРОВАНИЕ', 'УПРАВЛЕНИЕ']
x = ['Пользователь', 'Администрирование', 'Программирование', 'Управление']

fig = go.Figure()
for i in range(sh_0):
	fig.add_trace(go.Bar(x = x, y = np.delete(z[i,], 0), name=z[i,0]))

#fig.update_layout(barmode='stack')
#fig.show()
#fig.update_traces(showlegend=True, showscale=False)
fig.update_layout(title='Мои навыки в %', autosize=False,
				  width=800, height=600,
				  margin=dict(l=40, r=40, b=40, t=40)
				  )

fig = st.plotly_chart(fig, use_container_width=True)

link = 'Мои работы: https://sites.google.com/view/iiartpic'
st.markdown(link, unsafe_allow_html=True)

################################

#111111
#import numpy as np
#import pandas as pd
#import plotly.graph_objects as go
#import requests
#from io import StringIO
#import time

#st.title("Example Project")

#orig_url='https://drive.google.com/file/d/1NrMfNIJpBbF5_yHp9bVIXLsWehYaoiM2/view?usp=sharing'

#file_id = orig_url.split('/')[-2]
#dwn_url='https://drive.google.com/uc?export=download&id=' + file_id


# def get_data():
	# url = requests.get(dwn_url).text
	# csv_raw = StringIO(url)

	# z_data = pd.read_csv(csv_raw)

	# #st.dataframe(z_data)

	# z = z_data.values
	# sh_0, sh_1 = z.shape
	# x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
	
	# return x, y, z

# x, y, z = get_data()

# mask0 = np.logical_and(y>=.0, y<=.4)
# mask1 = np.logical_and(y>.4, y<.7)
# mask2 = np.logical_and(y>=.7, y<=1)

# #st.write(mask1)

# #mask3 = y>.5
# #st.write(z[_mask1].reshape(2, -1))
# #st.write(z)
# #st.write(z[_mask0].reshape(-1, 2))
# #st.write(z[mask2])
# #st.write(_y1)

# #fig = go.Figure(data=[go.Surface(x=_x1, y=_y1, z=z, colorscale="Greens")])
# fig = go.Figure(go.Surface(x=x, y=y, z=z, colorscale="Greens", name='nnnnnn'))
# #fig = go.Figure(data=[go.Surface(z=z[mask0], x=x, y=y)])
# #fig.add_trace(z=z, x=x, y=y, name='sssssss')])

# # fig = go.Figure(go.Isosurface(
                      # # x=x.ravel(), y=y.ravel(), z=z.ravel(),
                      # # isomin=1.9, isomax=1.9,
                      # # colorscale="BuGn",
                      # # name='isosurface'))

# #
# fig.add_trace(go.Surface(x=x, y=y, z=z,
                      # colorscale="Blues",
                      # name='cones'))
# fig.add_trace(go.Surface(x=x, y=y, z=z,
                      # colorscale="Reds",
                      # name='mmmmmm'))



# fig.update_traces(showlegend=True, showscale=False)
# fig.update_layout(title='Example Project', autosize=False,
				  # width=800, height=800,
				  # margin=dict(l=40, r=40, b=40, t=40)
				  # )

# my_fig = st.plotly_chart(fig)
#111111

#222222
# z1 = np.array([
    # [8.83,8.89,8.81,8.87,8.9,8.87],
    # [8.89,8.94,8.85,8.94,8.96,8.92],
    # [8.84,8.9,8.82,8.92,8.93,8.91],
    # [8.79,8.85,8.79,8.9,8.94,8.92],
    # [8.79,8.88,8.81,8.9,8.95,8.92],
    # [8.8,8.82,8.78,8.91,8.94,8.92],
    # [8.75,8.78,8.77,8.91,8.95,8.92],
    # [8.8,8.8,8.77,8.91,8.95,8.94],
    # [8.74,8.81,8.76,8.93,8.98,8.99],
    # [8.89,8.99,8.92,9.1,9.13,9.11],
    # [8.97,8.97,8.91,9.09,9.11,9.11],
    # [9.04,9.08,9.05,9.25,9.28,9.27],
    # [9,9.01,9,9.2,9.23,9.2],
    # [8.99,8.99,8.98,9.18,9.2,9.19],
    # [8.93,8.97,8.97,9.18,9.2,9.18]
# ])

# z2 = z1 + 1
# z3 = z1 - 1

# fig = go.Figure(data=[
    # go.Surface(z=z1),
    # go.Surface(z=z2, showscale=False, opacity=0.9),
    # go.Surface(z=z3, showscale=False, opacity=0.9)

# ])

# #fig.show()
# my_fig = st.plotly_chart(fig)

#222222



# while True:
	# x, y, z = get_data()
	# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
	# fig.update_layout(title='Example Project', autosize=False,
					  # width=800, height=800,
					  # margin=dict(l=40, r=40, b=40, t=40))

	# my_fig.plotly_chart(fig)
	
	# time.sleep(1.1)
