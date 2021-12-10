import os
import torch
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt


class Animation:

    def __init__(self, embs, time_list, group_labels, colors=None, color_name="Colors", sizes=None, title="",
                 padding=0.1, figure_path=None):

        self.__df = None
        self.__embs = embs
        self.__time_list = time_list
        self.__group_labels = group_labels
        self.__colors = colors if colors is not None else self.__group_labels
        self.__color_name = color_name
        self.__sizes = sizes if sizes is not None else [10] * self.__embs.shape[0]
        self.__title = title
        self.__padding = padding
        self.__figure_path = figure_path
        self.__run()

    def __get_dataframe(self):

        df = pd.DataFrame({"x-axis": self.__embs[:, 0],
                           "y-axis": self.__embs[:, 1],
                           "time": self.__time_list,
                           "group": self.__group_labels,
                           self.__color_name: self.__colors,
                           "size": self.__sizes,
                           })

        return df

    def __run(self):

        df = self.__get_dataframe()

        range_x = [df["x-axis"].min() - self.__padding, df["x-axis"].max() + self.__padding]
        range_y = [df["y-axis"].min() - self.__padding, df["y-axis"].max() + self.__padding]

        plt.figure()
        fig = px.scatter(df, x="x-axis", y="y-axis", animation_frame="time", size=df["size"], animation_group="group",
                         color=self.__color_name, color_discrete_sequence=px.colors.qualitative.Pastel,
                         size_max=10, title=self.__title, range_x=range_x, range_y=range_y)

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1
        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_traces(marker=dict(size=20, line=dict(width=1, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))

        if self.__figure_path is None:
            fig.show()
        else:
            fig.write_html(self.__figure_path, auto_open=False)



# import os
# import torch
# import plotly.express as px
# import pandas as pd
# import matplotlib.pyplot as plt
#
# class Animation:
#
#     def __init__(self, df, title, anim_frame, anim_group, anim_size, anim_color, anim_hover_name):
#
#         self.__df = df
#         self.__anim_frame = anim_frame
#         self.__anim_group = anim_group
#         self.__anim_size = anim_size
#         self.__anim_color = anim_color
#         self.__anim_hover_name = anim_hover_name
#         self.__title = title
#         self.__colors = ['r', 'b', 'k', 'm']
#
#         self.__run()
#
#     def __run(self):
#         #print(self.__df["node_id"])
#         node_colors = [self.__colors[node_id] for node_id in self.__df["node_id"]]
#         import numpy as np
#         pad = 0.1
#         range_x = [ self.__df["x"].min() - pad, self.__df["x"].max() + pad ]
#         range_y = [ self.__df["y"].min() - pad, self.__df["y"].max() + pad ]
#         plt.figure()
#         fig = px.scatter(self.__df, x="x", y="y", animation_frame=self.__anim_frame, animation_group=self.__anim_group,
#                    size=[10]*self.__df.shape[0], color=node_colors, hover_name=self.__anim_hover_name,
#                    size_max=10, range_x=range_x, range_y=range_y)
#         #log_x=True,
#         plt.title(self.__title)
#         fig.show()
#
#