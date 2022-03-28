import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns


class Animation:

    def __init__(self, embs, figsize=(12, 10), node_sizes=100, node2color: list = None, color_palette="rocket_r", padding=0.1, fps=6,):

        self._figsize = figsize
        self._anim = None
        self._fps = fps

        # Data properties
        self._embs = embs
        self._frames_num = embs.shape[0]
        self._nodes_num = embs.shape[1]
        self._dim = embs.shape[2]

        # Visual properties
        sns.set_theme(style="ticks")
        node2color = [0]*self._nodes_num if node2color is None else node2color
        self._color_num = 1 if node2color is None else len(set(node2color))
        self._palette = sns.color_palette(color_palette, self._color_num)
        self._node_colors = [self._palette.as_hex()[node2color[node]] for node in range(self._nodes_num)]
        self._node_sizes = [node_sizes]*self._nodes_num if type(node_sizes) is int else node_sizes
        self._linewidths = 1
        self._edgecolors = 'k'
        self._padding = padding



    def _render(self, fig, ax, repeat=False):
        global sc
        # scatter = ax.axes(xlim=(0, 2), ylim=(-2, 2))

        #line, = ax.plot([], [], lw=2)
        # ax.grid()
        # z = np.random.rand(100, 2)
        # sc = ax.scatter(z[:, 0]-20, z[:, 1]+2, c='r')

        def _init_func():
            global sc

            z = np.random.randn(100, 2)
            sc = ax.scatter(
                [0]*self._nodes_num, [0]*self._nodes_num,
                s=self._node_sizes, c=self._node_colors,
                linewidths=self._linewidths, edgecolors=self._edgecolors
            )

            xy_min = self._embs.min(axis=0, keepdims=False).min(axis=0, keepdims=False)
            xy_max = self._embs.max(axis=0, keepdims=False).max(axis=0, keepdims=False)
            xlen_padding = ( xy_max[0] - xy_min[0] ) * self._padding
            ylen_padding = ( xy_max[1] - xy_min[1] ) * self._padding
            ax.set_xlim([xy_min[0]-xlen_padding, xy_max[0]+xlen_padding])
            ax.set_ylim([xy_min[1]-ylen_padding, xy_max[1]+ylen_padding])

        def _func(f):
            global sc

            sc.set_offsets(np.c_[self._embs[f, :, 0], self._embs[f, :, 1]])

        anim = animation.FuncAnimation(
            fig=fig, init_func=_init_func, func=_func, frames=self._frames_num, interval=100, repeat=repeat
        )

        return anim

    def save(self, filepath, format="mp4"):

        fig, ax = plt.subplots(figsize=self._figsize)


        self._anim = self._render(fig, ax)

        if format == "mp4":
            writer = animation.FFMpegWriter(fps=self._fps)

        elif format == "gif":
            writer = animation.PillowWriter(fps=self._fps)

        else:
            raise ValueError("Invalid format!")

        self._anim.save(filepath, writer)


# embs = np.random.randn(100, 10, 2)
# anim = Animation(embs)
# anim.save("./deneme.mp4")