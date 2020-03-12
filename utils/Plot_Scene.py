import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time


class Plot_Scene:

    def _label2xy_mic(self,label):
        y = 0.5+np.mod(label-1,5)*0.5
        x = 3.50-((label-1)/5)
        return [x,y]

    def __init__(self, src_azi_all, mic_pos, room=None):
        room_xy = [7.1,5.1]
        mic_xy = self._label2xy_mic(mic_pos)
        src_dist = 1.5

        color_wall = 'black'
        color_mic = 'red'
        color_src = 'blue'
        color_est = 'gray'

        fig,ax = plt.subplots(1,1)
        # wall of room
        ax.plot([0,room_xy[0]],[0,0],color=color_wall)
        ax.plot([0,room_xy[0]],[room_xy[1],room_xy[1]],color=color_wall)
        ax.plot([0,0],[0,room_xy[1]],color=color_wall)
        ax.plot([room_xy[0],room_xy[0]],[0,room_xy[1]],color=color_wall)
        # mic
        ax.plot(mic_xy[0],mic_xy[1],marker='o',color=color_mic)
        # source (groundtruth)
        alpha = 1
        for azi in src_azi_all:
            azi_rad = (azi*5.-90.)/180.0*np.pi
            src_x = mic_xy[0] + (src_dist+0.2)*np.sin(azi_rad)
            src_y = mic_xy[1] + (src_dist+0.2)*np.cos(azi_rad)
            ax.scatter(src_x,src_y,marker='v',color=color_src,alpha=alpha)

        if room is not None:
            ax.text(0.8, 0.8, room)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        self.fig = fig
        self.ax = ax

        self.room_xy = room_xy
        self.mic_xy = mic_xy
        self.src_dist = src_dist
        self.color_wall = color_wall
        self.color_mic = color_mic
        self.color_src = color_src
        self.color_est = color_est


    def plot_est(self,src_azi_all):
        mic_xy = self.mic_xy
        src_dist = self.src_dist
        ax = self.ax
        alpha = 0.2
        for azi in src_azi_all:
            azi_rad = (azi*5.-90.)/180.0*np.pi
            src_x = mic_xy[0] + src_dist*np.sin(azi_rad)
            src_y = mic_xy[1] + src_dist*np.cos(azi_rad)
            ax.scatter(src_x,src_y,marker='v',color=self.color_est,alpha=alpha)


    def savefig(self,fpath):
        self.fig.savefig(fpath)
        plt.close(self.fig)


if __name__ == '__main__':

    scence = Plot_Scene([8,28],5)
    azi_est = np.concatenate((np.random.randn(20)+8,np.random.randn(20)+28))
    scence.plot_est(azi_est)
    scence.savefig('plot_scence_example.png')
