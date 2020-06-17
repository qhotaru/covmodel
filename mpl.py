#!/usr/bin/python
#
# mpl.py
#
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time
import numpy as np
import argparse
#
#
#
def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--anim", action='store_true', help="animation")

    
    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    parser.add_argument("-v", "--valiation1", action='store_true', help="variation")
    parser.add_argument("-d", "--dq", action='store_true', help="variation")
    parser.add_argument("--pq", action='store_true', help="show pq graph")
    parser.add_argument("--qp", action='store_true', help="show Dq graph")

    # JAG JAPAN
    parser.add_argument("--jag", action='store_true', help="show jag")

    # JHU
    parser.add_argument("-j", "--jhu", action='store_true', help="show jhu")
    parser.add_argument("--datalist", action='store_true', help="show data list")
    parser.add_argument("--jhulast", action='store_true', help="show jhu last")
    parser.add_argument("--jhulast2", action='store_true', help="show jhu last 2")
    parser.add_argument("--jhulast3", action='store_true', help="show jhu last 3")
    parser.add_argument("--korea", action='store_true', help="show korea")
    parser.add_argument("--plot", help="plot type, line, bar")

    # JHU option
    parser.add_argument("--nation", nargs='+', help="nation option")
    parser.add_argument("--pop", action='store_true', help="option pop")
    parser.add_argument("--new", action='store_true', help="option new")
    parser.add_argument("--rolling", action='store_true', help="option rolling")
    parser.add_argument("--offset", type=int, help="option offset")
    parser.add_argument("--reff", action='store_true', help="option Reff")

    # SWS
    parser.add_argument("--sws", action='store_true', help="show sws bar")
    parser.add_argument("--pref", action='store_true', help="show sws bar")
    parser.add_argument("--option", help="show sws option")
    parser.add_argument("--domestic", action='store_true', help="show domestic data")

    # TOKYO
    parser.add_argument("--tokyo", action='store_true', help="show tokyo")
    parser.add_argument("--death", action='store_true', help="show tokyo death")

    # TOKYO option
    parser.add_argument("--save", action='store_true', help="save tokyo csv")

    
    # graph
    parser.add_argument("-x", "--xrange", type=int, help="x axis range")
    parser.add_argument("-l", "--linear", action='store_true', help="show with linear")
    parser.add_argument("--diff", action='store_true', help="show diff option")

    parser.add_argument("--i", action='store_true', help="show I")
    parser.add_argument("--q", action='store_true', help="show Q")
    parser.add_argument("--n", action='store_true', help="show N")

    parser.add_argument("--kcdcage", action='store_true', help="show korea age")

    args = parser.parse_args()
    return args
#
#
#
class view:
    def __init__(self, args):
        self.args = args
        self.styles = ['r-','g-','b-','c-','y-','m-']
        self.niter = 1000
        pass

    def domain(self):
        if args.anim:
            self.show_sin_animation()
        else:
            self.show_sin()
        pass

    def doplots(self, ax, style):
        return ax.plot(self.x, self.y, style, animated=True)[0]
    
    def show_sin(self):
        self.x = np.arange(0, 2*np.pi, 0.1)
        self.y = np.sin(self.x)
        fig, axes = plt.subplots(nrows=6)
        fig.show()
        fig.canvas.draw()
        lines = [self.doplots(ax,style) for ax, style in zip(axes, self.styles)]
        bgs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]

        plt.ion()

        tstart = time.time()
        for i in range(1,self.niter):
            items = enumerate(zip(lines,axes, bgs), start=1)
            for j, (line, ax, bg) in items:
                fig.canvas.restore_region(bg)
                ax.draw_artist(line)
                fig.canvas.blit(ax.bbox)

        tend = time.time()
        print('FPS {}'.format( self.niter / (tend-tstart) ))
        
    pass

    def animate(self,i):
        for j, line in enumerate(self.lines, start=1):
            line.set_ydata( np.sin(j*self.x + i/10.0) )
            return self.lines

    def show_sin_animation(self):
        self.x = np.arange(0, 2*np.pi, 0.1)
        self.y = np.sin(self.x)
        fig, axes = plt.subplots(nrows=6)
        # fig.show()

        self.lines = [self.doplots(ax,style) for ax, style in zip(axes, self.styles)]
        tstart = time.time()
        # ani = animation.FuncAnimation( fig, self.animate, range(1,self.niter), interval=1, blit=True, frames=100)
        ani = animation.FuncAnimation( fig, self.animate, interval=1, blit=True, frames=self.niter)

        plt.show()
        tend = time.time()
        print('FPS {}'.format( self.niter / (tend-tstart) ))
        
    pass

if __name__ == '__main__':
    args = doparse()
    v = view(args)
    v.domain()

    
