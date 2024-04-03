import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from FlightGrapher.sub_tool import get_extent_values, update_limits
from PlotLandingScatter.coordinate import ENU2LLH
from FlightGrapher.make_kml import getparachutepoint, post_kml
import os
from PIL import Image

class GraphPlotterParachute:
    def __init__(self, df_log, df_summary, filepath, launch_LLH):
        self.filepath = filepath

        self.time_array = np.array(df_log['time [sec]'])
        self.pos_ENU_log = np.array([[east, north, up] 
                                     for east, north, up 
                                     in zip(df_log['pos_east [m]'], 
                                            df_log['pos_north [m]'], 
                                            df_log['pos_up [m]'])])
        self.vel_ENU_log = np.array([[v_east, v_north, v_up] 
                                     for v_east, v_north, v_up 
                                     in zip(df_log['vel_east [m/s]'], 
                                            df_log['vel_north [m/s]'], 
                                            df_log['vel_up [m/s]'])])
        self.altitude_log = np.array(df_log['altitude [km]'])
        self.downrange_log = np.array(df_log['downrange [km]'])
        self.vel_air_ENU_log = np.array([[vair_east, vair_north, vair_up] 
                                         for vair_east, vair_north, vair_up 
                                         in zip(df_log['vel_air_east [m/s]'], 
                                                df_log['vel_air_north [m/s]'], 
                                                df_log['vel_air_up [m/s]'])])
        self.vel_air_abs_log = np.array(df_log['vel_air_abs [m/s]'])


        self.flighType = 'Parachute'

        time_actuate = df_summary['Time Engine Actuate [sec]'].to_numpy()
        time_apogee = df_summary['Time Apogee [sec]'].to_numpy()
        time_para1 = df_summary['Time 1st Parachute Open [sec]'].to_numpy()
        time_para2 = df_summary['Time 2nd Parachute Open [sec]'].to_numpy()
        self.time_end = self.time_array[-1]
        self.index_coast = np.argmax(time_actuate < self.time_array)
        self.index_apogee = np.argmax(time_apogee < self.time_array)
        self.index_para1 = np.argmax(time_para1 < self.time_array)
        self.index_para2 = np.argmax(time_para2 < self.time_array)

        # self.point = [logdata[len(self.time_array)-1, 1], logdata[len(self.time_array)-1, 2], logdata[len(self.time_array)-1, 3]]
        self.point = [self.pos_ENU_log[-1, 0], self.pos_ENU_log[-1, 1], self.pos_ENU_log[-1, 2]]
        self.Launch_LLH = launch_LLH

    def plot_graph(self, land_point):
        flightType = '_01_parachute'

        # img_logo = 'Quabla_logo.png'
        img_logo = Image.open('Quabla_logo.png')
        aspect_logo = img_logo.height / img_logo.width

        plt.close('all')

        fig, ax = plt.subplots()
        ax.set_title('Position ENU')
        ax.plot(self.time_array[:self.index_coast], self.pos_ENU_log[:self.index_coast,0], color='#FF4B00', linestyle='-', label='East')
        ax.plot(self.time_array[:self.index_coast], self.pos_ENU_log[:self.index_coast,1], color='#005AFF', linestyle='-', label='North')
        ax.plot(self.time_array[:self.index_coast], self.pos_ENU_log[:self.index_coast,2], color='#03AF7A', linestyle='-', label='Up')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.pos_ENU_log[self.index_coast:self.index_para1,0], color='#FF4B00', linestyle='--')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.pos_ENU_log[self.index_coast:self.index_para1,1], color='#005AFF', linestyle='--')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.pos_ENU_log[self.index_coast:self.index_para1,2], color='#03AF7A', linestyle='--')
        if self.index_para2 > self.index_para1:
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.pos_ENU_log[self.index_para1:self.index_para2,0], color='#FF4B00', linestyle=':')
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.pos_ENU_log[self.index_para1:self.index_para2,1], color='#005AFF', linestyle=':')
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.pos_ENU_log[self.index_para1:self.index_para2,2], color='#03AF7A', linestyle=':')
            ax.plot(self.time_array[self.index_para2:], self.pos_ENU_log[self.index_para2:,0], color='#FF4B00', linestyle='-.')
            ax.plot(self.time_array[self.index_para2:], self.pos_ENU_log[self.index_para2:,1], color='#005AFF', linestyle='-.')
            ax.plot(self.time_array[self.index_para2:], self.pos_ENU_log[self.index_para2:,2], color='#03AF7A', linestyle='-.')
        else:
            ax.plot(self.time_array[self.index_para1:], self.pos_ENU_log[self.index_para1:,0], color='#FF4B00', linestyle=':')
            ax.plot(self.time_array[self.index_para1:], self.pos_ENU_log[self.index_para1:,1], color='#005AFF', linestyle=':')
            ax.plot(self.time_array[self.index_para1:], self.pos_ENU_log[self.index_para1:,2], color='#03AF7A', linestyle=':')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Position [m]')
        ax.set_xlim(xmin=0.0, xmax=self.time_end)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.imshow(img_logo, extent=(get_extent_values(fig, ax, aspect_logo)), alpha=0.5)
        ax.set_aspect('auto')
        ax.grid()
        ax.legend()
        fig.savefig(self.filepath + os.sep + flightType + os.sep + 'PositionENU.png')

        fig1 = plt.figure('Flightlog' + flightType)
        origin = np.array([0.0, 0.0, 0.0])
        # ax = fig1.gca(projection = '3d')
        ax = fig1.add_subplot(projection='3d')
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_zlabel('Up [m]')
        ax.set_title('Trajectory')
        ax.plot(self.pos_ENU_log[:self.index_coast,0],self.pos_ENU_log[:self.index_coast,1],self.pos_ENU_log[:self.index_coast,2], label='Powered', color='#FF4B00')
        ax.plot(self.pos_ENU_log[self.index_coast:self.index_para1,0], self.pos_ENU_log[self.index_coast:self.index_para1,1], self.pos_ENU_log[self.index_coast:self.index_para1,2], label='Coasting', color='#005AFF')
        if self.index_para2 > self.index_para1:
            ax.plot(self.pos_ENU_log[self.index_para1:self.index_para2,0], self.pos_ENU_log[self.index_para1:self.index_para2,1], self.pos_ENU_log[self.index_para1:self.index_para2,2], label='Descent (Drogue chute)', color='#03AF7A')
            ax.plot(self.pos_ENU_log[self.index_para2:,0], self.pos_ENU_log[self.index_para2:,1], self.pos_ENU_log[self.index_para2:,2], label='Descent (Main chute)', color='#4DC4FF')
        else:
            ax.plot(self.pos_ENU_log[self.index_para1:,0], self.pos_ENU_log[self.index_para1:,1], self.pos_ENU_log[self.index_para1:,2], label='Parachute Falling', color='#03AF7A')
        ax.scatter(origin[0], origin[1], origin[2], marker='o', label='Launch Point', color='r')
        ax.scatter(self.pos_ENU_log[-1,0],self.pos_ENU_log[-1,1],self.pos_ENU_log[-1,2],label='Landing Point', s=30, marker='*',color='y')
        ax.legend(bbox_to_anchor=(.65, 1.05), loc='upper left')
        # ax.set_aspect('equal')
        ax.set_zlim(bottom=0.0)
        fig1.savefig(self.filepath + os.sep + flightType + os.sep + 'Flightlog.png')

        fig2, trajectory = plt.subplots()
        trajectory.set_title('Trajectory')
        trajectory.plot(self.downrange_log[:self.index_coast], self.altitude_log[:self.index_coast], color='#FF4B00', linestyle='-')
        trajectory.plot(self.downrange_log[self.index_coast:self.index_para1], self.altitude_log[self.index_coast:self.index_para1], color='#FF4B00', linestyle='--')
        trajectory.text(x=self.downrange_log[self.index_coast], y=self.altitude_log[self.index_coast], s='\n+\nEngine cut-off', fontsize='large', horizontalalignment='center', verticalalignment='center')
        if self.index_para2 > self.index_para1:
            trajectory.plot(self.downrange_log[self.index_para1:self.index_para2], self.altitude_log[self.index_para1:self.index_para2], color='#FF4B00', linestyle=':')
            trajectory.plot(self.downrange_log[self.index_para2:], self.altitude_log[self.index_para2:], color='#FF4B00', linestyle='-.')
            trajectory.text(x=self.downrange_log[self.index_para1], y=self.altitude_log[self.index_para1], s='\n+\nDrogue Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
            trajectory.text(x=self.downrange_log[self.index_para2], y=self.altitude_log[self.index_para2], s='\n+\nMain Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
        else:
            trajectory.plot(self.downrange_log[self.index_para1:], self.altitude_log[self.index_para1:], color='#FF4B00', linestyle=':')
            trajectory.text(x=self.downrange_log[self.index_para1], y=self.altitude_log[self.index_para1], s='\n+\nMain Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
        trajectory.set_xlabel('Downrange [km]')
        trajectory.set_ylabel('Altitude [km]')
        trajectory.set_ylim(ymin = 0.0)
        xmin, xmax, ymin, ymax = update_limits(trajectory.get_xlim(), trajectory.get_ylim(), fig2.get_figheight() / fig2.get_figwidth())
        trajectory.set_xlim(xmin=xmin, xmax=xmax)
        trajectory.set_ylim(ymin=ymin, ymax=ymax)
        trajectory.imshow(img_logo, extent=(get_extent_values(fig2, trajectory, aspect_logo)), alpha=0.5)
        trajectory.grid()
        trajectory.set_aspect('equal')
        fig2.savefig(self.filepath + os.sep + flightType + os.sep + 'Trajectory.png')

        fig2, trajectory = plt.subplots()
        trajectory.set_title('Downrange')
        trajectory.plot(self.pos_ENU_log[:self.index_coast, 0] / 1000.0 , self.pos_ENU_log[:self.index_coast, 1] / 1000.0, color='#FF4B00', linestyle='-')
        trajectory.plot(self.pos_ENU_log[self.index_coast:self.index_para1, 0] / 1000.0, self.pos_ENU_log[self.index_coast:self.index_para1, 1] / 1000.0, color='#FF4B00', linestyle='--')
        trajectory.text(x=self.pos_ENU_log[self.index_coast, 0] / 1000., y=self.pos_ENU_log[self.index_coast, 1] / 1000., s='\n+\nEngine cut-off', fontsize='large', horizontalalignment='center', verticalalignment='center')
        if self.index_para2 > self.index_para1:
            trajectory.plot(self.pos_ENU_log[self.index_para1:self.index_para2, 0] / 1000.0 , self.pos_ENU_log[self.index_para1:self.index_para2, 1] / 1000.0, color='#FF4B00', linestyle=':')
            trajectory.plot(self.pos_ENU_log[self.index_para2:, 0] / 1000.0 , self.pos_ENU_log[self.index_para2:, 1] / 1000.0, color='#FF4B00', linestyle='-.')
            trajectory.text(x=self.pos_ENU_log[self.index_para1, 0] / 1000., y=self.pos_ENU_log[self.index_para1, 1] / 1000., s='\n+\nDrogue Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
            trajectory.text(x=self.pos_ENU_log[self.index_para2, 0] / 1000., y=self.pos_ENU_log[self.index_para2, 1] / 1000., s='\n+\nMain Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
        else:
            trajectory.plot(self.pos_ENU_log[self.index_para1:, 0] / 1000.0 , self.pos_ENU_log[self.index_para1:, 1] / 1000.0, color='#FF4B00', linestyle=':')
            trajectory.text(x=self.pos_ENU_log[self.index_para1, 0] / 1000., y=self.pos_ENU_log[self.index_para1, 1] / 1000., s='\n+\nMain Chute Open', fontsize='large', horizontalalignment='center', verticalalignment='center')
        trajectory.text(x=self.pos_ENU_log[0, 0] / 1000., y=self.pos_ENU_log[0, 1] / 1000., s='\n+\nLaunch Point', fontsize='large', horizontalalignment='center', verticalalignment='center')
        trajectory.text(x=self.pos_ENU_log[-1, 0] / 1000., y=self.pos_ENU_log[-1, 1] / 1000., s='\n+\nLanding Point', fontsize='large', horizontalalignment='center', verticalalignment='center')
        xmin, xmax, ymin, ymax = update_limits(trajectory.get_xlim(), trajectory.get_ylim(), fig2.get_figheight() / fig2.get_figwidth())
        trajectory.set_xlim(xmin=xmin, xmax=xmax)
        trajectory.set_ylim(ymin=ymin, ymax=ymax)
        trajectory.imshow(img_logo, extent=(get_extent_values(fig2, trajectory, aspect_logo)), alpha=0.5)
        trajectory.set_xlabel('East [km]')
        trajectory.set_ylabel('North [km]')
        trajectory.grid()
        trajectory.set_aspect('equal')
        fig2.savefig(self.filepath + os.sep + flightType +  os.sep + 'Downrange.png')

        fig, ax = plt.subplots()
        ax.set_title('Velocity ENU')
        ax.plot(self.time_array[:self.index_coast], self.vel_ENU_log[:self.index_coast, 0], color='#FF4B00', linestyle='-', label = 'Roll')
        ax.plot(self.time_array[:self.index_coast], self.vel_ENU_log[:self.index_coast, 1], color='#005AFF', linestyle='-', label = 'Pitch')
        ax.plot(self.time_array[:self.index_coast], self.vel_ENU_log[:self.index_coast, 2], color='#03AF7A', linestyle='-', label = 'Yaw')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.vel_ENU_log[self.index_coast:self.index_para1, 0], color='#FF4B00', linestyle='--')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.vel_ENU_log[self.index_coast:self.index_para1, 1], color='#005AFF', linestyle='--')
        ax.plot(self.time_array[self.index_coast:self.index_para1], self.vel_ENU_log[self.index_coast:self.index_para1, 2], color='#03AF7A', linestyle='--')
        if self.index_para2 > self.index_para1:
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.vel_ENU_log[self.index_para1:self.index_para2, 0], color='#FF4B00', linestyle=':')
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.vel_ENU_log[self.index_para1:self.index_para2, 1], color='#005AFF', linestyle=':')
            ax.plot(self.time_array[self.index_para1:self.index_para2], self.vel_ENU_log[self.index_para1:self.index_para2, 2], color='#03AF7A', linestyle=':')
            ax.plot(self.time_array[self.index_para2:], self.vel_ENU_log[self.index_para2:, 0], color='#FF4B00', linestyle='-.')
            ax.plot(self.time_array[self.index_para2:], self.vel_ENU_log[self.index_para2:, 1], color='#005AFF', linestyle='-.')
            ax.plot(self.time_array[self.index_para2:], self.vel_ENU_log[self.index_para2:, 2], color='#03AF7A', linestyle='-.')
        else:
            ax.plot(self.time_array[self.index_para1:], self.vel_ENU_log[self.index_para1:, 0], color='#FF4B00', linestyle=':')
            ax.plot(self.time_array[self.index_para1:], self.vel_ENU_log[self.index_para1:, 1], color='#005AFF', linestyle=':')
            ax.plot(self.time_array[self.index_para1:], self.vel_ENU_log[self.index_para1:, 2], color='#03AF7A', linestyle=':')

        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Velocity [m/s]')
        ax.set_xlim(xmin=0.0, xmax=self.time_end)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.imshow(img_logo, extent=(get_extent_values(fig, ax, aspect_logo)), alpha=0.5)
        ax.set_aspect('auto')
        ax.grid()
        ax.legend()
        fig.savefig(self.filepath + os.sep + flightType + os.sep + 'VelocityENU.png')

        vENU2LLH = np.vectorize(ENU2LLH, excluded=['launch_LLH'], signature="(1),(3)->(3)")
        log_LLH = vENU2LLH(self.Launch_LLH, self.pos_ENU_log)
        point_LLH = ENU2LLH(self.Launch_LLH, self.point)
        getparachutepoint(point_LLH)
        post_kml(log_LLH, self.filepath, '_02_soft')
        land_point.get_point_parachute(self.point)
        

