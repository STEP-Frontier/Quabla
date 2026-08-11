import numpy as np
from PlotLandingScatter.plot_graph import PlotGraph
import PlotLandingScatter.coordinate as cd
import matplotlib.cm as cm
from PlotLandingScatter.judge_inside.judge_inside_circle import JudgeInsideCircle
from PlotLandingScatter.judge_inside.judge_inside_border import JudgeInsideBorder
from PlotLandingScatter.judge_inside.judge_inside_poly import JudgeInsidePoly
import pandas as pd
import os

class LaunchSite():

    def __init__(self, elevation, exist_payload, result_dir, glp_list, launch_site_info, safety_exist):
        self.img = launch_site_info.img
        self.exist_payload = exist_payload
        self.safety_exist = safety_exist
        self.type_safety = launch_site_info.type_safety
        self.magnetic_dec = 0.0
        
        self.glp_tra = glp_list[0]
        self.glp_par = glp_list[1]
        self.wind_array = self.glp_tra.get_wind_name()

        self.trajectory = PlotGraph("Trajectory " + elevation + "[deg]", result_dir, self.glp_tra)
        self.parachute  = PlotGraph("Parachute " + elevation + "[deg]", result_dir, self.glp_par)
        if exist_payload:
            self.glp_pay = glp_list[2]
            self.payload = PlotGraph("Payload " + elevation + "[deg]", result_dir, self.glp_pay)

        # 射点
        self.launch_LLH = launch_site_info.launch_LLH

        # 射点
        self.launch_LLH = launch_site_info.launch_LLH
        # Polygon
        if self.type_safety == 'polygon':
            # 落下保安域
            self.safety_LLH = launch_site_info.safety_area_LLH
            point_LLH_array = self.safety_LLH
            point_NED_array = []
            for point_LLH in point_LLH_array:
                point_NED = cd.LLH2ENU(self.launch_LLH, point_LLH)
                point_NED_array.append(point_NED)
            self.safety_NED = np.array(point_NED_array)

            self.radius            = 0.0
            self.safety_line1      = 0.0
            self.safety_line2      = 0.0
            self.center_circle_LLH = 0.0

        # Circle
        elif self.type_safety == 'circle':
            # 保安円半径
            self.radius = launch_site_info.radius
            # 端点
            self.safety_line1 = launch_site_info.edge1_LLH
            self.safety_line2 = launch_site_info.edge2_LLH
            # 保安円中心
            self.center_circle_LLH = launch_site_info.center_circle_LLH
            # 境界線端点
            edge1_LLH = launch_site_info.edge1_LLH
            edge2_LLH = launch_site_info.edge2_LLH

            point_LLH_array = [self.center_circle_LLH, edge1_LLH, edge2_LLH]
            point_NED_array = []
            for point_LLH in point_LLH_array:
                point_NED = cd.LLH2ENU(self.launch_LLH, point_LLH)
                point_NED_array.append(point_NED)
            self.center_circle_ENU, self.edge1_ENU, self.edge2_ENU = point_NED_array

            self.safety_LLH = 0.0
        
        self.xlim = launch_site_info.x_offset + np.array(launch_site_info.xlim)
        self.ylim = launch_site_info.y_offset + np.array(launch_site_info.ylim)

    def plot_landing_scatter(self):
        flight_mode = [self.trajectory, self.parachute]
        color_cm = [cm.cool, cm.spring]
        if self.exist_payload:
            flight_mode.append(self.payload)
            color_cm.append(cm.autumn)

        for index, obj in enumerate(flight_mode):
            obj.ax.plot(0.0, 0.0, marker='o', markersize=3, color='r')
            if self.safety_exist == True:
                # Polygon
                if self.type_safety == 'polygon':
                    obj.plot_polygon(self.safety_NED)
                    obj.plot_circle(np.array([0.0, 0.0, 0.0]), 0.001)
                    #ここで保安円を描画する
                # Circle
                elif self.type_safety == 'circle':
                    obj.plot_circle(self.center_circle_ENU, self.radius)
                    obj.plot_line(self.edge1_ENU, self.edge2_ENU)
            obj.grid_on()
            obj.plot_scatter(self.img, self.xlim, self.ylim, color_cm[index], self.magnetic_dec)
            obj.save_fig()

    def judge_inside(self):
        # Polygon
        if self.type_safety == 'polygon':
            jip = JudgeInsidePoly(self.safety_NED)
            jic = JudgeInsideCircle(np.array([0.0, 0.0]), 0.001)
        #ここで保安円に入っているかどうかの判定を行う
        # Circle
        elif self.type_safety == 'circle':
            jic = JudgeInsideCircle(self.center_circle_ENU, self.radius)
            jib = JudgeInsideBorder(self.edge1_ENU, self.edge2_ENU, self.center_circle_ENU)


        judge_list = []
        for point_tra, point_par in zip(self.glp_tra.pos_ENU_array, self.glp_par.pos_ENU_array):
            if self.type_safety == 'polygon':
                judge_tra = [all([jip.judge_inside(point), not jic.judge_inside(point)]) for point in point_tra]
                judge_par = [all([jip.judge_inside(point), not jic.judge_inside(point)]) for point in point_par]
            elif self.type_safety == 'circle':
                judge_tra = [jic.judge_inside(point) and jib.judge_inside(point) for point in point_tra]
                judge_par = [jic.judge_inside(point) and jib.judge_inside(point) for point in point_par]
            judge_list.append([judge_traje and judge_para for judge_traje, judge_para in zip(judge_tra, judge_par)])

        if self.exist_payload:
            judge_list_rocket = judge_list
            judge_list = []
            for point, judge_rocket in zip(self.glp_pay.pos_ENU_array, judge_list_rocket):
                if self.type_safety == 'polygon':
                    judge = [all([jip.judge_inside(p), not jic.judge_inside(p)]) for p in point]
                elif self.type_safety == 'circle':
                    judge = [jic.judge_inside(p) and jib.judge_inside(p) for p in point]
                
                judge_list.append([rocket and payload for rocket, payload in zip(judge_rocket, judge)])

        angle_step = 360 / self.glp_tra.num_angle
        angle_array = np.arange(0.0, 360 + angle_step, angle_step)
        pd.DataFrame(judge_list, index=self.wind_array, columns=angle_array).to_csv(self.trajectory.result_dir + os.sep + 'judge.csv')

    def output_kml(self):
        self.trajectory.output_kml(self.launch_LLH, self.magnetic_dec, cm.cool  , self.radius, self.safety_line1,self.safety_line2, self.center_circle_LLH, self.safety_LLH, self.safety_exist)
        self.parachute.output_kml( self.launch_LLH, self.magnetic_dec, cm.spring, self.radius, self.safety_line1,self.safety_line2, self.center_circle_LLH, self.safety_LLH, self.safety_exist)
        if self.exist_payload:
            self.payload.output_kml( self.launch_LLH, self.magnetic_dec, cm.autumn, self.radius, self.safety_line1,self.safety_line2, self.center_circle_LLH, self.safety_LLH, self.safety_exist)

def magnetic_declination(lat, lon):
    delta_lat = lat - 37.
    delta_lon = lon - 138.
    return (7.0 + 57.201 / 60.0) + (18.750 / 60.0) * delta_lat - (6.761 / 60.0) * delta_lon - (0.059 / 60.0) * delta_lat**2 - (0.014 / 60.0) * delta_lat * delta_lon - (0.579 / 60.0) * delta_lon**2 #[deg]
