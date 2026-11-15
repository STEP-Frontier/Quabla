'''
加太共同実験射場（和歌山）用の落下分散図・落下判定
2027年3月の加太共同実験に向けて作成（元ファイル名 27.3kada_land.py）

出典：加太共同実験射場 シミュ審査書補足資料 6訂版（2026/6/22）
  - 射点A（ハイブリッド）・保安範囲（表7）・海抜90m は input/launch_site.json の "kada"
  - 点火点（表4 東側, 表5 西側）はこのファイルに直書き
  - 「図2 の円で囲まれた点火点及び射点の半径10ｍに機体を落下しないこと」を判定に入れる
    （共通の LaunchSite は射点の半径50m だけを見て、点火点を見ないため）

地図画像 img/kada.png は真北が上・約0.986 m/px。磁気偏角で回さない（magnetic_dec = 0）。
'''
import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm

import PlotLandingScatter.coordinate as cd
from PlotLandingScatter.launch_site.launch_site import LaunchSite
from PlotLandingScatter.judge_inside.judge_inside_poly import JudgeInsidePoly
from PlotLandingScatter.judge_inside.judge_inside_circle import JudgeInsideCircle

# 落下禁止円の半径 [m]（補足資料 3.1）
RADIUS_NO_LANDING = 10.0
# 点火点（補足資料 表4・表5）。高さは射点と同じ海抜90m
FIRE_EAST_LLH = np.array([34.285101, 135.091224, 90.0])
FIRE_WEST_LLH = np.array([34.28549, 135.0900545, 90.0])


class KadaLand(LaunchSite):

    def __init__(self, elevation, exist_payload, result_dir, glp_list, launch_site_info, safety_exist):
        # 射点・保安範囲(polygon)・表示範囲・画像などは共通クラスで読む
        super().__init__(elevation, exist_payload, result_dir, glp_list, launch_site_info, safety_exist)
        self.magnetic_dec = 0.0

        # 落下禁止円の中心（射点A・東西の点火点）を射点基準の ENU [m] に
        self.no_landing_ENU = [
            np.array([0.0, 0.0, 0.0]),
            cd.LLH2ENU(self.launch_LLH, FIRE_EAST_LLH),
            cd.LLH2ENU(self.launch_LLH, FIRE_WEST_LLH),
        ]

    def plot_landing_scatter(self):
        flight_mode = [self.trajectory, self.parachute]
        color_cm = [cm.cool, cm.spring]
        if self.exist_payload:
            flight_mode.append(self.payload)
            color_cm.append(cm.autumn)

        for index, obj in enumerate(flight_mode):
            if self.safety_exist:
                obj.plot_polygon(self.safety_NED)
                for center in self.no_landing_ENU:
                    obj.plot_circle(center, RADIUS_NO_LANDING)
            obj.grid_on()
            obj.plot_scatter(self.img, self.xlim, self.ylim, color_cm[index], self.magnetic_dec)
            obj.save_fig()

    def _judge(self, point, jip, jic_list):
        # 保安範囲の内側 かつ どの落下禁止円にも入っていない
        return jip.judge_inside(point) and not any(jic.judge_inside(point) for jic in jic_list)

    def judge_inside(self):
        jip = JudgeInsidePoly(self.safety_NED)
        jic_list = [JudgeInsideCircle(center[0:2], RADIUS_NO_LANDING) for center in self.no_landing_ENU]

        judge_list = []
        for point_tra, point_par in zip(self.glp_tra.pos_ENU_array, self.glp_par.pos_ENU_array):
            judge_tra = [self._judge(p, jip, jic_list) for p in point_tra]
            judge_par = [self._judge(p, jip, jic_list) for p in point_par]
            judge_list.append([a and b for a, b in zip(judge_tra, judge_par)])

        if self.exist_payload:
            judge_rocket_list = judge_list
            judge_list = []
            for point_pay, judge_rocket in zip(self.glp_pay.pos_ENU_array, judge_rocket_list):
                judge_pay = [self._judge(p, jip, jic_list) for p in point_pay]
                judge_list.append([a and b for a, b in zip(judge_rocket, judge_pay)])

        angle_step = 360 / self.glp_tra.num_angle
        angle_array = np.arange(0.0, 360 + angle_step, angle_step)
        pd.DataFrame(judge_list, index=self.wind_array, columns=angle_array).to_csv(self.trajectory.result_dir + os.sep + 'judge.csv')
