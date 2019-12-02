package quabla.simulator;

import java.io.IOException;

import quabla.output.OutputLandingScatter;
import quabla.parameter.InputParam;

public class MultiSolver {
	InputParam spec;
	double speed_min,speed_step;
	int speed_num,angle_num;

	public MultiSolver(InputParam spec) {
		this.spec = spec;

		this.spec.Wind_file_exsit = false;//分散の時は強制的にべき法則

		this.speed_min = spec.speed_min;
		this.speed_step = spec.speed_step;
		this.speed_num = spec.speed_num;
		this.angle_num = spec.angle_num;

	}

	public void solve_multi() {
		double[][] wind_map_trajectory = new double[2*speed_num][angle_num+1];
		double[][] wind_map_parachute = new double[2*speed_num][angle_num+1];
		//row:風速, column:風向

		double[] speed_array = new double[speed_num];
		double[] azimuth_array = new double[angle_num + 1];
		for(int i = 0; i < speed_num; i++) {
			speed_array[i] = speed_min + i * speed_step;
		}
		for(int i = 0; i <= angle_num; i++) {
			azimuth_array[i] = 360.0 * i / angle_num;
		}

		int i = 0;
		for(double speed: speed_array) {
			spec.wind_speed = speed;
			int j = 0;
			for(double azimuth: azimuth_array) {
				spec.wind_azimuth = azimuth;

				//solverのインスタンスの生成
				Solver single_solver = new Solver(spec,false);//Multi_solverでは各フライトでのlogは保存しない
				single_solver.solve_dynamics();

				//trajectory,parachuteでの落下地点を取得
				double[] pos_ENU_landing_trajectory = single_solver.pos_ENU_landing_trajectory;
				double[] pos_ENU_landing_parachute = single_solver.pos_ENU_landing_parachute;

				wind_map_trajectory[2*i][j] = pos_ENU_landing_trajectory[0];
				wind_map_trajectory[2*i+1][j] = pos_ENU_landing_trajectory[1];
				wind_map_parachute[2*i][j] = pos_ENU_landing_parachute[0];
				wind_map_parachute[2*i+1][j] = pos_ENU_landing_parachute[1];
				j++;
			}
			i++;
		}

		OutputLandingScatter trajectory = new OutputLandingScatter();
		try {
			trajectory.output(spec.result_filepath + "trajectory"+spec.elevation_launcher+"[deg].csv",wind_map_trajectory, speed_array);
		} catch (IOException e) {
			e.printStackTrace();
		}

		OutputLandingScatter parachute = new OutputLandingScatter();
		try {
			parachute.output(spec.result_filepath + "parachute"+spec.elevation_launcher+"[deg].csv",wind_map_parachute, speed_array);
		} catch (IOException e) {
			e.printStackTrace();
		}


	}

}
