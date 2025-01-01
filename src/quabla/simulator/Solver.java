package quabla.simulator;

import java.io.IOException;
import java.util.function.Function;

import quabla.output.OutputCsv;
import quabla.output.OutputFlightlogParachute;
import quabla.output.OutputFlightlogTrajectory;
import quabla.output.OutputTxt;
import quabla.simulator.dynamics.DynamicsOnLauncher;
import quabla.simulator.dynamics.DynamicsParachute;
import quabla.simulator.dynamics.DynamicsTipOff;
import quabla.simulator.dynamics.DynamicsTrajectory;
import quabla.simulator.logger.LoggerVariable;
import quabla.simulator.logger.LoggerVariableParachute;
import quabla.simulator.logger.event_value.EventValueSingle;
import quabla.simulator.numerical_analysis.ODEsolver.AbstractODEsolver;
import quabla.simulator.numerical_analysis.ODEsolver.PredictorCorrector;
import quabla.simulator.numerical_analysis.ODEsolver.RK4;
import quabla.simulator.rocket.Rocket;
import quabla.simulator.variable.AbstractVariable;
import quabla.simulator.variable.VariableParachute;
import quabla.simulator.variable.VariableTrajectory;

/**
 * Solver command functions in single condition, which are needed simulating dynamics and storing.
 * */
public class Solver {

	private final String resultDir;

	private EventValueSingle eventValue;

	private LoggerVariable trajectoryLog;
	private LoggerVariableParachute parachuteLog;
	private LoggerVariableParachute payloadLog;

	private boolean is2ndPara, exitPayload;
	private double timeBurnAct;

	public Solver(String resultDir) {
		this.resultDir = resultDir;
	}

	public void solveDynamics(Rocket rocket) {
		int index = 0;
		int indexLaunchClear, indexLandingTrajectory, indexLandingParachute, index2ndPara = 0, index1stPara, indexLandingPayload;
		double h = rocket.dt;
		boolean isTipOff = false;
		is2ndPara = rocket.para2Exist;
		exitPayload = rocket.existPayload;
		timeBurnAct = rocket.engine.timeActuate;

		VariableTrajectory variableTrajectory = new VariableTrajectory(rocket);

		// Dynamics
		DynamicsTrajectory dynTrajectory = new DynamicsTrajectory(rocket);
		DynamicsOnLauncher dynOnLauncher = new DynamicsOnLauncher(rocket);
		Function<AbstractVariable, double[]> dynamics;

		// ODE solver
		AbstractODEsolver ODEsolver = new RK4(h); // 最初は4次ルンゲクッタで計算
		PredictorCorrector predCorr = new PredictorCorrector(h);

		// Predictor-Corrector法で用いる過去の微分値を保存するための配列
		double[][] deltaArray = new double[3][13];

		// どの飛行状態に遷移したかを判定
		FlightEventJudgement eventJudgement = new FlightEventJudgement(rocket) ;

		trajectoryLog = new LoggerVariable(rocket);
		parachuteLog = new LoggerVariableParachute(rocket);

		// log Initial Variable
		trajectoryLog.log(variableTrajectory, 0.0);

		// **************************************************************************************** /
		//    on Launcher                                                                           /
		// **************************************************************************************** /
		dynamics = dynOnLauncher::calculateDynamics;
		for(;;) {
			index ++;

	        // Change ODE solver
			if(index == 4) {
				predCorr.setDelta(deltaArray[2], deltaArray[1], deltaArray[0]);
				ODEsolver = predCorr;	// ODE解法の変更
			}

			//  solve ODE
			double[] delta = ODEsolver.compute(variableTrajectory, dynamics);	// 
			variableTrajectory.update(ODEsolver.getTimeStep(), delta);			// 基礎変数の更新
			trajectoryLog.log(variableTrajectory, ODEsolver.getTimeStep());		// ログ記録
			
			// 最初の3回はRunge-Kuttaで解く
			if(index <= 3) {
				deltaArray[index - 1] = delta;
			}

			// ------------------------------------------------------------------------------------ /
			//    Tip-Off                                                                             /
			// ------------------------------------------------------------------------------------ /
			if(rocket.existTipOff && eventJudgement.judgeTipOff(variableTrajectory) && !isTipOff) {// 1回のみ実行
				// indexTipOff = index;
				DynamicsTipOff dynTipOff = new DynamicsTipOff(rocket);
				dynamics = dynTipOff::calculateDynamics;
				isTipOff = true;
			}

			if(eventJudgement.judgeLaunchClear(variableTrajectory)) {
				indexLaunchClear = index;
				break;
			}
		}


		// **************************************************************************************** /
		//    Trajectory                                                                            /
		// **************************************************************************************** /
		boolean flag = true;
		dynamics = dynTrajectory::calculateDynamics;
		for(;;) {
			index ++;

			double[] delta = ODEsolver.compute(variableTrajectory, dynamics);
			variableTrajectory.update(ODEsolver.getTimeStep(), delta);
			trajectoryLog.log(variableTrajectory, ODEsolver.getTimeStep());

			// if (flag && eventJudgement.judgeApogee(variableTrajectory)) {
			if (flag && variableTrajectory.getTime() >= rocket.engine.timeActuate * 1.2) {
				predCorr.effectiveATS();
				flag = false;
			}

			if(eventJudgement.judgeLanding(variableTrajectory)) {
				indexLandingTrajectory = index;
				break;
			}
		}
		trajectoryLog.makeArray(indexLaunchClear);

		// store Event Value
		eventValue = new EventValueSingle(trajectoryLog, rocket);
		eventValue.setIndexLaunchClear(indexLaunchClear);
		eventValue.setIndexLandingTrajectory(indexLandingTrajectory);

		index1stPara = eventValue.getIndex1stPara();
		eventValue.setIndex1stPara(index1stPara);
		parachuteLog.copy(index1stPara, trajectoryLog);
		trajectoryLog.dumpArrayList();

		
		// Change ODE solver
		h = 0.05;
		ODEsolver = new RK4(h);
		predCorr = new PredictorCorrector(h);
		predCorr.effectiveATS();
		
		// Parachute 用の変数にパラシュート放出時の変数を渡す
		VariableParachute variablePara = new VariableParachute();
		variablePara.set(trajectoryLog, index1stPara);
		
		// **************************************************************************************** /
		//    Payload                                                                               /
		// **************************************************************************************** /
		if (exitPayload) {
			rocket.deployPayload();
			VariableParachute varPayload = new VariableParachute();
			varPayload.set(trajectoryLog, index1stPara);
			DynamicsParachute dynPayload = new DynamicsParachute(rocket.payload, rocket.atm, rocket.wind);
			dynamics = dynPayload::calculateDynamics;
			payloadLog = new LoggerVariableParachute(rocket.payload, rocket.atm, rocket.wind);
			deltaArray = new double[3][4];
			
			int indexPaylaod = 0;
			index = index1stPara;
			
			for( ; ; ) {
				index ++;
				indexPaylaod ++;
				
				// Change ODE solver
				if(indexPaylaod == 4) {
					predCorr.setDelta(deltaArray[2], deltaArray[1], deltaArray[0]);
					ODEsolver = predCorr;
				}
				
				// solve ODE
				double[] delta = ODEsolver.compute(varPayload, dynamics);
				varPayload.update(ODEsolver.getTimeStep(), delta);
				payloadLog.log(varPayload, ODEsolver.getTimeStep());
				
				if(indexPaylaod <= 3) {
					deltaArray[indexPaylaod - 1] = delta;
				}
				
				if(eventJudgement.judgeLanding(varPayload)) {
					indexLandingPayload = indexPaylaod - 1;
					break;
				}
			}
			payloadLog.makeArray();
			eventValue.setIndexLandingPayload(indexLandingPayload);
		}
		
		// indexの更新
		index = index1stPara;
		int index_para = 0;
		DynamicsParachute dynParachute = new DynamicsParachute(rocket);
		deltaArray = new double[3][4];
		ODEsolver = new RK4(h);
		
		// **************************************************************************************** /
		//    Parachute                                                                             /
		// **************************************************************************************** /
		dynamics = dynParachute::calculateDynamics;
		for( ; ; ) {
			index ++;
			index_para ++;
			
			// Change ODE solver
			if(index_para == 4) {
				predCorr.setDelta(deltaArray[2], deltaArray[1], deltaArray[0]);
				ODEsolver = predCorr;
			}
			
			// solve ODE
			double[] delta = ODEsolver.compute(variablePara, dynamics);
			variablePara.update(ODEsolver.getTimeStep(), delta);
			parachuteLog.log(variablePara, ODEsolver.getTimeStep());
			
			if(index_para <= 3) {
				deltaArray[index_para - 1] = delta;
			}

			if(eventJudgement.judge2ndPara(variablePara)) {
				index2ndPara = index;
			}

			if(eventJudgement.judgeLanding(variablePara)) {
				indexLandingParachute = index;
				break;
			}
		}
		parachuteLog.makeArray();

		eventValue.setIndexLandingParachute(indexLandingParachute);

		eventValue.setIndex2ndPara(index2ndPara);
		if (exitPayload) {
			eventValue.calculate(trajectoryLog, parachuteLog, payloadLog);
		} else {
			eventValue.calculate(trajectoryLog, parachuteLog);
		}

	}

	public EventValueSingle getEventValueSingle() {
		return eventValue;
	}

	public void dumpFlightLog() {
		trajectoryLog = null;
		parachuteLog = null;
		if (exitPayload) {
			payloadLog = null;
		}
	}

	public void makeResult() {

		OutputFlightlogTrajectory oft = new OutputFlightlogTrajectory(trajectoryLog);
		OutputFlightlogParachute ofp = new OutputFlightlogParachute(parachuteLog);
		oft.runOutputLine(resultDir + "flightlog_trajectory.csv");
		ofp.runOutputLine(resultDir + "flightlog_parachute.csv");
		if (exitPayload) {
			new OutputFlightlogParachute(payloadLog).runOutputLine(resultDir + "flightlog_payload.csv");
		}
	}

	/**
	 * outputResultTxt writes txt file about event values.
	 * */
	public void outputResultTxt() {
		OutputTxt resultTxt = null;

		try {
			resultTxt = new OutputTxt(resultDir + "result.txt");
		}catch(IOException e) {
			throw new RuntimeException(e);
		}

		try {

			resultTxt.outputLine("----------------------- * Static Margin * ---------------------- ");
			
			resultTxt.outputLine(String.format(" Minimum Static Margin, Fst : %.3f [-]", eventValue.getFstMin()));
			resultTxt.outputLine(String.format(" Max Static Margin, Fst     : %.3f [-]", eventValue.getFstMax()));
			
			resultTxt.outputLine("\n----------------------- * Launch Clear * ----------------------- ");
			
			resultTxt.outputLine(String.format(" Launch Clear Time         : %.3f [sec]", eventValue.getTimeLaunchClear()));
			resultTxt.outputLine(String.format(" Launch Clear Velocity     : %.3f [m/s]", eventValue.getVelLaunchClear()));
			resultTxt.outputLine(String.format(" Launch Clear Acceleration : %.3f G", eventValue.getAccLaunchClear() / 9.80665));
			
			resultTxt.outputLine("\n-------------------------- * Apogee * -------------------------- ");
			
			resultTxt.outputLine(String.format(" Apogee Time      : %.3f [sec]", eventValue.getTimeApogee()));
			resultTxt.outputLine(String.format(" Apogee Altitude  : %.3f [km]", eventValue.getAltApogee()));
			resultTxt.outputLine(String.format(" Apogee Downrange : %.3f [km]", eventValue.getDownrangeApogee()));
			resultTxt.outputLine(String.format(" Apogee Air Speed : %.3f [m/s]", eventValue.getVelAirApogee()));
			
			resultTxt.outputLine("\n----------------------- * Max Air Speed * ---------------------- ");
			
			resultTxt.outputLine(String.format(" Max Air Speed Time     : %.3f [sec]", eventValue.getTimeMaxVelAir()));
			resultTxt.outputLine(String.format(" Max Air Speed          : %.3f [m/s]", eventValue.getVelAirMax()));
			resultTxt.outputLine(String.format(" Max Air Speed Altitude : %.3f [km]", eventValue.getAltitudeMaxVelAir()));
			
			resultTxt.outputLine("\n--------------------------- * Max-Q * -------------------------- ");
			
			resultTxt.outputLine(String.format(" Max-Q Time              : %.3f [sec]", eventValue.getTimeMaxQ()));
			resultTxt.outputLine(String.format(" Max-Q Dynamics Pressure : %.3f [kPa]", eventValue.getDynamicsPressureMax()));
			resultTxt.outputLine(String.format(" Max-Q Altitude          : %.3f [km]", eventValue.getAltitudeMaxQ()));
			
			resultTxt.outputLine("\n------------------------- * Max Mach * ------------------------- ");
			
			resultTxt.outputLine(String.format(" Max Mach Time     : %.3f [sec]", eventValue.getTimeMaxMach()));
			resultTxt.outputLine(String.format(" Max Mach          : %.3f [-]", eventValue.getMachMax()));
			resultTxt.outputLine(String.format(" Max Mach Altitude : %.3f [km]", eventValue.getAltitudeMaxMach()));
			
			resultTxt.outputLine("\n--------------------- * Max Normal Force * --------------------- ");
			
			resultTxt.outputLine(String.format(" Max Normal Force Time     : %.3f [sec]", eventValue.getTimeNormalMax()));
			resultTxt.outputLine(String.format(" Max Normal Force          : %.3f [N]", eventValue.getNormalMax()));
			resultTxt.outputLine(String.format(" Max Normal Force Altitude : %.3f [km]", eventValue.getAltitudeNormalMax()));
			
			resultTxt.outputLine("\n---------------------- * Max Side Force * ---------------------- ");
			
			resultTxt.outputLine(String.format(" Max Side Force Time     : %.3f [sec]", eventValue.getTimeSideMach()));
			resultTxt.outputLine(String.format(" Max Side Force          : %.3f [N]", eventValue.getSideMax()));
			resultTxt.outputLine(String.format(" Max Side Force Altitude : %.3f [km]", eventValue.getAltitudeSideMax()));
			
			resultTxt.outputLine("\n---------------------- * Parachute Open * ---------------------- ");
			
			resultTxt.outputLine(String.format(" 1st Parachute Open Time : %.3f [sec]", eventValue.getTime1stPara()));
			if (is2ndPara) {
				resultTxt.outputLine(String.format(" 2nd Parachute Open Time : %.3f [sec]", eventValue.getTime2ndPara()));
			}
			
			double[] posNEDTrajectory = eventValue.getPosNEDlandingTrajectory();
			double[] posNEDParachute = eventValue.getPosNEDlandingParachute();
			
			resultTxt.outputLine("\n-------------------- * Landing Trajectory * -------------------- ");
			resultTxt.outputLine(String.format(" Landing Trajectory Time      : %.3f [sec]", eventValue.getTimeLandingTrajectory()));
			resultTxt.outputLine(String.format(" Landing Trajectory Downrange : %.3f [km]", eventValue.getDownrangeLandingTrajectory()));
			resultTxt.outputLine(String.format(" Landing Trajectory Point NED : [ %.3f , %.3f ]", posNEDTrajectory[0], posNEDTrajectory[1]));
			// resultTxt.outputLine(String.format(" Landing Trajectory LLH       : [ %.8f , %.8f ]", ENUtoLLH.ENU2LLH(pointTrajectory)[0], ENUtoLLH.ENU2LLH(pointTrajectory)[1]));
			
			
			resultTxt.outputLine("\n-------------------- * Landing Parachute * -------------------- ");
			resultTxt.outputLine(String.format(" Landing Parachute Time             : %.3f [sec]", eventValue.getTimeLandingParachute()));
			resultTxt.outputLine(String.format(" Landing Parachute Velocity Descent : %.3f [m/s]", eventValue.getVelDescentLandingParachute()));
			resultTxt.outputLine(String.format(" Landing Parachute Downrange        : %.3f [km]", eventValue.getDownrangeLandingParachute()));
			resultTxt.outputLine(String.format(" Landing Parachute Point NED        : [ %.3f , %.3f ]", posNEDParachute[0], posNEDParachute[1]));
			// resultTxt.outputLine(String.format(" Landing Parachute LLH              : [ %.8f , %.8f ]", ENUtoLLH.ENU2LLH(pointParachute)[0], ENUtoLLH.ENU2LLH(pointParachute)[1]));

		}catch(IOException e) {
			throw new RuntimeException(e) ;
		}

		try {
			resultTxt.close();
		}catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	public void outputResultSummary(){

		OutputCsv summary = null;

		String[] name = {
			"Time Launch Clear [sec]",
			"Time Engine Actuate [sec]",
			"Time Apogee [sec]",
			"Time 1st Parachute Open [sec]",
			"Time 2nd Parachute Open [sec]"
		};

		try {
			summary = new OutputCsv(resultDir + "summary.csv", name);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		try {
			summary.outputFirstLine();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		double[] results = {
			eventValue.getTimeLaunchClear(),
			timeBurnAct,
			eventValue.getTimeApogee(),
			eventValue.getTime1stPara(),
			eventValue.getTime2ndPara()
		};


		try {
			summary.outputLine(results);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		try {
			summary.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	public void outputLandPoint() {
		
		OutputCsv landPointCsv;

		String[] name;

		if (exitPayload) {

			name = new String[3];
			name[0] = "Trajectory";
			name[1] = "Parachute";
			name[2] = "Payload";
			
		} else {
			
			name = new String[2];
			name[0] = "Trajectory";
			name[1] = "Parachute";

		}

		try {
			landPointCsv = new OutputCsv(resultDir + "land_point_NED.csv", name);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		try {
			landPointCsv.outputFirstLine();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		
		for (int i = 0; i < 3; i++) {

			double[] results = new double[name.length];
			
			if (i < 2) {
				results[0] = eventValue.getPosNEDlandingTrajectory()[i];
				results[1] = eventValue.getPosNEDlandingParachute()[i];
				if (exitPayload) {
					results[2] = eventValue.getPosNEDlandingPayload()[i];
				}
			}
			
			try {
				landPointCsv.outputLine(results);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		try {
			landPointCsv.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}

}