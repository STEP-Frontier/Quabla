package quabla.simulator.rocket;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.function.Function;

import com.fasterxml.jackson.databind.JsonNode;

import quabla.QUABLA;
import quabla.output.OutputTxt;
import quabla.simulator.numerical_analysis.Interpolation;
import quabla.simulator.rocket.wind.AbstractWind;
import quabla.simulator.rocket.wind.Constant;
import quabla.simulator.rocket.wind.Original;
import quabla.simulator.rocket.wind.Power;

public class Rocket extends AbstractRocket {

	// TODO 離陸前，燃焼後の時の重量，慣性モーメントをcsvに出力
	// TODO: make clone() function

	public final Engine engine;
	public final AeroParameter aero;
	public final Atmosphere atm;
	public final AbstractWind wind;
	public final Payload payload;

	public final double L, 
	                    D, 
	                    S,
						upperLug,
						lowerLug;
	private final double mBef,
						 mAft,
						 mSt,
						 mDry;
	private final double lcgDry,
						 lcgSt,
						 lcgAft;
	public final double lcgBef;
	private final double IjPitchBef,
						 IjPitchAft,
						 IjRollBef,
						 IjRollAft,
						 IjPitchDry,
						 IjRollDry,
						 IjStRollUnit;
	public final double CdS1,
						CdS2;
	public final double timeParaLag;
	public final boolean para2Exist, para2Timer;
	public final double alt_para2, time_para2;
	public final double dt;
	public final boolean existTipOff;
	public final boolean existPayload;
	public final double lengthLauncherRail, 
						elevationLauncher, 
						azimuthLauncher, 
						magneticDec;
	
	public static int site;
	public static double[] point = {0, 0, 0};
						
	private Interpolation massAnaly;
	private final Interpolation 
								massDepAnaly,
								lcgAnaly,
								lcgPropAnaly,
								IjPropPitchAnaly,
								IjPitchAnaly,
								IjDotPitchAnaly;

	public Rocket(JsonNode spec) {

		engine = new Engine(spec.get("Engine"));
		aero   = new AeroParameter(spec.get("Aero"));
		atm    = new Atmosphere(spec.get("Atmosphere").get("Temperature at 0 m [℃]").asDouble());

		existPayload = spec.get("Payload").get("Payload Exist").asBoolean();
		if (existPayload) {
			payload = new Payload(spec.get("Payload"));
		} else {
			payload = null;
		}
		
		JsonNode structure  = spec.get("Structure");
		JsonNode parachute  = spec.get("Parachute");
		JsonNode launchCond = spec.get("Launch Condition");
		
		if(spec.get("Wind").get("Wind File Exist").asBoolean() && QUABLA.simulationModeCheck.equals("single")) {
			wind = new Original(
					spec.get("Wind").get("Wind File").asText(),
					launchCond.get("Input Magnetic Azimuth [deg]").asDouble());
		}else if(spec.get("Wind").get("Wind Model").asText().equals("law")){
			wind = new Power(
					spec.get("Wind").get("Wind Speed [m/s]").asDouble(),
					spec.get("Wind").get("Wind Azimuth [deg]").asDouble(),
					spec.get("Wind").get("Wind Reference Altitude [m]").asDouble(),
					spec.get("Wind").get("Wind Power Law Coefficient").asDouble(),
					launchCond.get("Input Magnetic Azimuth [deg]").asDouble());
		}else {
			wind  = new Constant(
					spec.get("Wind").get("Wind Speed [m/s]").asDouble(),
					spec.get("Wind").get("Wind Azimuth [deg]").asDouble(),
					launchCond.get("Input Magnetic Azimuth [deg]").asDouble());
		}

		// For Interpolation
		double[] timeArray = new double[100];
		for (int i = 0; i < timeArray.length; i++) {
			timeArray[i] = ((double)i / (double)timeArray.length) * engine.timeActuate;
		}

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Geometry
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		L = structure.get("Length [m]").asDouble();
		D = structure.get("Diameter [m]").asDouble();
		S = 0.25 * Math.PI * Math.pow(D, 2);
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Mass 
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		mDry = structure.get("Dry Mass [kg]").asDouble();
		mSt  = mDry - engine.mFuelBef;
		mBef = mDry + engine.mOxBef;
		mAft = mDry - (engine.mFuelBef - engine.mFuelAft);

		Function<Double, Double> calcMass = (time) -> { return mSt + engine.getMassFuel(time) + engine.getMassOx(time); };

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Center of Gravity (C.G.)
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		lcgDry = structure.get("Dry Length-C.G. from Nosetip [m]").asDouble();
		lcgSt  = (lcgDry * mDry - (L - engine.lcgFuel)  * engine.mFuelBef) /  mSt;
		lcgBef = (lcgDry * mDry + (L - engine.lcgOxBef) * engine.mOxBef)   / (mDry + engine.mOxBef);
		lcgAft = (lcgSt  * mSt  + (L - engine.lcgFuel)  * engine.mFuelAft) / (mSt  + engine.mFuelAft);
		
		Function<Double, Double> calcLcg     = (time) -> { return (lcgSt * mSt + (L - engine.lcgFuel) * engine.getMassFuel(time) + (L - engine.getLcgOx(time))* engine.getMassOx(time)) / (mSt + engine.getMassFuel(time) + engine.getMassOx(time)); };
		Function<Double, Double> calcLcgProp = (time) -> { return ((L - engine.lcgFuel) * engine.getMassFuel(time) + (L - engine.getLcgOx(time)) * engine.getMassOx(time)) / (engine.getMassFuel(time) + engine.getMassOx(time)); };
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Moment of Inertia (MOI)
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// @ Dry ----------------------------------------------------------------------------------------------
		IjPitchDry = structure.get("Dry Moment of Inertia  Pitch-Axis [kg*m^2]").asDouble();
		IjRollDry  = structure.get("Dry Moment of Inertia Roll-Axis [kg*m^2]").asDouble();
		
		// @ Before Burn --------------------------------------------------------------------------------------
		// Pitch, Yaw Axis
		IjPitchBef = IjPitchDry + mDry * Math.pow(lcgDry - lcgBef, 2)
		           + (engine.IjOxPitchBef + engine.mOxBef * Math.pow(lcgBef - (L - engine.lcgOxBef), 2)); // 燃焼前全機重心回り酸化剤Pitch慣性モーメント
		
		// Roll Axis
		IjStRollUnit = IjRollDry - engine.IjFuelRollBef;
		IjRollBef    = IjRollDry + engine.IjOxRollBef;
		
		// @ After Burnout ------------------------------------------------------------------------------------
		IjPitchAft = IjPitchDry + mDry * Math.pow(lcgDry - lcgAft, 2)
		           - ((engine.IjFuelPitchBef + engine.mFuelBef * Math.pow(lcgBef - (L - engine.lcgFuel), 2)))
		           +  (engine.IjFuelPitchAft + engine.mFuelAft * Math.pow(lcgAft - (L - engine.lcgFuel), 2));
		IjRollAft  = IjStRollUnit + engine.IjFuelRollAft;

		Function<Double, Double> calcIjPropPitch = (time) -> { return engine.getIjFuelPitch(time) + engine.getMassFuel(time) * Math.pow((L - engine.lcgFuel) - calcLcg.apply(time), 2) + engine.getIjOxPitch(time) + engine.getMassOx(time) * Math.pow((L - engine.getLcgOx(time)) - calcLcg.apply(time), 2); };
		Function<Double, Double> calcIjPitch     = (time) -> { return (IjPitchDry + mDry * Math.pow(lcgDry - calcLcg.apply(time), 2)) - ((engine.IjFuelPitchBef + engine.mFuelBef * Math.pow(lcgBef - (L - engine.lcgFuel), 2))) + calcIjPropPitch.apply(time);};
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Interpolation
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		double[] massLog        = new double[timeArray.length];
		double[] lcgLog         = new double[timeArray.length];
		double[] lcgPropLog     = new double[timeArray.length];
		double[] IjPropPitchLog = new double[timeArray.length];
		double[] IjPitchLog     = new double[timeArray.length];
		double[] IjDotPitchLog  = new double[timeArray.length];
		
		// make Log
		for (int i = 0; i < timeArray.length; i++) {
			massLog[i]        = calcMass.apply(timeArray[i]);
			lcgLog[i]         = calcLcg.apply(timeArray[i]);
			lcgPropLog[i]     = calcLcgProp.apply(timeArray[i]);
			IjPropPitchLog[i] = calcIjPropPitch.apply(timeArray[i]); 
			IjPitchLog[i]     = calcIjPitch.apply(timeArray[i]); 
		}
		for (int i = 1; i < timeArray.length - 1; i++) {
			// 2nd-order Central Differential Scheme
			IjDotPitchLog[i] = (IjPitchLog[i + 1] - IjPitchLog[i - 1]) / (timeArray[i + 1] - timeArray[i - 1]);
		}

		// set Initial and Final Value
		massLog[0] = mBef;
		massLog[massLog.length - 1] = mAft;
		lcgLog[0] = lcgBef;
		lcgLog[lcgLog.length - 1] = lcgAft;
		// lcgPropLog[0] = L - engine.lcgFuel;
		lcgPropLog[lcgPropLog.length - 1] = L - engine.lcgFuel;
		IjPropPitchLog[IjPropPitchLog.length - 1] = engine.IjFuelPitchAft + engine.mFuelAft * Math.pow(lcgAft - (L - engine.lcgFuel), 2);
		IjPitchLog[0] = IjPitchBef;
		IjPitchLog[IjPitchLog.length - 1] = IjPitchAft;
		IjDotPitchLog[0] = 0.0;
		IjDotPitchLog[IjDotPitchLog.length - 1] = 0.0;
		
		// Make Instance
		massAnaly        = new Interpolation(timeArray, massLog);
		lcgAnaly         = new Interpolation(timeArray, lcgLog);
		lcgPropAnaly     = new Interpolation(timeArray, lcgPropLog);
		IjPropPitchAnaly = new Interpolation(timeArray, IjPropPitchLog);
		IjPitchAnaly     = new Interpolation(timeArray, IjPitchLog);
		IjDotPitchAnaly  = new Interpolation(timeArray, IjDotPitchLog);
		
		if (existPayload) {
			double[] massDepLog = new double[timeArray.length];
			
			for (int i = 0; i < massDepLog.length; i++) {
				double massPayload = payload.getMass(timeArray[i]);
				massDepLog[i] = massLog[i] - massPayload;
			}

			massDepLog[massDepLog.length - 1] = mAft - payload.getMass(engine.timeActuate);
			massDepAnaly = new Interpolation(timeArray, massDepLog);

		} else {
			massDepAnaly = null;
		}
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Parachute
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		CdS1        = parachute.get("1st Parachute CdS [m2]").asDouble();
		timeParaLag = parachute.get("Parachute Opening Lag [sec]").asDouble();
		para2Exist  = parachute.get("2nd Parachute Exist").asBoolean();
		if (para2Exist) {
			CdS2       = parachute.get("2nd Parachute CdS [m2]").asDouble();
			alt_para2  = parachute.get("2nd Parachute Opening Altitude [m]").asDouble();
			time_para2 = parachute.get("2nd Timer [s]").asDouble();
			para2Timer = parachute.get("2nd Parachute Timer Mode").asBoolean();
		} else {
			CdS2       = 0.0;
			alt_para2  = 0.0;
			time_para2 = 0.0;
			para2Timer = false;
		}
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Solver
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		dt = spec.get("Solver").get("Time Step [sec]").asDouble();
		
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Launch Configuration
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		point[0] = launchCond.get("Launch lat").asDouble();
		point[1] = launchCond.get("Launch lon").asDouble();
		point[2] = launchCond.get("Launch height").asDouble();
		site     = launchCond.get("Site").asInt();
		lengthLauncherRail = launchCond.get("Launcher Rail Length [m]").asDouble();
		elevationLauncher  = launchCond.get("Launch Elevation [deg]").asDouble();
		azimuthLauncher    = launchCond.get("Launch Azimuth [deg]").asDouble();
		//magneticDec        = Math.abs(launchCond.get("Input Magnetic Azimuth [deg]").asDouble());
		magneticDec		= launchCond.get("Magnetic Declination [deg]").asDouble();
		existTipOff        = launchCond.get("Tip-Off Calculation Exist").asBoolean();
		if (existTipOff) {
			upperLug = structure.get("Upper Launch Lug [m]").asDouble();
			lowerLug = structure.get("Lower Launch Lug [m]").asDouble();
		} else {
			// 機体後端が抜けたときLaunch Clear
			upperLug = L - 0.001;
			lowerLug = L;
			// launch clearは重心がランチャを抜けたとき
			// upperLug = lcgBef;
			// lowerLug = lcgBef + 0.01;
		}
	}

	/**
	 * @return 質量時間変化率。負の値で返す
	 * */
	public double mdot(double t) {
		return - engine.getMdotProp(t);
	}

	@Override
	public double getMass(double t) {

		return massAnaly.linearInterp1column(t);
		
	}

	public double getLcg(double t) {
		
		return lcgAnaly.linearInterp1column(t);
		
	}

	public double getLcgProp(double t) {
		
		return lcgPropAnaly.linearInterp1column(t);

	}

	private double getIjPropPitch(double t) {
		
		return IjPropPitchAnaly.linearInterp1column(t);
		
	}

	private double getIjPropRoll(double t) {
		if(t < engine.timeActuate) {
			return engine.getIjFuelRoll(t) + engine.getIjOxRoll(t);
		}else {
			return engine.IjFuelRollAft;
		}
	}

	public double getIjPitch(double t) {
		
		return IjPitchAnaly.linearInterp1column(t);
		
	}	

	public double getIjRoll(double t) {
		if(t < engine.timeActuate) {
			return IjStRollUnit + getIjPropRoll(t);
		}else {
			return IjRollAft;
		}	
	}	

	public double getIjDotPitch(double t) {

		return IjDotPitchAnaly.linearInterp1column(t);

	}

	public void deployPayload() {
		massAnaly = massDepAnaly;
	}

	public double getCdS(double time, double altitude) {
		
		if (para2Exist) {
			if (para2Timer) {
				if(time >= time_para2) {
					return CdS1 + CdS2;
				}else {
					return CdS1;
				}
			}else {
				if (altitude <= alt_para2) {
					return CdS1 + CdS2;
				}else {
					return CdS1;
				}
			}
		}else {
			return CdS1;
		}
	}
	
	public void outputSpec(String resultDir, String simulationMode) {
		OutputTxt specTxt = null;
		
		try {
			specTxt = new OutputTxt(resultDir + "rocket_param.txt");
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
		try {
			
			specTxt.outputLine("------------------------- * Structure * ------------------------");
			specTxt.outputLine(format(" Length: ", 20) + String.format("%.3f [m]", L));
			specTxt.outputLine(format(" Diameter: ", 20) + String.format("%.3f [m]", D));
			specTxt.outputLine(format(" Area: ", 20) + String.format("%.3f [m^2]", S));
			if(existTipOff) {
				specTxt.outputLine(format(" Upper Launch Lug: ", 20) + String.format("%.3f [m]", upperLug));
				specTxt.outputLine(format(" Lower Launch Lug: ", 20) + String.format("%.3f [m]", lowerLug));
			}
			specTxt.outputLine("");
			
			specTxt.outputLine("--------------------------- * Mass * ---------------------------");
			specTxt.outputLine(format(" Dry mass: ", 20) + String.format("%.4f [kg]", mDry));
			specTxt.outputLine(format(" Structure mass: ", 20) + String.format("%.4f [kg]", mSt));
			specTxt.outputLine(format(" Propellant mass: ", 20) + String.format("%.4f [kg]", mBef - mSt));
			specTxt.outputLine(format(" Mass @ before burn: ", 20) + String.format("%.4f [kg]", mBef));
			specTxt.outputLine(format(" Mass @ after burn: ", 20) + String.format("%.4f [kg]", mAft));
			specTxt.outputLine("");
			
			specTxt.outputLine("----------------- * Center of Gravity (C.G.) * -----------------");
			specTxt.outputLine(format(" Dry Length-G.G. from Nosetip: ", 40) + String.format("%.4f [m]", lcgDry));
			specTxt.outputLine(format(" Structure Length-G.G. from Nosetip: ", 40) + String.format("%.4f [m]", lcgSt));
			specTxt.outputLine(format(" Length-G.G. @ before burn from Nosetip: ", 40) + String.format("%.4f [m]", lcgBef));
			specTxt.outputLine(format(" Length-G.G. @ after burn from Nosetip: ", 40) + String.format("%.4f [m]", lcgAft));
			specTxt.outputLine(format(" Propellant Length-G.G. @ before burn from Nosetip: ", 52) + String.format("%.4f [m]", getLcgProp(0.0)));
			specTxt.outputLine("");
			
			specTxt.outputLine("------------------ * Moment of Inertia (MOI) * -----------------");
			specTxt.outputLine(format(" Dry MOI Roll-Axis: ", 45) + String.format("%.4f [kg*m^2]", IjRollDry));
			specTxt.outputLine(format(" Dry MOI Pitch-Axis: ", 45) + String.format("%.4f [kg*m^2]", IjPitchDry));
			specTxt.outputLine(format(" MOI @ before burn Roll-Axis: ", 45) + String.format("%.4f [kg*m^2]", IjRollBef));
			specTxt.outputLine(format(" MOI @ before burn Pitch-Axis: ", 45) + String.format("%.4f [kg*m^2]", IjPitchBef));
			specTxt.outputLine(format(" MOI @ after burn Roll-Axis:", 45) + String.format("%.4f [kg*m^2]", IjRollAft));
			specTxt.outputLine(format(" MOI @ after burn Pitch-Axis: ", 45) + String.format("%.4f [kg*m^2]", IjPitchAft));
			specTxt.outputLine(format(" Propellant MOI @ before burn Roll-Axis: ", 45) + String.format("%.4f [kg*m^2]", getIjPropRoll(0.0)));
			specTxt.outputLine(format(" Propellant MOI @ before burn Pitch-Axis: ", 45) + String.format("%.4f [kg*m^2]", getIjPropPitch(0.0)));
			specTxt.outputLine("");
			
			specTxt.outputLine("----------------------- * Aero Parameter * ---------------------");
			if(!aero.getLcpFileExist()) {
				specTxt.outputLine(format(" Constant Length-C.P. from Nosetip: ", 40) + String.format("%.3f [m]", aero.Lcp(0.0)));
			}
			if(!aero.getCdFileExist()) {
				specTxt.outputLine(format(" Constant Cd: ", 40) + String.format("%.3f [-]", aero.Cd(0.0)));
			}
			if(!aero.getCNaFileExist()) {
				specTxt.outputLine(format(" Constant CNa: ", 40) + String.format("%.3f [1/rad]", aero.CNa(0.0)));
			}
			specTxt.outputLine(format(" Constant Clp: ", 40) + String.format("%.3f [1/rad]", aero.Clp));
			specTxt.outputLine(format(" Constant Cmq: ", 40) + String.format("%.3f [1/rad]", aero.Cmq));
			specTxt.outputLine(format(" Constant Cnr: ", 40) + String.format("%.3f [1/rad]", aero.Cnr));
			specTxt.outputLine("");
			
			specTxt.outputLine("------------------------- * Parachute * ------------------------");
			specTxt.outputLine(format(" 1st Parachute CdS: ", 35) + String.format("%.3f [m^2]", CdS1));
			if(para2Exist) {
				specTxt.outputLine(format(" 2nd Parachute CdS: ", 35) + String.format("%.3f [m^2]", CdS2));
				if(para2Timer) {
					specTxt.outputLine(format(" 2nd Timer: ", 35) + String.format("%.1f [s]", time_para2));
				}else {
					specTxt.outputLine(format(" 2nd Parachute Opening Altitude: ", 35) + String.format("%.3f [m]", alt_para2));
				}
			}
			specTxt.outputLine("");
			
			specTxt.outputLine("---------------------- * Launch Condition * --------------------");
			specTxt.outputLine(format(" Launcher Rail Length: ", 30) + String.format("%.1f [m]", lengthLauncherRail));
			specTxt.outputLine(format(" Launch Elevation: ", 30) + String.format("%.1f [deg]", elevationLauncher));
			specTxt.outputLine(format(" Launch Magnetic Azimuth: ", 30) + String.format("%.1f [deg]", azimuthLauncher));
			specTxt.outputLine(format(" Launch True Azimuth: ", 30) + String.format("%.1f [deg]", azimuthLauncher - magneticDec));
			specTxt.outputLine("");
			
			specTxt.outputLine("----------------------- * Wind Condition * ---------------------");
			switch (simulationMode) {
			
				case "single":
					if(wind instanceof Original) {
						specTxt.outputLine(format(" Wind Model: ", 20) + "Original");
						specTxt.outputLine(format(" Wind File: ", 20) + wind.getFilePath());
					}else if(wind instanceof Power) {
						specTxt.outputLine(format(" Wind Model: ", 30) + "Power Law");
						specTxt.outputLine(format(" Wind Speed: ", 30) + String.format("%.1f [m/s]", wind.getRefWindSpeed()));
						specTxt.outputLine(format(" Wind Azimuth: ", 30) + String.format("%.1f [deg]", wind.getRefWindAzimuth()));
						specTxt.outputLine(format(" Wind Power Law Coefficient: ", 30) + String.format("%.1f [-]", wind.getExponent()));
					}else if(wind instanceof Constant) {
						specTxt.outputLine(format(" Wind Model: ", 20) + "Constant");
						specTxt.outputLine(format(" Wind Speed: ", 20) + String.format("%.1f [m/s]", wind.getRefWindSpeed()));
						specTxt.outputLine(format(" Wind Azimuth: ", 20) + String.format("%.1f [deg]", wind.getRefWindAzimuth()));
					}
					break;

				case "multi":
					if(wind instanceof Power) {
						specTxt.outputLine(format(" Wind Model: ", 30) + "Power Law");
						specTxt.outputLine(format(" Wind Power Law Coefficient: ", 30) + String.format("%.1f [-]", wind.getExponent()));
					}else if(wind instanceof Constant) {
						specTxt.outputLine(format(" Wind Model: ", 20) + "Constant");
					}
					break;
			}
			specTxt.outputLine("");
			
		} catch (IOException e) {
			 throw new RuntimeException(e);
		}
		
		try {
			specTxt.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
    // 文字列の整形用の関数
	private static String format(String target, int length){
        int byteDiff = (getByteLength(target, Charset.forName("UTF-8"))-target.length())/2;
        return String.format("%-"+(length-byteDiff)+"s", target);
    }

    private static int getByteLength(String string, Charset charset) {
        return string.getBytes(charset).length;
    }

	public double getIjDotRoll(double t) {
		if(t < engine.timeBurnout) {
			if(t == 0.0) {
				return 0.0;
			}else {
				return (getIjPropRoll(t) - getIjPropRoll(t - dt)) / dt;
			}
		}else {
			return 0.0;
		}
	}

	public double[] getPoint() {
		return point;
	}

	public int getSite() {
		return site;
	}


}
