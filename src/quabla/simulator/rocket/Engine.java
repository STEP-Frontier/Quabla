package quabla.simulator.rocket;

import java.util.function.Function;

import com.fasterxml.jackson.databind.JsonNode;

import quabla.simulator.GetCsv;
import quabla.simulator.numerical_analysis.Interpolation;

public class Engine {

	public final double
	Ae, // Area of outlet
	de; // diameter of throat
	private double
	dFuelInBef,						// グレイン内径（燃焼前）
	dFuelInAft,						// グレイン内径（燃焼後）
	dFuelOut,						// グレイン外径
	lFuel;							// グレイン全長
	private double
	lTank,							// タンク全長
	dTank,							// タンク直径
	distanceTank;					// タンク口金位置（タンクネジ穴と機体後端までの距離）
	public final  double
	mOxBef,							// 酸化剤質量（燃焼前）
	mFuelBef,						// グレイン質量（燃焼前）
	mFuelAft;						// グレイン質量（燃焼後）
	private double 
	mDotFuel;						// グレイン質量流量
	/* エンジン関連の重心は全部エンジン後端(ノズル側)からの長さ */
	public final double
	lcgFuel,						// グレイン重心
	lcgOxBef,						// 酸化剤重心（燃焼前）
	lcgOxAft;						// 酸化剤重心（燃焼後）
	/* 燃料重心回りの慣性モーメント */
	public final  double
	IjFuelPitchBef,					// 燃料ピッチ軸慣性モーメント（燃焼前）
	IjFuelRollBef,					// 燃料ロール軸慣性モーメント（燃焼前）
	IjFuelPitchAft,					// 燃料ピッチ軸慣性モーメント（燃焼後）
	IjFuelRollAft;					// 燃料ロール軸慣性モーメント（燃焼後）
	public final double
	IjOxPitchBef,					// 酸化剤ピッチ軸慣性モーメント（燃焼前）
	IjOxRollBef;					// 酸化剤ロール軸慣性モーメント（燃焼前）
	private Interpolation
	thrustAnaly,					// 推力補間用インスタンス
	mDotPropAnaly,					// 推進剤質量流量補間用インスタンス
	mOxAnaly,						// 酸化剤質量補間用インスタンス
	lcgOxAnaly;						// 酸化剤重心補間用インスタンス
	public final double
	timeBurnout,					// 燃焼時間
	timeActuate;					// 作動時間
	private double IspAve;			// 平均比推力
	private double totalImpulse;	// トータルインパルス
	private double thrustAve;		// 平均推力
	private double mDotPropAve;		// 推進剤平均質量流量

	public Engine(JsonNode engine) {

		timeBurnout = engine.get("Burn Time [sec]").asDouble();

		mOxBef	 = (engine.get("Tank Volume [cc]").asDouble() * Math.pow(10, -6)) * engine.get("Oxidizer Density [kg/m^3]").asDouble();
		mFuelBef = engine.get("Fuel Mass Before [kg]").asDouble();
		mFuelAft = engine.get("Fuel Mass After [kg]").asDouble();
		mDotFuel = (mFuelBef - mFuelAft) / timeBurnout;

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Thrust
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		/*  Configuration of Thrust History csv file:
		 *  1st Column : Time [s]
		 *  2nd Column : Thrust@Ground [N] **/
		double[][] thrustData = GetCsv.get2ColumnArray(engine.get("Thrust Curve").asText());
		double[] timeArray    = new double[thrustData.length];
		double[] thrustArray  = new double[thrustData.length];
		double[] mDotPropLog  = new double[thrustData.length];
		double[] mDotFuelLog  = new double[thrustData.length];
		double[] mDotOxLog    = new double[thrustData.length];
		
		// Thin out of thrust data
		int lengthInterpRaw = Math.min(thrustData.length, 101);
		int stepInterpRaw = Math.max(1, thrustData.length / 100);
		lengthInterpRaw = thrustData.length / stepInterpRaw;
		if (thrustData.length % stepInterpRaw != 0){
			lengthInterpRaw ++;
		}
		double[] timeInterpRaw     = new double[lengthInterpRaw];
		double[] mDotPropInterpRaw = new double[lengthInterpRaw];
		double[] mOxInterpRaw      = new double[lengthInterpRaw];
		double[] lcgOxInterpRaw    = new double[lengthInterpRaw];

		for (int i = 0; i < thrustData.length; i++) {
			timeArray[i]   = thrustData[i][0];
			thrustArray[i] = thrustData[i][1];
		}
		thrustArray[thrustArray.length - 1] = 0.0;
		totalImpulse = getSum(timeArray, thrustArray);		// トータルインパルス算出
		thrustAve = totalImpulse / timeBurnout;				// 平均推力算出
		mDotPropAve = mDotFuel + mOxBef / timeBurnout;		// 推進剤平均質量流量算出
		IspAve = thrustAve / (mDotPropAve * 9.80665);		// 平均比推力算出
		int j = 0;
		for (int i = 0; i < mDotOxLog.length; i++) {
			mDotPropLog[i] = thrustArray[i] / (IspAve * 9.80665);
			
			if (i % stepInterpRaw == 0) {
				timeInterpRaw[j] = timeArray[i];
				mDotPropInterpRaw[j] = mDotPropLog[i];
				j ++;
			}

			if(timeArray[i] < timeBurnout) {
				mDotFuelLog[i] = mDotFuel;
			}
			else {
				mDotFuelLog[i] = 0.0;
			}
			mDotOxLog[i] = Math.max(0.0, mDotPropLog[i] - mDotFuelLog[i]);
		}

		thrustAnaly   = new Interpolation(timeArray, thrustArray);
		mDotPropAnaly = new Interpolation(timeInterpRaw, mDotPropInterpRaw);
		timeActuate = timeArray[thrustData.length - 1];
		
		double[] mOxLog = new double[thrustData.length];
		mOxLog[0] = mOxBef;
		Interpolation mDotOxAnaly = new Interpolation(timeArray, mDotOxLog);
		Function<Double, Double> calculateMDotOx = (time) -> { return mDotOxAnaly.linearInterp1column(time); };
		j = 0;
		for (int i = 0; i < timeArray.length - 1; i++) {
			double t = timeArray[i];
			double dt = timeArray[i + 1] - timeArray[i];

			double k1 = - calculateMDotOx.apply(t);
			double k2 = - calculateMDotOx.apply(t + 0.5 * dt);
			double k3 = - calculateMDotOx.apply(t + 0.5 * dt);
			double k4 = - calculateMDotOx.apply(t + dt);
			
			mOxLog[i + 1] = Math.max(0.0, mOxLog[i] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4));

			if (i % stepInterpRaw == 0) {
				timeInterpRaw[j] = timeArray[i + 1];
				mOxInterpRaw[j] = mOxLog[i + 1];
				j ++;
			}
		}
		mOxAnaly = new Interpolation(timeInterpRaw, mOxInterpRaw);

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Length, Diameter
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// dth = engine.get("Nozzle Throat Diameter [mm]").asDouble() * Math.pow(10, -3);// [mm] => [m]
		// eps = engine.get("Nozzle Expansion Ratio").asDouble();
		// Ath = 0.25 * Math.PI * Math.pow(dth, 2);
		// Ae  = Ath * eps;
		// de  = Math.sqrt(Ae * 4 / Math.PI);
		de = engine.get("Nozzle Exit Diameter [mm]").asDouble() * Math.pow(10, -3);// [mm] => [m]
		Ae = 0.25 * Math.PI * Math.pow(de, 2);

		// Fuel(grain)
		dFuelInBef = engine.get("Fuel Inside Diameter [mm]").asDouble() * Math.pow(10, -3);
		//燃料は密度分布が一様であると仮定して,燃焼前後の重量比を用いて算出
		dFuelInAft = Math.sqrt((1 - mFuelAft / mFuelBef) * Math.pow(dFuelOut, 2) + (mFuelAft / mFuelBef) * Math.pow(dFuelInBef, 2));

		dFuelOut = engine.get("Fuel Outside Diameter [mm]").asDouble() * Math.pow(10, -3);
		lFuel = engine.get("Fuel Length [m]").asDouble();

		// Oxidizer tank
		lTank = engine.get("Tank Length [m]").asDouble();
		dTank = engine.get("Tank Diameter [mm]").asDouble() * Math.pow(10, -3);
		distanceTank = engine.get("Length Tank-End from End [m]").asDouble();

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//    Center of Gravity
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//燃料は半径方向にのみ一様に減少するとし,重心が機軸方向に移動しない仮定
		lcgFuel  = engine.get("Length Fuel-C.G. from End [m]").asDouble();
		lcgOxBef = distanceTank + 0.5 * lTank;
		lcgOxAft = distanceTank;
		double[] lcgOxLog = new double[thrustData.length];
		j = 0;
		Function<Double, Double> calculateLcgOx = (mOx) -> { return lcgOxAft - (mOx / mOxBef) * (lcgOxAft - lcgOxBef); };
		
		for (int i = 0; i < lcgOxLog.length; i++) {
			// lcgOxLog[i] = lcgOxAft - (mOxLog[i] / mOxBef) * (lcgOxAft - lcgOxBef);
			lcgOxLog[i] = calculateLcgOx.apply(mOxLog[i]);
			
			if (i % stepInterpRaw == 0) {
				timeInterpRaw[j]	= timeArray[i];
				lcgOxInterpRaw[j]	= lcgOxLog[i];
				j ++;
			}
		}
		lcgOxAnaly = new Interpolation(timeInterpRaw, lcgOxInterpRaw);

		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//     Moment of Inertia
		// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		IjFuelPitchBef = mFuelBef * ((Math.pow(dFuelInBef, 2) + Math.pow(dFuelOut, 2)) / 16.0 + Math.pow(lFuel, 2) / 12.0);
		IjFuelRollBef  = mFuelBef * (Math.pow(dFuelInBef, 2) + Math.pow(dFuelOut, 2)) / 8.0;
		IjFuelPitchAft = mFuelAft * ((Math.pow(dFuelInAft, 2) + Math.pow(dFuelOut, 2)) / 16.0 + Math.pow(lFuel, 2) / 12.0);
		IjFuelRollAft  = mFuelAft * ((Math.pow(dFuelInAft, 2) + Math.pow(dFuelOut, 2))) / 8.0;

		IjOxPitchBef = mOxBef * (Math.pow(dTank, 2) / 16.0 + Math.pow(lTank, 2) / 12.0);
		IjOxRollBef  = mOxBef * Math.pow(dTank, 2) / 8.0;
	}

	/**
	 * 推進剤質量流量取得
	 * @param t 時間
	 * @return 推進剤質量時間変化率。正の値で返されるので注意
	 * */
	public double getMdotProp(double t) {
		if(t < timeBurnout) {
			return mDotPropAnaly.linearInterp1column(t);
		}
		else {
			return 0.0;
		}
	}

	public double getMassFuel(double t) {
		if(t < timeBurnout) {
			return mFuelBef + (mFuelAft - mFuelBef) * (t / timeBurnout);
		}
		else {
			return mFuelAft;
		}
	}

	public double getMassOx(double t) {
		if(t < timeActuate) {
			return mOxAnaly.linearInterp1column(t);
		}else {
			return 0.0;
		}
	}

	/**
	 * 燃料の内径を燃焼前後の値で線形的に変化するとして計算
	 * コンストラクタの計算みたいに，質量比から計算してもいいが，あまり結果に影響ないと思う
	 * @return 燃料の内径
	 * */
	private double getDiamFuelIn(double t) {
		if(t < timeBurnout) {
			return dFuelInBef + (dFuelInAft - dFuelInBef) * (t / timeBurnout);
		}else {
			return dFuelInAft;
		}
	}

	private double getLengthOx(double t) {
		if(t < timeActuate) {
			// return lTank * (t / timeBurnout);
			return lTank * (1.0 - t / timeActuate);
		}else {
			return 0.0;
		}
	}

	/**
	 * @return エンジン後端から酸化剤重心までの距離 [m]
	 * */
	public double getLcgOx(double t) {
		if(t < timeActuate) {
			// return distanceTank + 0.5 * getLengthOx(t);
			return lcgOxAnaly.linearInterp1column(t);
		}else {
			return lcgOxAft;
		}
	}

	/**
	 * @return 燃料重心回りの燃料のPitch慣性モーメント
	 * */
	public double getIjFuelPitch(double t) {
		if(t < timeBurnout) {
			/* 中空円筒の慣性モーメント
			 * Iyy = m * ((din^2 + dout^2) / 16 + L / 12)
			 * **/
			return getMassFuel(t) * ((Math.pow(dFuelOut, 2) + Math.pow(getDiamFuelIn(t), 2)) / 16.0 + Math.pow(lFuel, 2) / 12.0);
		}else {
			return IjFuelPitchAft;
		}
	}

	/**
	 * @return 燃料重心回りの燃料のPitch慣性モーメント
	 * */
	public double getIjFuelRoll(double t) {
		if(t < timeBurnout) {
			/*中空円筒の慣性モーメント
			 * Ixx = m * (din^2 + dout^2) / 8
			 * **/
			return getMassFuel(t) * (Math.pow(getDiamFuelIn(t), 2) + Math.pow(dFuelOut, 2)) / 8.0;
		}else {
			return IjFuelRollAft;
		}
	}

	/** 
	 * 酸化剤ピッチ軸慣性モーメント取得
	 * <p>
	 * スロッシングの影響を無視。
	 * 液体酸化剤を一様円柱と仮定。
	 * @return 酸化剤重心回りの酸化剤のPitch慣性モーメント
	 * */
	public double getIjOxPitch(double t) {
		if(t < timeActuate) {
			/* 中実円筒の慣性モーメント
			 * Iyy = m * (d^2 / 16 + L^2 / 12)
			 * **/
			return getMassOx(t) * (Math.pow(dTank, 2) / 16.0 + Math.pow(getLengthOx(t), 2) / 12.0);
		}else {
			return 0.0;
		}
	}

	/**
	 * スロッシングの影響を無視。
	 * 液体酸化剤を一様円柱と仮定。
	 * @return 酸化剤重心回りの酸化剤のPitch慣性モーメント
	 * */
	public double getIjOxRoll(double t) {
		if(t < timeActuate) {
			/* 中実円筒の慣性モーメント
			 * Ixx = m * d^2 / 8
			 * **/
			return getMassOx(t) * Math.pow(dTank, 2) / 8.0;
		}else {
			return 0.0;
		}
	}

	public double thrust(double t) {
		
		return thrustAnaly.linearInterp1column(t);
	}

	private double getSum(double[] xArray, double[] yArray) {

		double sum = 0.0;

		for (int i = 0; i < xArray.length - 1; i++) {
			double dx = xArray[i + 1] - xArray[i];
			sum = sum + 0.5 * (yArray[i + 1] + yArray[i]) * dx;
		}

		return sum;
	}

}
