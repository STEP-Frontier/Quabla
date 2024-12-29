package quabla.simulator.rocket;

/**
 * Atmosphere offers values about atmosphere.
 *
 * <p> Atmosphere copes with geopotential altitude under 11 km.
 *
 * <p> Atmosphere is based of U.S. standard atmosphere 1976.
 * */
public class Atmosphere {

	private double temperture0;
	private final double Re = 6378.137 * Math.pow(10, 3); // 地球の半径


	public Atmosphere(double temperture0){
		this.temperture0 = temperture0;
	}

	private double getGeopotentialAltitude(double alt) {
		return Re * alt / (Re + alt);
	}


	public double getAirDensity(double alt) {
		double density;//[kg/m^3] air density
		double temperture = getTemperture(alt) + 273.15;//[K]
		double pressure = getAtomosphericPressure(alt);

		density = 0.0034837 * pressure / temperture;

		return density;
	}

	public double getAtomosphericPressure(double alt) {
		double pressure;//[Pa] atomospheric pressure
		double pressure0 = 101325;//[Pa] atomospheric pressure at 0 m
		double gamma = -5.256;

		pressure = pressure0 * Math.pow(288.15 / (273.15 + getTemperture(alt)), gamma);

		return pressure ;
	}

	public double getTemperture(double alt) {
		double temperture ;

		double geo_alt = getGeopotentialAltitude(alt);

		temperture = temperture0 - 0.0065 * geo_alt;

		return temperture;
	}

	public double getSoundSpeed(double alt) {
		double Cs; //[m/s] sound speed

		Cs = 20.0468 * Math.sqrt(getTemperture(alt) + 273.15);

		return Cs;
	}

	public double getGravity(double alt) {
		double gravity;
		double g0 = 9.80665;

		gravity = g0 * Math.pow(Re / (Re + alt) , 2);

		return gravity;
	}


}
