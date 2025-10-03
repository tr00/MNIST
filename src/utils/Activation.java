package utils;

public interface Activation {
	
	/**
	 * 
	 */
	public static final Activation Identity = new Activation() {
		
		public double f(double val) {
			return val;
		}
	};
	
	/**
	 * 
	 */
	public static final Activation Sigmoid = new Activation() {
		
		public double f(double val) {
			return 1 / (1 + Math.pow(Math.E, -val));
		}
	};
	
	/**
	 * 
	 */
	public static final Activation ReLU = new Activation() {
		
		public double f(double val) {
			return Math.max(0, val);
		}
	};
	
	/**
	 * 
	 */
	public static final Activation LeakyReLU = new Activation() {
		
		public double f(double val) {
			return Math.max(0.01 * val, val);
		}
	};
	
	/**
	 * 
	 */
	public static final Activation SmoothReLU = new Activation() {
		
		public double f(double val) {
			return Math.log(1 + Math.pow(Math.E, val));
		}
	};
	
	/**
	 * 
	 */
	public static final Activation Tanh = new Activation() {
		
		public double f(double val) {
			return Math.tanh(val);
		}
	};
	
	/**
	 * the function f(x)
	 */
	public double f(double val);

}
