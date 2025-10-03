package utils;

public interface Derivative {

	/**
	 * 
	 */
	public static final Derivative Identity = new Derivative() {
		
		public double d(double val) {
			return 1;
		}
	};
	
	/**
	 * 
	 */
	public static final Derivative Sigmoid = new Derivative() {
		
		public double d(double val) {
			return Activation.Sigmoid.f(val) * (1 - Activation.Sigmoid.f(val));
		}
	};
	
	/**
	 * 
	 */
	public static final Derivative ReLU = new Derivative() {
		
		public double d(double val) {
			return val < 0 ? 0 : 1;
		}
	};
	
	public static final Derivative LeakyReLU = new Derivative() {
		
		public double d(double val) {
			return val < 0 ? 0.01 : 1;
		}
	};
	
	public static final Derivative SmoothReLU = new Derivative() {
		
		public double d(double val) {
			return Activation.Sigmoid.f(val);
		}
	};
	
	/**
	 * 
	 */
	public static final Derivative Tanh = new Derivative() {
		
		public double d(double val) {
			return 4 * Math.pow(Math.cosh(val), 2) / Math.pow(Math.cosh(2 * val) + 1, 2);
		}
	};
	
	/**
	 * the function f'(x)
	 */
	public double d(double val);
	
}
