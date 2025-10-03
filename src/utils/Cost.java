package utils;

public interface Cost {
	
	public static final Cost QUADRATIC = new Cost() {

		public Matrix d(Matrix out, Matrix res, Matrix sum, Derivative derivative) {
			return Matrix.pro(Matrix.sub(out, res, null), Matrix.vec(sum, null, derivative), null);
		}
	};

	public static final Cost CROSS_ENTROPY = new Cost() {

		public Matrix d(Matrix out, Matrix res, Matrix sum, Derivative derivative) {
			return Matrix.sub(out, res, null);
		}
	};

	public Matrix d(Matrix out, Matrix res, Matrix sum, Derivative derivative);

}
