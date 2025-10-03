package utils;

public class Matrix {
	
	private double[] values;
	private int rows, cols;
	

	public Matrix(double[] values, int rows, int cols) {
		if (values.length != rows * cols)
			throw new RuntimeException("Invalid array size. Must be of size numRows * numColumns");
		
		this.values = values;
		this.rows = rows;
		this.cols = cols;
	}
	
	public Matrix(int rows, int cols) { // Create an empty matrix
		this.rows = rows;
		this.cols = cols;
		values = new double[rows * cols];
	}
	
	public int getRows() {
		return rows;
	}
	
	public int getColumns() {
		return cols;
	}
	
	public double[] getValues() {
		return values;
	}
	
	public boolean withinRange(int row, int column) {
		return row < rows && row >= 0 && column < cols && column >= 0;
	}
	
	public void set(int row, int column, double value) {
		if(withinRange(row, column))
			values[row * cols + column] = value;
		else
			throw new RuntimeException("Out of matrix bound");
	}
	
	public double get(int row, int column) {
		if(withinRange(row, column))
			return values[row * cols + column];
		else
			throw new RuntimeException("Out of matrix bound");
	}
	
	public void clear() {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				set(i, j, 0);
			}
		}
	}
	
	public void fill(double a){
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				set(i, j, a);
			}
		}
	}
	
	/*** TRANSPOSE ***/
	
	public Matrix transpose() {
		if (rows == 1 || cols == 1) // More efficient for vectors
			return new Matrix(values, cols, rows);
		
		Matrix transposed = new Matrix(cols, rows);
		transpose(this, transposed);
		
		return transposed;
	}
	
	public static void transpose(Matrix m, Matrix dst) {
		if (m.getRows() != m.getColumns() || m.getColumns() != m.getRows())
			throw new RuntimeException("Matrix size mismatch");
			
		for(int i = 0; i < m.getRows(); i++) {
			for(int j = 0; j < m.getColumns(); j++) {
				dst.set(j, i, m.get(i, j));
			}
		}
	}
	
	/*** ADDITION ***/
	
	public Matrix add(Matrix m) {
		Matrix newMatrix = new Matrix(rows, cols);
		add(this, m, newMatrix);
		
		return newMatrix;
	}
	
	public void addSelf(Matrix m) {
		add(this, m, this);
	}
	
	public static Matrix add(Matrix src1, Matrix src2, Matrix dst) {
		if(src1.getRows() != src2.getRows() || src1.getColumns() != src2.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		} else if(dst == null) {
			dst = new Matrix(src1.getRows(), src1.getColumns());
		} else if(src1.getRows() != dst.getRows() || src1.getColumns() != dst.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		}
			
		for(int i = 0; i < src1.getRows(); i++) {
			for(int j = 0; j < src1.getColumns(); j++) {
				dst.set(i, j, src1.get(i, j) + src2.get(i, j));
			}
		}
		return dst;
	}
	
	/*** SUBTRACTION ***/
	
	public Matrix sub(Matrix m) {
		Matrix newMatrix = new Matrix(rows, cols);
		sub(this, m, newMatrix);
		
		return newMatrix;
	}
	
	public void subtractSelf(Matrix m) {
		sub(this, m, this);
	}
	
	public static Matrix sub(Matrix a, Matrix b, Matrix dst) {
		if(dst == null) {
			dst = new Matrix(a.getRows(), a.getColumns());
		} else if(a.getRows() != b.getRows() || a.getColumns() != b.getColumns() || a.getRows() != dst.getRows() || a.getColumns() != dst.getColumns())
			throw new RuntimeException("Matrix size mismatch");
		
		for(int i = 0; i < a.getRows(); i++) {
			for(int j = 0; j < a.getColumns(); j++) {
				dst.set(i, j, a.get(i, j) - b.get(i, j));
			}
		}
		return dst;
	}
	
	/*** MULTIPLICATION ***/
	
	public double getMultipliedEntry(Matrix m, int row, int column) {
		return getMultipliedEntry(this, m, row, column);
	}
	
	public static double getMultipliedEntry(Matrix a, Matrix b, int row, int column) {
		//if(!(row < a.getRows() && row >= 0 && column < b.getColumns() && column >= 0))
		//	throw new RuntimeException("Out of matrix bounds");
		
		double sum = 0;
		
		for(int i = 0; i < a.getColumns(); i++) {
			sum += a.get(row, i) * b.get(i, column);
		}
		
		return sum;
	}
	
	public Matrix multiply(double scalar) { //Scalar multiple
		Matrix newMatrix = new Matrix(rows, cols);
		scl(this, newMatrix, scalar);
		
		return newMatrix;
	}
	
	public void multiplySelf(double scalar) {
		scl(this, this, scalar);
	}
	
	public static Matrix scl(Matrix src, Matrix dst, double val) {
		if(dst == null) {
			dst = new Matrix(src.getRows(), src.getColumns());
		} else if(src.getRows() != dst.getRows() || src.getColumns() != dst.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		}
		
		for(int i = 0; i < src.getRows(); i++) {
			for(int j = 0; j < src.getColumns(); j++) {
				dst.set(i, j, val * src.get(i, j));
			}
		}
		return dst;
	}
	
	public Matrix dot(Matrix m) {
		Matrix newMatrix = new Matrix(rows, m.getColumns());
		dot(this, m, newMatrix);
		
		return newMatrix;
	}
	
	public static Matrix dot(Matrix src1, Matrix src2, Matrix dst) {
		if(src1.getColumns() != src2.getRows()) {
			throw new RuntimeException("Matrix size mismatch");
		} else if(dst == null) {
			dst = new Matrix(src1.getRows(), src2.getColumns());
		} else if(dst.getRows() != src1.getRows() || dst.getColumns() != src2.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		}
		
		for(int i = 0; i < src1.getRows(); i++) {
			for(int j = 0; j < src2.getColumns(); j++) {
				dst.set(i, j, getMultipliedEntry(src1, src2, i, j));
			}
		}
		return dst;
	}
	
	/*** MULTIPLY WITH TRANSPOSE ***/

	// Transpose this matrix and multiply with m
	public Matrix multiplyTransposeSelf(Matrix m) {
		Matrix newMatrix = new Matrix(cols, m.getColumns());
		multiplyTransposeA(this, m, newMatrix);

		return newMatrix;
	}
	
	public static void multiplyTransposeA(Matrix a, Matrix b, Matrix dst) {
		if(a.getRows() != b.getRows() || dst.getRows() != a.getColumns() || dst.getColumns() != b.getColumns())
			throw new RuntimeException("Matrix size mismatch");
		for(int i = 0; i < a.getColumns(); i++) {
			for(int j = 0; j < b.getColumns(); j++) {
				double total = 0;
				for (int k = 0; k < a.getRows(); k++) {
					total += a.get(k, i) * b.get(k, j);
				}
				dst.set(i, j, total);
			}
		}
	}

	// Multiply this matrix with the transpose of m
	public Matrix multiplyTransposeM(Matrix m) {
		Matrix newMatrix = new Matrix(rows, m.getRows());
		multiplyTransposeB(this, m, newMatrix);

		return newMatrix;
	}
	
	public static void multiplyTransposeB(Matrix a, Matrix b, Matrix dst) {
		if(a.getColumns() != b.getColumns() || dst.getRows() != a.getRows() || dst.getColumns() != b.getRows())
			throw new RuntimeException("Matrix size mismatch");
		
		for(int i = 0; i < a.getRows(); i++) {
			for(int j = 0; j < b.getRows(); j++) {
				double total = 0;
				for (int k = 0; k < a.getColumns(); k++) {
					total += a.get(i, k) * b.get(j, k);
				}		
				dst.set(i, j, total);
			}
		}
	}

	/*** HADAMARD PRODUCT ***/
	
	public Matrix hadamardProduct(Matrix m) {
		Matrix newMatrix = new Matrix(rows, cols);
		pro(this, m, newMatrix);
		
		return newMatrix;
	}
	
	public void hadamardProductSelf(Matrix m) {
		pro(this, m, this);
	}
	
	public static Matrix pro(Matrix a, Matrix b, Matrix dst) {
		if(dst == null) {
			dst = new Matrix(a.getRows(), a.getColumns());
		} else if(a.getRows() != b.getRows() || a.getColumns() != b.getColumns() || a.getRows() != dst.getRows() || a.getColumns() != dst.getColumns())
			throw new RuntimeException("Matrix size mismatch");
		
		for(int i = 0; i < a.getRows(); i++) {
			for(int j = 0; j < a.getColumns(); j++) {
				dst.set(i, j, a.get(i, j) * b.get(i, j));
			}
		}
		return dst;
	}
	
/*** HADAMARD DIVISION ***/
	
	public Matrix hadamardDivision(Matrix m) {
		Matrix newMatrix = new Matrix(rows, cols);
		hadamardDivision(this, m, newMatrix);
		
		return newMatrix;
	}
	
	public void hadamardDivisionSelf(Matrix m) {
		hadamardDivision(this, m, this);
	}
	
	public static void hadamardDivision(Matrix a, Matrix b, Matrix dst) {
		if(a.getRows() != b.getRows() || a.getColumns() != b.getColumns() || a.getRows() != dst.getRows() || a.getColumns() != dst.getColumns())
			throw new RuntimeException("Matrix size mismatch");
		
		for(int i = 0; i < a.getRows(); i++) {
			for(int j = 0; j < a.getColumns(); j++) {
				dst.set(i, j, a.get(i, j) / b.get(i, j));
			}
		}
	}
	
	/*** STATIC MATRIX GENERATORS ***/
	
	public static Matrix getIdentity(int size) {
		Matrix newMatrix = new Matrix(size, size);
		
		for(int i = 0; i < size; i++) {
			newMatrix.set(i, i, 1);
		}
		
		return newMatrix;
	}
	
	public static Matrix getZero(int size) {
		return new Matrix(size, size);
	}
	
	/*** VECTORIZATION ***/
	
	public Matrix vectorize(Activation f) {
		Matrix newMatrix = new Matrix(rows, cols);
		vec(this, newMatrix, f);
		
		return newMatrix;
	}
	
	public void vectorizeSelf(Activation f) {
		vec(this, this, f);
	}
	
	public static Matrix vec(Matrix m, Matrix dst, Activation f) {
		if(dst == null) {
			dst = new Matrix(m.getRows(), m.getColumns());
		} else if(m.getRows() != dst.getRows() || m.getColumns() != dst.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		}
		
		for(int i = 0; i < m.getRows(); i++) {
			for(int j = 0; j < m.getColumns(); j++) {
				dst.set(i, j, f.f(m.get(i, j)));
			}
		}
		return dst;
	}
	
	public static Matrix vec(Matrix m, Matrix dst, Derivative d) {
		if(dst == null) {
			dst = new Matrix(m.getRows(), m.getColumns());
		} else if(m.getRows() != dst.getRows() || m.getColumns() != dst.getColumns()) {
			throw new RuntimeException("Matrix size mismatch");
		}
		
		for(int i = 0; i < m.getRows(); i++) {
			for(int j = 0; j < m.getColumns(); j++) {
				dst.set(i, j, d.d(m.get(i, j)));
			}
		}
		return dst;
	}
	
	public static int[] max(Matrix src) {
		double maxValue = Double.NEGATIVE_INFINITY;
		int[] maxIndex = new int[] {0, 0};
		for(int i = 0; i < src.getRows(); i++) {
			for(int j = 0; j < src.getColumns(); j++) {
				if(src.get(i, j) > maxValue) {
					maxValue = src.get(i, j);
					maxIndex = new int[] {i, j};
				}
			}
		}
		return maxIndex;
	}
	
	/*** OBJECT METHODS ***/
	
	public boolean equals(Matrix m) {
		if(rows != m.getRows() || cols != m.getColumns())
			return false;
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				if(get(i, j) != m.get(i, j)) {
					return false;
				}
			}
		}
		
		return true;
	}
	
	public String toString() {
		String str = "";
		for(int i = 0; i < rows; i++) {
			str += "{";
			for(int j = 0; j < cols; j++) {
				str += j != cols - 1 ? get(i, j) + ", " : get(i, j);
			}
			str += i != rows - 1 ? "},\n" : "}";
		}
		
		return str;
	}
	
}