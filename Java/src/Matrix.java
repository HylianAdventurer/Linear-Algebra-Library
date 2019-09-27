import java.awt.*;

public class Matrix implements Cloneable {
    private Double[][] data;

    /**
     * CONSTRUCTORS
     **/

    public Matrix() {
        this(2, 2);
    }

    public Matrix(int m) {
        this(m, m);
    }

    public Matrix(int m, int n) {
        data = new Double[m][n];
        for (int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == j) {
                    data[i][i] = 1.0;
                } else {
                    data[i][j] = 0.0;
                }
            }
        }
    }

    public Matrix(Rectangle size) {
        this(size.width, size.height);
    }

    public Matrix(Double[][] data) {
        this.data = data;
    }

    /**
     * MECHANICS
     **/

    public Rectangle getSize() {
        return new Rectangle(this.data.length, this.data[0].length);
    }

    public int getNumRows() {
        return this.data.length;
    }

    public int getNumColumns() {
        return this.data[0].length;
    }

    public boolean isSquare() {
        return this.getNumRows() == this.getNumColumns();
    }

    public double trace() {
        if(this.isSquare()) {
            double result = 0.0;
            for(int i = 0; i < this.getNumRows(); i++) result += this.get(i,i);
            return result;
        } else {
            throw new RuntimeException("Not Square Exception:\nMatrix must be square to find the trace\n" + this.getSize());
        }
    }

    public static Matrix add(Matrix m1, Matrix m2) {
        if(m1.getSize() == m2.getSize()) {
            Matrix result = new Matrix(m1.getSize());
            for(int i = 0; i < m1.getNumRows(); i++) {
                for(int j = 0; j < m1.getNumColumns(); j++) {
                    result.set(i, j, m1.get(i, j) + m2.get(i, j));
                }
            }
            return result;
        } else {
            throw new RuntimeException("Size Mismatch Exception:\nMatrix 1: " + m1.getSize() + " and Matrix 2: " + m2.getSize());
        }
    }

    public Matrix add(Matrix m) {
        this.data = add(this, m).data;
        return this;
    }

    public static Matrix subtract(Matrix m1, Matrix m2) {
        return add(m1, multiply(-1,m2));
    }

    public Matrix subtract(Matrix m) {
        this.data = add(this, multiply(-1, m)).data;
        return this;
    }

    public static Matrix multiply(double k, Matrix m) {
        Matrix result = (Matrix) m.clone();

        for(int i = 0; i < result.data.length; i++) {
            for(int j = 0; j < result.data[i].length; j++) {
                result.set(i, j, result.get(i, j) * k);
            }
        }

        return result;
    }

    public Matrix multiply(double k) {
        this.data = Matrix.multiply(k, this).data;
        return this;
    }

    public static Matrix multiply(Matrix m1, Matrix m2) {
        if(m1.getNumColumns() == m2.getNumRows()) {

            Matrix result = new Matrix(m1.getNumRows(), m2.getNumColumns());

            for (int i = 0; i < result.getNumRows(); i++) {
                for (int j = 0; j < result.getNumColumns(); j++) {
                    double x = 0;

                    for (int k = 0; k < m1.getNumColumns(); k++) {
                        x += m1.get(i,k) * m2.get(k, j);
                    }

                    result.set(i,j, x);
                }
            }

            return result;
        } else {
            throw new RuntimeException("Incompatible size Exception:\nMatrix 1: " + m1.getSize() + " and Matrix 2: " + m2.getSize());
        }
    }

    public Matrix multiply(Matrix m) {
        this.data = Matrix.multiply(this, m).data;
        return this;
    }

    public double determinant() {
        if(this.isSquare()) {
            if(this.getNumRows() == 1) {
                return this.get(0,0);
            }else if(this.getNumRows() == 2) {
                return this.get(0,0) * this.get(1,1) - this.get(0,1) * this.get(1, 0);
            } else {
                double result = 0.0;

                for(int i = 0; i < this.getNumColumns(); i++) result += (i % 2 == 0 ? -1.0 : 1.0) * this.get(1, i) * minor(this, 1, i).determinant();

                return result;
            }
        } else {
            throw new RuntimeException("Not Square Exception:\nMatrix must be square to find determinant\n" + this.getSize());
        }
    }

    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.getNumColumns(), m.getNumRows());

        for(int i = 0; i < m.getNumRows(); i++) {
            for(int j = 0; j < m.getNumColumns(); j++) {
                result.set(j, i, m.get(i, j));
            }
        }

        return result;
    }

    public Matrix transpose() {
        this.data = transpose(this).data;
        return this;
    }

    public Matrix inverse() {
        if(isSquare() && determinant() != 0) {
            return multiply(1 / determinant(), adjoint());
        } else if(!isSquare()){
            throw new RuntimeException("Not Square Exception:\nMatrix must be square to find inverse\n" + getSize());
        } else {
            throw new RuntimeException("Not Singular Exception:\nMatrix must be singular, that is, the determinant must not be 0");
        }

    }

    public Matrix adjoint() {
        return transpose(matrixOfCofactors());
    }

    public Matrix matrixOfCofactors() {
        Matrix result = new Matrix(this.getSize());

        for(int i = 0; i < result.getNumRows(); i++) {
            for(int j = 0; j < result.getNumColumns(); j++) {
                result.set(i, j, cofactor(i, j));
            }
        }

        return result;
    }

    public double cofactor(int i, int j) {
        return ((i + j) % 2 == 0 ? 1.0 : -1.0) * minor(this, i, j).determinant();
    }

    private static Matrix minor(Matrix m, int r, int c) {
        Matrix result = new Matrix(m.getNumRows() - 1, m.getNumColumns() - 1);

        for(int i = 0; i < m.getNumRows(); i++) {
            for(int j = 0; j < m.getNumColumns(); j++) {
                if(i != r && j != c) result.set((i < r ? i : (i - 1)), (j < c ? j : (j - 1)), m.get(i, j));
            }
        }

        return result;
    }

    /**
     * GETTERS/SETTERS
     **/

    public Double[][] get() {
        return data;
    }

    public void set(Double[][] data) {
        this.data = data;
    }

    public Double[] get(int m) {
        return data[m];
    }

    public void set(int m, Double[] row) {
        if (m > data.length) {
            throw new RuntimeException("The row selected (" + m + ") does not exist.\n" +
                    "There are " + data.length + " rows in the matrix");
        } else if (row.length != data[0].length) {
            throw new RuntimeException("The row sizes do not match.\n" +
                    "Matrix size: " + data[0].length + "\n" +
                    "Row size: " + row.length);
        } else {
            data[m] = row;
        }
    }

    public double get(int m, int n) {
        return data[m][n];
    }

    public void set(int m, int n, double x) {
        if (m > data.length || n > data[0].length) {
            throw new RuntimeException("Specified position (" + m + "," + n + ") " +
                    "is not the the matrix " + this.getSize());
        } else {
            data[m][n] = x;
        }
    }


    /**
     * INTERFACING
     **/

    @Override
    public String toString() {
        String result = "";
        for (int i = 0; i < data[0].length; i++) {
            result += "[";
            for (int j = 0; j < data.length; j++) {
                result += String.format("%6.5s", (data[i][j] < 10000 && data[i][j] > -10000) ? data[i][j] : "ovrflw") + ",";
            }
            result = result.substring(0, result.length() - 1) + "]\n";
        }
        return result;
    }

    @Override
    public Object clone() {
        Matrix result = new Matrix(this.getSize());

        for(int i = 0; i < this.data.length; i++) {
            for(int j = 0; j < this.data[i].length; j++) {
                result.set(i, j, this.get(i, j));
            }
        }

        return result;
    }
}
