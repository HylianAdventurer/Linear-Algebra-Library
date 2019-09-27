import java.awt.*;
import java.util.Vector;

public class LinearAlgebra {
    /* ***********
        FUNCTIONS
       *********** */

    /* Mathematics */
    /**
     * Returns the trace of a matrix
     * @param m double[][]: Matrix to use
     * @return double
     * @throws NotSquareException Throws when the matrix is not square
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double trace(double[][] m) {
        if(!isSquare(m)) throw new NotSquareException(m, "Not Square Exception: Matrix must be square to find the trace");

        double result = 0.0;
        for(int i = 0; i < m.length; i++) result += m[i][i];
        return result;
    }

    /**
     * Adds two matrices together
     * @param m1 double[][]: First matrix to be added
     * @param m2 double[][]: Second matrix to be added
     * @return double[][]: Result matrix
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[][] add(double[][] m1, double[][] m2) {
        if(!getSize(m1).equals(getSize(m2))) throw new MatrixSizeMismatchException(m1,m2, "MatrixSizeMismatchException: Matrices must be the same size to add");

        double[][] result = newMatrix(getSize(m1));

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                result[i][j] = m1[i][j] + m2[i][j];
            }
        }

        return result;
    }

    /**
     * Adds two vectors together
     * @param v1 double[]: First vector to be added
     * @param v2 double[]: Second vector to be added
     * @return double[][]: Result vector
     * @throws VectorSizeMismatchException Thrown if sizes of vectors do not match
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[] add(double[] v1, double[] v2) {
        if(v1.length!=v2.length) throw new VectorSizeMismatchException(v1,v2, "VectorSizeMismatchException: Vectors must be the same size to add");

        double[] result = new double[v1.length];

        for(int i = 0; i < v1.length; i++) result[i] = v1[i] + v2[i];

        return result;
    }

    /**
     * Subtracts two matrices together
     * @param m1 double[][]: First matrix to be subtracted
     * @param m2 double[][]: Second matrix to be subtracted
     * @return double[][]: Result matrix
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[][] subtract(double[][] m1, double[][] m2) {
        if(!getSize(m1).equals(getSize(m2))) throw new MatrixSizeMismatchException(m1,m2, "MatrixSizeMismatchException: Matrices must be the same size to subtract");

        double[][] result = newMatrix(getSize(m1));

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                result[i][j] = m1[i][j] - m2[i][j];
            }
        }

        return result;
    }

    /**
     * Subtracts two vectors together
     * @param v1 double[]: First vector to be subtracted
     * @param v2 double[]: Second vector to be subtracted
     * @return double[][]: Result vector
     * @throws VectorSizeMismatchException Thrown if sizes of vectors do not match
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[] subtract(double[] v1, double[] v2) {
        if(v1.length!=v2.length) throw new VectorSizeMismatchException(v1,v2, "VectorSizeMismatchException: Vectors must be the same size to subtract");

        double[] result = new double[v1.length];

        for(int i = 0; i < v1.length; i++) result[i] = v1[i] - v2[i];

        return result;
    }

    /**
     * Multiplies a matrix by a constant and returns the result
     * @param k double: Constant matrix is multiplied by
     * @param m double[][]: Matrix to be multiplied
     * @return double[][]: Result of multiplication
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[][] multiply(double k, double[][] m) {
        double[][] result = newMatrix(getSize(m));

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                result[i][j] = k * m[i][j];
            }
        }

        return result;
    }

    /**
     * Multiplies a vector by a constant and returns the result
     * @param k double: Constant matrix is multiplied by
     * @param v double[][]: Vector to be multiplied
     * @return double[]: Result of multiplication
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static double[] multiply(double k, double[] v) {
        double[] result = new double[v.length];
        for(int i = 0; i < v.length; i++) result[i] = k * v[i];
        return result;
    }

    /**
     * Multiplies two matrices together and returns the new matrix
     * @param m1 double[][]: First matrix being multiplied
     * @param m2 double[][]: Second matrix being multiplied
     * @return double[][]: Resulting matrix
     * @throws InvalidMatrixException Throws when matrix is invalid
     * @throws MatrixSizeMismatchException Throws when m1 columns and m2 rows do not match
     */
    public static double[][] multiply(double[][] m1, double[][] m2) {
        if(getWidth(m1) != getHeight(m2)) throw new MatrixSizeMismatchException(m1, m2, "MatrixSizeMismatchException: Number of columns in matrix 1 must be equal to number of rows in matrix 2");
        double[][] result = newMatrix(m1.length,m2[0].length);

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                double x = 0;

                for(int k = 0; k < m1[0].length; k++) {
                    x += m1[i][k] * m2[k][j];
                }

                result[i][j] = x;
            }
        }

        return result;
    }


    /* Utilities */
    /**
     * Checks to make sure all rows in a matrix are the same size. If so returns true; else returns false
     * @param m double[][]: Matrix to be check
     * @return boolean
     */
    public static boolean validMatrix(double[][] m) {
        int width = m[0].length;
        for(int i = 1; i < m.length; i++) {
            if(width!=m[i].length) return false;
        }
        return true;
    }

    /**
     * Returns a Rectangle object representing the size of the matrix
     * @param m double[][]: Matrix to find size of
     * @return Rectangle: Size of matrix
     * @throws InvalidMatrixException Throws when matrix is invalid
     * @apiNote new Rectangle(m.length,m[0].length) returns the same result and is faster but will not catch InvalidMatrixException
     */
    public static Rectangle getSize(double[][] m) {
        if(!validMatrix(m)) throw new InvalidMatrixException(m);
        return new Rectangle(m.length,m[0].length);
    }

    /**
     * Returns the height (the number of rows) of given matrix
     * @param m double[][]: Matrix to be checked
     * @return int: Height of matrix
     * @throws InvalidMatrixException Throws when matrix is invalid
     * @apiNote m.length returns the same result and is faster but will not catch InvalidMatrixException
     */
    public static int getHeight(double[][] m) {
        if(!validMatrix(m)) throw new InvalidMatrixException(m);
        return m.length;
    }

    /**
     * Returns the height (the number of columns) of a given matrix
     * @param m double[][]: Matrix to be check
     * @return int: Width of matrix
     * @throws InvalidMatrixException Throws when matrix is invalid
     * @apiNote m[0].length returns the same result and is faster but will not catch InvalidMatrixException
     */
    public static int getWidth(double[][] m) {
        if(!validMatrix(m)) throw new InvalidMatrixException(m);
        return m[0].length;
    }

    /**
     * Checks if the number of rows is equal to number of columns
     * @param m double[][]: Matrix to be checked
     * @return boolean
     */
    public static boolean isSquare(double[][] m) {
        return getHeight(m) == getWidth(m);
    }

    /**
     * Returns a new matrix of specified size with all zeros
     * @param height int: Height (number of rows) of new matrix
     * @param width int: Width (number of columns) of new matrix
     * @return double[][]: New Matrix
     */
    public static double[][] newMatrix(int height, int width) {
        return new double[height][width];
    }

    /**
     * Returns a new matrix of specified size with all zeros
     * @param size Rectangle: Specifies size of new matrix
     * @return double[][]: New matrix
     */
    public static double[][] newMatrix(Rectangle size) {
        return new double[size.height][size.width];
    }

    /**
     * Returns an identity matrix of specified size
     * @param size int: size of new identity matrix
     * @return double[][]: New identity matrix
     */
    public static double[][] newIdentity(int size) {
        double[][] result = new double[size][size];
        for(int i = 0; i < size; i++) {
            result[i][i] = 1.0;
        }
        return result;
    }

    /**
     * Returns a string that describes an matrix
     * @param m double[][]: Matrix to be described
     * @return String
     */
    public static String toString(double[][] m) {
        StringBuilder result = new StringBuilder();
        for(double[] r : m) {
            result.append("[");
            for(double x : r) {
                result.append(String.format("%6.5s,", (x < 10000 && x > -10000) ? x : "ovrflw"));
            }
            result.replace(result.lastIndexOf(","),result.lastIndexOf(",") + 1, "]\n");
        }
        return result.toString();
    }

    /**
     * Returns a string that describes an vector
     * @param v double[][]: Vector to be described
     * @return String
     */
    public static String toString(double[] v) {
        StringBuilder result = new StringBuilder();
        result.append("[");
        for(double x : v) {
            result.append(String.format("%6.5s,", (x < 10000 && x > -10000) ? x : "ovrflw"));
        }
        result.replace(result.lastIndexOf(","),result.lastIndexOf(",") + 1, "]\n");

        return result.toString();
    }


    /* ************
        EXCEPTIONS
       ************ */
    /* Matrix Exceptions */
    /**
     * Contains a copy of the matrix that caused the exception to occur for error checking purposes
     */
    private static class MatrixRuntimeException extends RuntimeException {
        double[][] matrix;

        private MatrixRuntimeException(double[][] m, String message) {
            super(message);
            matrix = m;
        }
    }

    /**
     * Occurs when the matrix has an invalid format
     * @throws InvalidMatrixException Throws when matrix is invalid
     */
    public static class InvalidMatrixException extends MatrixRuntimeException {
        public InvalidMatrixException(double[][] m, String message) {
            super(m,message);
        }

        public InvalidMatrixException(double[][] m) {
            super(m, "InvalidMatrixException: Not all rows are of the same size");
        }
    }

    /**
     * Occurs when a matrix is expected to be square and is not
     * @throws NotSquareException Throws when the matrix is not square
     */
    public static class NotSquareException extends MatrixRuntimeException {
        public NotSquareException(double[][] m, String message) {
            super(m,message);
        }

        public NotSquareException(double[][] m) {
            super(m,"Not Square Exception: Operation requires matrix to be square");
        }
    }

    /**
     * Occurs when two matrices are expected to have compatible sizes
     * Stores both matrices
     */
    public static class MatrixSizeMismatchException extends MatrixRuntimeException {
        double[][] matrix2;
        public MatrixSizeMismatchException(double[][] m1, double[][] m2, String message) {
            super(m1, message);
            this.matrix2 = m2;
        }

        public MatrixSizeMismatchException(double[][] m1, double[][] m2) {
            super(m1, "Matrix Size Mismatch Exception: Matrices must have compatible sizes to perform operation");
            this.matrix2 = m2;
        }
    }

    /* Vector Exceptions */
    /**
     * Contains a copy of the vector that caused the exception to occur for error checking purposes
     */
    private static class VectorRuntimeException extends RuntimeException {
        double[] vector;

        private VectorRuntimeException(double[] v, String message) {
            super(message);
            this.vector = v;
        }
    }

    /**
     * Occurs when two vectors are expected to have compatible sizes
     * Stores both vectors
     */
    public static class VectorSizeMismatchException extends VectorRuntimeException {
        double[] vector2;

        public VectorSizeMismatchException(double[] v, double[] v2, String message) {
            super(v, message);
            this.vector2 = v2;
        }

        public VectorSizeMismatchException(double[] v, double[] v2) {
            super(v, "VectorSizeMismatchException: Vectors must be the same size to perform operation");
            this.vector2 = v2;
        }
    }
}