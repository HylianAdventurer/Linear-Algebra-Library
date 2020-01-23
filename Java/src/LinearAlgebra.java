public class LinearAlgebra {
    /* *************
         FUNCTIONS
       ************* */

    /* Mathematics */
    /**
     * Adds two matrices together
     * @param matrix1 double[][]: First matrix used for addition
     * @param matrix2 double[][]: Second matrix used for addition
     * @return double[][]: Result matrix
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] add(final double[][] matrix1, final double[][] matrix2) {
        if(!validMatrix(matrix1)) throw new InvalidMatrixException(matrix1);
        if(!validMatrix(matrix2)) throw new InvalidMatrixException(matrix2);
        if(matrix1.length!=matrix2.length||matrix1[0].length!=matrix2[0].length) throw new MatrixSizeMismatchException(matrix1,matrix2, "MatrixSizeMismatchException: Matrices must be the same size to add");

        double[][] result = new double[matrix1.length][matrix1[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = matrix1[i][j] + matrix2[i][j];

        return result;
    }

    /**
     * Adds two vectors together
     * @param vector1 double[]: First vector to be added
     * @param vector2 double[]: Second vector to be added
     * @return double[][]: Result vector
     * @throws VectorSizeMismatchException Thrown if sizes of vectors do not match
     */
    public static double[] add(final double[] vector1, final double[] vector2) {
        if(vector1.length!=vector2.length) throw new VectorSizeMismatchException(vector1,vector2, "VectorSizeMismatchException: Vectors must be the same size to add");

        double[] result = new double[vector1.length];

        for(int i = 0; i < vector1.length; i++) result[i] = vector1[i] + vector2[i];

        return result;
    }

    /**
     * Returns the adjugate matrix of a matrix
     * @param matrix double[][]: The matrix to find the adjugate matrix of
     * @return double[][]: The adjugate matrix of the given matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double cofactor(double[][],int,int)
     * @uses double determinant(double[][])
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] adjugateMatrix(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[j][i] = cofactor(matrix, i, j);

        return result;
    }

    /**
     * Returns the cofactor of a matrix for some given row and column
     * @param matrix double[][]: The matrix to find the cofactor of
     * @param row int: The row number used to find the cofactor [Starts at 1]
     * @param column int: The column number used to find the cofactor [Starts at 1]
     * @return double: The cofactor of the matrix given row r and column c
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double determinant(double[][])
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean validMatrix(double[][])
     */
    public static double cofactor(final double[][] matrix, int row, int column) {
        return ((row-1 + column-1) % 2 == 0 ? 1.0 : -1.0) * determinant(minor(matrix, row, column));
    }

    /**
     * Returns the determinant of the given matrix
     * @param matrix double[][]: The matrix to find the determinant of
     * @return double: The determinant of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean isSquare(double[][])
     * @uses boolean validMatrix(double[][])
     */
    public static double determinant(final double[][] matrix) {
        if(!isSquare(matrix)) throw new NotSquareException(matrix);
        if(matrix.length==1) return matrix[0][0];
        if(matrix.length==2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        double result = 0.0;

        for(int i = 0; i < matrix.length; i++) result += (i % 2 != 0 ? -1.0 : 1.0) * matrix[0][i] * determinant(minor(matrix,1,i + 1));

        return result;
    }

    /**
     * Finds the dot product of two vectors
     * @param vector1 double[][]: First vector to find dot product of
     * @param vector2 double[][]: Second vector to find dot product of
     * @return double: Dot product of two matrices
     * @throws VectorSizeMismatchException Thrown when v1 and v2 are not the same size
     */
    public static double dotProduct(final double[] vector1, final double[] vector2) {
        if(vector1.length!=vector2.length) throw new VectorSizeMismatchException(vector1,vector2, "MatrixSizeMismatchException: Matrices must be the same size to find the dot product");

        double result = 0;
        for(int i = 0; i < vector1.length; i++)
            result += vector1[i] * vector2[i];

        return result;
    }

    /**
     * Finds the dot product of two matrices
     * @param matrix1 double[][]: First matrix to find dot product of
     * @param matrix2 double[][]: Second matrix to find dot product of
     * @return double: Dot product of two matrices
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown when m1 columns and m2 rows do not match
     * @uses boolean validMatrix(double[][])
     */
    public static double dotProduct(final double[][] matrix1, final double[][] matrix2) {
        if(!validMatrix(matrix1)) throw new InvalidMatrixException(matrix1);
        if(!validMatrix(matrix2)) throw new InvalidMatrixException(matrix2);
        if(matrix1.length!=matrix2.length||matrix1[0].length!=matrix2[0].length) throw new MatrixSizeMismatchException(matrix1,matrix2, "MatrixSizeMismatchException: Matrices must be the same size to find the dot product");

        double result = 0;
        for(int i = 0; i < matrix1.length; i++)
            for(int j = 0; j < matrix1[0].length; j++)
                result += matrix1[i][j] * matrix2[i][j];
        return result;
    }

    /**
     * Returns the found eigenvalues of a given matrix
     * @param matrix double[][]: The matrix to find the eigenvalues of
     * @return double[]: An array of found eigenvalues
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses boolean isSquare(double[][])
     * @uses double[][] newIdentityMatrix(int)
     * @uses double[] p_findRoots(double[])
     * @uses double[][] p_polynomialDerivative(double[])
     * @uses double[] p_polynomialDeterminant(double[][],double[][])
     * @uses double p_polynomialValue(double[],double)
     * @uses double[] p_removeRoot(double[],double)
     * @uses boolean validMatrix(double[][])
     * @apiNote This function is not guaranteed to find all eigenvalues. This function may not find all eigenvalues for matrices with multiple irrational eigenvalues
     */
    public static double[] eigenvalues(final double[][] matrix) {
        if(!isSquare(matrix)) throw new NotSquareException(matrix);
        double[] equation = p_polynomialDeterminant(matrix,newIdentityMatrix(matrix.length));
        double[] eigenvalues = p_findRoots(equation);

        int n = eigenvalues.length;
        for(int i = 0; i < n; i++) {
            double min = eigenvalues[i];
            int min_pos = i;
            for(int j = i+1; j < n; j++)
                if(eigenvalues[j] < min) {
                    min = eigenvalues[j];
                    min_pos = j;
                }
            if(i != 0) {
                if(min == eigenvalues[i-1]) {
                    double t = eigenvalues[--n];
                    eigenvalues[n] = min;
                    eigenvalues[min_pos] = t;
                    i--;
                } else {
                    double t = eigenvalues[i];
                    eigenvalues[i] = min;
                    eigenvalues[min_pos] = t;
                }
            } else {
                double t = eigenvalues[i];
                eigenvalues[i] = min;
                eigenvalues[min_pos] = t;
            }
        }

        double[] result = new double[n];
        System.arraycopy(eigenvalues,0,result,0,n);
        return result;
    }

    /**
     * Returns the found eigenvectors of a given matrix
     * @param matrix double[][]: The matrix to find the eigenvectors
     * @return double[][]: An array of found eigenvector bases (as unit vectors)
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double[][] clone(double[][])
     * @uses boolean isSquare(double[][])
     * @uses double magnitude(double[])
     * @uses double[][] newIdentityMatrix(int)
     * @uses double[] p_findRoots(double[])
     * @uses double[][] p_polynomialDerivative(double[])
     * @uses double[] p_polynomialDeterminant(double[][],double[][])
     * @uses double p_polynomialValue(double[],double)
     * @uses double[] p_removeRoot(double[],double)
     * @uses double[][] rowEchelon(double[][])
     * @uses double[][] rowReducedEchelon(double[][])
     * @uses double[][] subtract(double[][],double[][])
     * @uses double[][] scalarMultiply(double,double[][])
     * @uses double[] unitVector(double[])
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] eigenvectors(final double[][] matrix) {
        if(!isSquare(matrix)) throw new NotSquareException(matrix);
        double[] eigenvalues = eigenvalues(matrix);
        double[][] result = new double[eigenvalues.length][matrix[0].length];

        for(int i = 0; i < eigenvalues.length; i++) {
            double[][] reduced_matrix = rowReducedEchelon(subtract(matrix,scalarMultiply(eigenvalues[i],newIdentityMatrix(matrix.length))));
            for(int j = reduced_matrix.length - 1; j >=0; j--) {
                if(reduced_matrix[j][j] == 0) result[i][j] = 1;
                else {
                    for(int k = 0; k < reduced_matrix[j].length; k++)
                        result[i][j] -= result[i][k] * reduced_matrix[j][k];
                    result[i][j] /= reduced_matrix[j][j];
                }
            }
            result[i] = unitVector(result[i]);
        }
        return result;
    }

    /**
     * Returns the inverse matrix of the given matrix
     * @param matrix double[][]: The matrix to find the inverse of
     * @return double[][] OR null: The inverse of the given matrix unless no inverse matrix exists; singular
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double cofactor(double[][],int,int)
     * @uses double determinant(double[][])
     * @uses boolean isSquare(double[][])
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] inverse(final double[][] matrix) {
        double d = determinant(matrix);
        if (d==0) return null;
        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[j][i] = cofactor(matrix, i+1, j+1) / d;

        return result;
    }

    /**
     * Returns the magnitude of a vector
     * @param vector double[]: The vector to find the magnitude of
     * @return double: The magnitude of the vector
     */
    public static double magnitude(final double[] vector) {
        double result = 0;
        for(double x : vector) result += Math.pow(x,2);
        return Math.sqrt(result);
    }

    /**
     * Returns the matrix of cofactors for a given matrix
     * @param matrix double[][]: The matrix to find to matrix of cofactors of
     * @return double[][]: The matrix of cofactors for the given matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double cofactor(double[][])
     * @uses double determinant(double[][])
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean isSquare(double[][])
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] matrixOfCofactors(final double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = cofactor(matrix, i+1, j+1);

        return result;
    }

    /**
     * Returns the matrix of minors for the given matrix
     * @param matrix double[][]: The matrix to find the matrix of minors of
     * @return double[][]: The matrix of minors for the given matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses double determinant(double[][])
     * @uses boolean isSquare(double[][])
     * @uses double[][] minor(double[][],int,int)
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] matrixOfMinors(final double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = determinant(minor(matrix, i+1, j+1));

        return result;
    }

    /**
     * Returns the minor of the given matrix for the given row and column
     * @param matrix double[][]: The matrix to take the minor of
     * @param row int: The row to be removed. Must be 1 or greater [Starts at 1]
     * @param column int: The column to be removed. Must be 1 or greater [Starts at 1]
     * @return double[][]: The resulting matrix
     * @throws ArrayIndexOutOfBoundsException Thrown if the row or column numbers passed are outside of the bounds of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] minor(final double[][] matrix, int row, int column) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        if(matrix.length < row || matrix[0].length < column || column < 1 || row < 1)
            throw new ArrayIndexOutOfBoundsException("ArrayIndexOutOfBoundsException: The row and/or column to be removed from matrix is outside of the bounds of the matrix\n" +
                "Row: " + row + " Column: " + column + " Matrix size: " + matrix.length + "x" + matrix[0].length);

        double[][] result = new double[matrix.length - 1][matrix[0].length - 1];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result.length; j++)
                result[i][j] = matrix[i<row-1 ? i : i+1][j<column-1 ? j : j+1];

        return result;
    }

    /**
     * Multiplies two matrices together and returns the new matrix
     * @param matrix1 double[][]: First matrix being multiplied
     * @param matrix2 double[][]: Second matrix being multiplied
     * @return double[][]: Resulting matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown when m1 columns and m2 rows do not match
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] multiply(final double[][] matrix1, final double[][] matrix2) {
        if(!validMatrix(matrix1)) throw new InvalidMatrixException(matrix1);
        if(!validMatrix(matrix2)) throw new InvalidMatrixException(matrix2);
        if(matrix1[0].length != matrix2.length) throw new MatrixSizeMismatchException(matrix1, matrix2, "MatrixSizeMismatchException: Number of columns in matrix 1 must be equal to number of rows in matrix 2");
        double[][] result = new double[matrix1.length][matrix2[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                for(int k = 0; k < matrix1[0].length; k++)
                    result[i][j] += matrix1[i][k] * matrix2[k][j];

        return result;
    }

    /**
     * Returns the result of the given power of a matrix
     * @param matrix double[][]: The matrix to take a power of
     * @param exponent int: The power to which the matrix should be taken
     * @return double[][]: The result matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses double[][] clone(double[][])
     * @uses double[][] multiply(double[][], double[][])
     * @uses boolean validMatrix(double[][])
     * @apiNote If a number 1 or less is entered then the original matrix will be returned
     */
    public static double[][] pow(final double[][] matrix, int exponent) {
        double[][] result = clone(matrix);

        for(int i = 1; i < exponent; i++) result = multiply(result, matrix);

        return result;
    }

    /**
     * Returns the given matrix in Row Echelon form
     * @param matrix double[][]: The matrix to change to Row Echelon form
     * @return double[][]: The Row Echelon form of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @uses double[][] clone(double[][])
     * @uses double[][] scalarMultiply(double,double[][])
     * @uses double[][] subtract(double[][],double[][])
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] rowEchelon(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        double[][] result = clone(matrix);
        for(int i = 0; i < result.length; i++) {
            int[] leading_zeroes = new int[result.length];
            for(int j = 0,k; j < leading_zeroes.length; j++) {
                for(k = 0; k < result[0].length; k++) if(result[j][k] != 0) break;
                leading_zeroes[j] = k;
            }
            for(int j = 1; j < result.length; j++)
                for(int k = 1; k < result.length; k++)
                    if(leading_zeroes[k-1] > leading_zeroes[k]) {
                        int t = leading_zeroes[k];
                        double[] t_row = result[k];
                        leading_zeroes[k] = leading_zeroes[k-1];
                        result[k] = result[k-1];
                        leading_zeroes[k-1] = t;
                        result[k-1] = t_row;
                    }
            for(int j = i + 1; j < result.length && result[i][i] != 0; j++) {
                result[j] = subtract(result[j],scalarMultiply(result[j][i] / result[i][i], result[i]));
                for(int k = 0; k < result[j].length; k++) result[j][k] = (float)result[j][k];
            }
        }
        return result;
    }

    /**
     * Returns the given matrix in Row Reduced Echelon form
     * @param matrix double[][]: The matrix to change to Row Reduced Echelon form
     * @return double[][]: The Row Reduced Echelon form of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @uses double[][] clone(double[][])
     * @uses double[][] rowEchelon(double[][])
     * @uses double[][] subtract(double[][],double[][])
     * @uses double[][] scalarMultiply(double,double[][])
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] rowReducedEchelon(final double[][] matrix) {
        double[][] result = rowEchelon(matrix);
        for(int i = 0; i < result.length; i++) {
            double coefficient = result[i][i];
            for(int j = i; j < result[0].length; j++) result[i][j] /= coefficient == 0 ? 1 : coefficient;
        }

        for(int i = 0; i < result.length; i++)
            for(int j = i-1; j >= 0; j--)
                result[j] = subtract(result[j],scalarMultiply(result[j][i],result[i]));

        return result;
    }

    /**
     * Multiplies a matrix by a constant and returns the result
     * @param scalar double: Constant matrix is multiplied by
     * @param matrix double[][]: Matrix to be multiplied
     * @return double[][]: Result of multiplication
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] scalarMultiply(double scalar, final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = scalar * matrix[i][j];

        return result;
    }

    /**
     * Multiplies a vector by a constant and returns the result
     * @param scalar double: Scalar matrix is multiplied by
     * @param vector double[][]: Vector to be multiplied
     * @return double[]: Result of multiplication
     */
    public static double[] scalarMultiply(double scalar, final double[] vector) {
        double[] result = new double[vector.length];
        for(int i = 0; i < vector.length; i++) result[i] = scalar * vector[i];
        return result;
    }

    /**
     * Subtracts two matrices together
     * @param matrix1 double[][]: First matrix to be subtracted
     * @param matrix2 double[][]: Second matrix to be subtracted
     * @return double[][]: Result matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] subtract(final double[][] matrix1, final double[][] matrix2) {
        if(!validMatrix(matrix1)) throw new InvalidMatrixException(matrix1);
        if(!validMatrix(matrix2)) throw new InvalidMatrixException(matrix2);
        if(matrix1.length!=matrix2.length||matrix1[0].length!=matrix2[0].length) throw new MatrixSizeMismatchException(matrix1,matrix2, "MatrixSizeMismatchException: Matrices must be the same size to subtract");

        double[][] result = new double[matrix1.length][matrix1[0].length];

        for(int i = 0; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = matrix1[i][j] - matrix2[i][j];

        return result;
    }

    /**
     * Subtracts two vectors together
     * @param vector1 double[]: First vector to be subtracted
     * @param vector2 double[]: Second vector to be subtracted
     * @return double[][]: Result vector
     * @throws VectorSizeMismatchException Thrown if sizes of vectors do not match
     */
    public static double[] subtract(final double[] vector1, final double[] vector2) {
        if(vector1.length!=vector2.length) throw new VectorSizeMismatchException(vector1,vector2, "VectorSizeMismatchException: Vectors must be the same size to subtract");

        double[] result = new double[vector1.length];

        for(int i = 0; i < vector1.length; i++) result[i] = vector1[i] - vector2[i];

        return result;
    }

    /**
     * Returns the trace of a matrix
     * @param matrix double[][]: Matrix to use
     * @return double: The trace of the matrix
     * @throws NotSquareException Thrown when the matrix is not square
     * @throws InvalidMatrixException Thrown when matrix is invalid
     */
    public static double trace(double[][] matrix) {
        if(!isSquare(matrix)) throw new NotSquareException(matrix, "Not Square Exception: Matrix must be square to find the trace");

        double result = 0.0;
        for(int i = 0; i < matrix.length; i++) result += matrix[i][i];
        return result;
    }

    /**
     * Multiplies a matrix and a vector
     * @param matrix double[][]: Matrix to multiply
     * @param vector double[]: Vector to multiply
     * @return double[]: Result vector
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws VectorSizeMismatchException Thrown when number of columns in the Transformation Matrix does not match the dimension of the input vector
     */
    public double[] transform(final double[][] matrix, final double[] vector) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        if(matrix[0].length != vector.length)
            throw new VectorSizeMismatchException(matrix[0], vector, "VectorSizeMismatchException: Number of columns in Transformation Matrix must be equal to the number of elements in input vector");
        double[] result = new double[vector.length];

        for(int i = 0; i < result.length; i++)
            for (int k = 0; k < matrix[0].length; k++)
                result[i] += matrix[i][k] * vector[k];

        return result;
    }

    /**
     * Returns the transpose of the given matrix
     * @param matrix double[][]: The matrix to find the transpose of
     * @return double[][]: The transposed matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] transpose(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        double[][] result = clone(matrix);

        for(int i = 0 ; i < result.length; i++)
            for(int j = 0; j < result[0].length; j++)
                result[i][j] = result[j][i];

        return result;
    }

    /**
     * Returns the unit vector of a vector
     * @param vector double[]: The vector to find the unit vector of
     * @return double[]: The unit vector of the vector
     * @uses double magnitude(double[])
     */
    public static double[] unitVector(final double[] vector) {
        double[] result = new double[vector.length];
        double magnitude = magnitude(vector);
        for(int i = 0; i < vector.length; i++) result[i] = vector[i] / magnitude;
        return result;
    }

    /* Utilities */
    /**
     * Returns an unique copy of the given matrix
     * @param matrix double[][]: The matrix to make a copy of
     * @return double[][]: An unique copy of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static double[][] clone(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        double[][] result = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++)
            result[i] = matrix[i].clone();

        return result;
    }

    /**
     * Returns the height (the number of rows) of given matrix
     * @param matrix double[][]: Matrix to be checked
     * @return int: Height of matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     * @apiNote m.length returns the same result and is faster but will not catch InvalidMatrixException
     */
    public static int getHeight(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        return matrix.length;
    }

    /**
     * Returns the height (the number of columns) of a given matrix
     * @param matrix double[][]: Matrix to be check
     * @return int: Width of matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     * @apiNote m[0].length returns the same result and is faster but will not catch InvalidMatrixException
     */
    public static int getWidth(final double[][] matrix) {
        if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
        return matrix[0].length;
    }

    /**
     * Checks if the number of rows is equal to number of columns
     * @param matrix double[][]: Matrix to be checked
     * @return boolean
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @uses boolean validMatrix(double[][])
     */
    public static boolean isSquare(final double[][] matrix) {
        return matrix.length == getWidth(matrix);
    }

    /**
     * Returns an identity matrix of specified size
     * @param size int: size of new identity matrix
     * @return double[][]: New identity matrix
     */
    public static double[][] newIdentityMatrix(int size) {
        double[][] result = new double[size][size];
        for(int i = 0; i < size; i++) result[i][i] = 1.0;
        return result;
    }

    /**
     * Returns a string that describes an matrix
     * @param matrix double[][]: Matrix to be described
     * @return String
     */
    public static String toString(final double[][] matrix) {
        StringBuilder result = new StringBuilder();
        for(double[] r : matrix) {
            result.append("[");
            for(double x : r) result.append(String.format("%6.5s,", (x < 10000 && x > -10000) ? x : "ovrflw"));
            result.replace(result.lastIndexOf(","),result.lastIndexOf(",") + 1, "]\n");
        }
        result.deleteCharAt(result.length()-1);
        return result.toString();
    }

    /**
     * Returns a string that describes an vector
     * @param vector double[][]: Vector to be described
     * @return String
     */
    public static String toString(final double[] vector) {
        StringBuilder result = new StringBuilder();
        result.append("<");
        for(double x : vector) result.append(String.format("%6.5s,", (x < 10000 && x > -10000) ? x : "ovrflw"));
        result.replace(result.lastIndexOf(","),result.lastIndexOf(",") + 1, ">");

        return result.toString();
    }

    /**
     * Checks to make sure all rows in a matrix are the same size. If so returns true; else returns false
     * @param matrix double[][]: Matrix to be check
     * @return boolean
     */
    public static boolean validMatrix(final double[][] matrix) {
        int width = matrix[0].length;
        for(int i = 1; i < matrix.length; i++)
            if(width!=matrix[i].length) return false;
        return true;
    }

    /* Private Functions */
    /**
     * Finds the roots of a given polynomial equation
     * @param equation double[]: Equation to find the roots of
     * @return double[]: Roots that were found
     * @uses double[] p_polynomialDerivative(double[])
     * @uses double p_polynomialValue(double[],double)
     * @uses double[] p_removeRoot(double[],double)
     */
    private static double[] p_findRoots(double[] equation) {
        if(equation.length < 2) return new double[] {};
        if(equation.length == 2) return new double[] {-equation[0] / equation[1]};
        if(equation.length == 3) {
            if(Math.pow(equation[1],2) - 4 * equation[0] * equation[2] < 0) return new double[] {};
            else return new double[] {(-equation[1] + Math.sqrt(Math.pow(equation[1],2) - 4 * equation[0] * equation[2]))/2/equation[2],(-equation[1] - Math.sqrt(Math.pow(equation[1],2) - 4 * equation[0] * equation[2]))/2/equation[2]};
        }
        for(int i = 0; i < Integer.MAX_VALUE; i = i <= 0 ? -i + 1 : -i) {
            if(Math.min(p_polynomialValue(equation, i-1),p_polynomialValue(equation, i+1)) < 0 && Math.max(p_polynomialValue(equation, i-1),p_polynomialValue(equation, i+1)) > 0) {
                System.out.println();
                double x = i;
                for(int counter = 0; p_polynomialValue(equation,x) != 0 && counter < 1000; counter++)
                    x = x - p_polynomialValue(equation,x) / p_polynomialValue(p_polynomialDerivative(equation),x);
                double[] roots = p_findRoots(p_removeRoot(equation, x));
                double[] result = new double[roots.length + 1];
                result[0] = x;
                System.arraycopy(roots,0,result,1,roots.length);
                return result;
            }
        }
        return new double[] {};
    }

    /**
     * Returns the derivative of the given polynomial equation
     * @param equation double[]: The polynomial equation to find the derivative of
     * @return double[]: The derivative of the equation
     */
    private static double[] p_polynomialDerivative(double[] equation) {
        double[] result = new double[equation.length-1];
        for(int i = 0; i < result.length; i++) result[i] = (i+1) * equation[i+1];
        return result;
    }

    /**
     * Returns a polynomial solution to the determinant of a matrix
     * @param matrix double[][]: The matrix to find the determinant of
     * @param lambda double[][]: The matrix representing where unknown variables are
     * @return double: The polynomial solution to the determinant of the matrix
     * @throws InvalidMatrixException Thrown when matrix is invalid
     * @throws MatrixSizeMismatchException Thrown if sizes of matrices do not match
     * @throws NotSquareException Thrown when the matrix is not square
     * @uses boolean isSquare(double[][])
     * @uses boolean validMatrix(double[][])
     */
    private static double[] p_polynomialDeterminant(double[][] matrix, double[][] lambda) {
        if(!isSquare(matrix)) throw new NotSquareException(matrix);
        if(!isSquare(lambda)) throw new NotSquareException(lambda);
        if(matrix.length!=lambda.length) throw new MatrixSizeMismatchException(matrix,lambda,"MatrixSizeMismatchException: Matrices must be the same size");
        if(matrix.length==2) return new double[] {
                matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
                lambda[0][1] * matrix[1][0] + lambda[1][0] * matrix[0][1] - lambda[0][0] * matrix[1][1] - lambda[1][1] * matrix[0][0],
                lambda[0][0] * lambda[1][1] - lambda[0][1] * lambda[1][0]
        };

        double[] result = new double[matrix.length + 1];
        for(int i = 0; i < matrix.length; i++) {
            double[] eq = p_polynomialDeterminant(minor(matrix,1,i+1),minor(lambda,1,i+1));
            for(int j = 0; j < eq.length; j++)
                if(lambda[0][i] == 1) {
                    result[j] += matrix[0][i] * eq[j] * Math.pow(-1,i);
                    result[j+1] += -eq[j] * Math.pow(-1,i);
                } else result[j] += matrix[0][i] * eq[j] * Math.pow(-1,i);
        }

        return result;
    }

    /**
     * Returns the value of a polynomial equation for a given x
     * @param equation double[]: The equation of the polynomial
     * @param x double: The value for x in the polynomial
     * @return double: The value of the equation given x
     */
    private static double p_polynomialValue(double[] equation, double x) {
        double result = 0;
        for(int i = 0; i < equation.length; i++) result += equation[i] * Math.pow(x,i);
        return result;
    }

    /**
     * Removes a given root from the given polynomial equation
     * @param equation double[]: The equation to remove the root from
     * @param root double: The value for x in the polynomial
     * @return double: The new equation after removing the root
     */
    private static double[] p_removeRoot(double[] equation, double root) {
        double[] result = new double[equation.length - 1];
        double remainder = 0;
        for(int i = equation.length-1; i > 0; i--) {
            result[i-1] = equation[i] - remainder * root;
            remainder = result[i-1];
        }
        return result;
    }

    /* ***************
        INNER CLASSES
       *************** */
    public static class LinearTransformation {
        double[][] transformationMatrix;

        /**
         * Creates a new Linear Transformation instance
         * @param transformation_matrix double[][]: Transformation Matrix for linear Transformation
         * @throws InvalidMatrixException Thrown when matrix is invalid
         * @throws NotSquareException Thrown when the matrix is not square
         */
        public LinearTransformation(double[][] transformation_matrix) {
            if(!isSquare(transformation_matrix)) throw new NotSquareException(transformation_matrix);
            this.transformationMatrix = transformation_matrix;
        }

        /**
         * Transforms a given vector according to the set Linear Transformation
         * @param vector double[]: Vector to transform
         * @return double[]: Transformed vector
         * @throws VectorSizeMismatchException Thrown when number of columns in the Transformation Matrix does not match the dimension of the input vector
         */
        public double[] transform(double[] vector) {
            if(this.transformationMatrix[0].length != vector.length)
                throw new VectorSizeMismatchException(this.transformationMatrix[0], vector, "VectorSizeMismatchException: Number of columns in Transformation Matrix must be equal to the number of elements in input vector");

            for(int i = 0; i < vector.length; i++)
                for (int k = 0; k < this.transformationMatrix[0].length; k++)
                    vector[i] += this.transformationMatrix[i][k] * vector[k];

            return vector;
        }

        /**
         * Transforms a given matrix according to the set Linear Transformation
         * @param matrix double[][]: Matrix to transform
         * @return double[][]: Transformed matrix
         * @throws MatrixSizeMismatchException Thrown when number of columns in the Transformation Matrix does not match the number of rows of the input matrix
         */
        public double[][] transform(double[][] matrix) {
            if(!validMatrix(matrix)) throw new InvalidMatrixException(matrix);
            if(this.transformationMatrix[0].length != matrix.length)
                throw new MatrixSizeMismatchException(this.transformationMatrix, matrix, "MatrixSizeMismatchException: Number of columns in Transformation Matrix must be equal to number of rows in input matrix");

            for(int i = 0; i < matrix.length; i++)
                for(int j = 0; j < matrix[0].length; j++)
                    for(int k = 0; k < this.transformationMatrix[0].length; k++)
                        matrix[i][j] += this.transformationMatrix[i][k] * matrix[k][j];

            return matrix;
        }
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

        private MatrixRuntimeException(double[][] matrix, String message) {
            super(message);
            this.matrix = matrix;
        }
    }

    /**
     * Occurs when the matrix has an invalid format
     * @throws InvalidMatrixException Thrown when matrix is invalid
     */
    public static class InvalidMatrixException extends MatrixRuntimeException {
        public InvalidMatrixException(double[][] matrix, String message) {
            super(matrix,message);
        }

        public InvalidMatrixException(double[][] matrix) {
            super(matrix, "InvalidMatrixException: Not all rows are of the same size");
        }
    }

    /**
     * Occurs when a matrix is expected to be square and is not
     * @throws NotSquareException Thrown when the matrix is not square
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
        public MatrixSizeMismatchException(double[][] matrix1, double[][] matrix2, String message) {
            super(matrix1, message);
            this.matrix2 = matrix2;
        }

        public MatrixSizeMismatchException(double[][] matrix1, double[][] matrix2) {
            super(matrix1, "Matrix Size Mismatch Exception: Matrices must have compatible sizes to perform operation");
            this.matrix2 = matrix2;
        }
    }

    /* Vector Exceptions */
    /**
     * Contains a copy of the vector that caused the exception to occur for error checking purposes
     */
    private static class VectorRuntimeException extends RuntimeException {
        double[] vector;

        private VectorRuntimeException(double[] vector, String message) {
            super(message);
            this.vector = vector;
        }
    }

    /**
     * Occurs when two vectors are expected to have compatible sizes
     * Stores both vectors
     */
    public static class VectorSizeMismatchException extends VectorRuntimeException {
        double[] vector2;

        public VectorSizeMismatchException(double[] vector1, double[] vector2, String message) {
            super(vector1, message);
            this.vector2 = vector2;
        }

        public VectorSizeMismatchException(double[] vector1, double[] vector2) {
            super(vector1, "VectorSizeMismatchException: Vectors must be the same size to perform operation");
            this.vector2 = vector2;
        }
    }
}