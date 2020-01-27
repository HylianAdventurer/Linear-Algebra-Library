import java.util.Objects;

public class Testing extends LinearAlgebra{
    public static void main(String[] args) {
        double[][] m = {
                {0,1,0,1,0},
                {1,0,1,1,1},
                {0,1,0,0,1},
                {1,1,0,0,0},
                {0,1,1,0,0}
        };
        double[][] m2 = {{1,-2},{-2,0}};
        double[][] m3 = {
                {1,2,1},
                {6,-1,0},
                {-1,-2,-1}
        };

        //System.out.println(toString(rowEchelon(m3)));
        //System.out.println(toString(rowReducedEchelon(m3)));
        for(double x : eigenvalues(m3)) System.out.print(x + " ");
        System.out.println();

        for(double[] v : eigenvectors(m3)) System.out.println(toString(v));

//        double[] arr = new double[] {6,4,3,7,3,2,2,9,1,5};
//
//        int n = arr.length;
//        for(int i = 0; i < n; i++) {
//            for(double x : arr) System.out.print(x + " ");
//            System.out.println();
//            double min = arr[i];
//            int min_pos = i;
//            for(int j = i+1; j < n; j++)
//                if(arr[j] < min) {
//                    min = arr[j];
//                    min_pos = j;
//                }
//            if(i != 0) {
//                if(min == arr[i-1]) {
//                    double t = arr[--n];
//                    arr[n] = min;
//                    arr[min_pos] = t;
//                    i--;
//                } else {
//                    double t = arr[i];
//                    arr[i] = min;
//                    arr[min_pos] = t;
//                }
//            } else {
//                double t = arr[i];
//                arr[i] = min;
//                arr[min_pos] = t;
//            }
//        }
//        double result[] = new double[n];
//        System.arraycopy(arr,0,result,0,n);
//        for(double x : result) System.out.print(x + " ");
//        System.out.println();
    }
}
