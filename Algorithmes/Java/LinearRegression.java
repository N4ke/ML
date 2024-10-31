import java.util.Arrays;


public class LinearRegression {

    private double w;
    private double b;
    private final double alpha;
    private final double[] x_cor;
    private final double[] y_cor;


    public LinearRegression(double[] x, double[] y, double learningRate) {
        this.w = 0;
        this.b = 0;
        this.alpha = learningRate;
        this.x_cor = Arrays.copyOf(x, x.length);
        this.y_cor = Arrays.copyOf(y, y.length);
    }


    private double linFunc(double x) {
        return w * x + b;
    }


    private double MSE() {
        double sum = 0;
        for (int i = 0; i < x_cor.length; i++) {
            double error = linFunc(x_cor[i]) - y_cor[i];
            sum += error * error;
        }
        return sum / (2 * x_cor.length);
    }


    private void gradientDescent() {
        double sumErrorW = 0;
        double sumErrorB = 0;

        for (int i = 0; i < x_cor.length; i++) {
            double error = linFunc(x_cor[i]) - y_cor[i];
            sumErrorW += error * x_cor[i];
            sumErrorB += error;
        }

        double temp_w = w - alpha * (sumErrorW / x_cor.length);
        double temp_b = b - alpha * (sumErrorB / x_cor.length);

        w = temp_w;
        b = temp_b;
    }


    public void fit(double accuracy, int maxIters) {
        int iters = 0;
        while (MSE() > accuracy && iters < maxIters) {
            gradientDescent();
            iters++;
        }

        System.out.println("w: " + w + ", b: " + b);
    }
}
