package edu.upvictoria.fpoo;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.Arrays;

public class MeanShift {

    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public void run() {
        Mat m = Imgcodecs.imread("F:\\bloquesIMG\\legoblocks.jpg");
        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2RGB);

        Mat hsv_roi = new Mat(), mask = new Mat(), roi;
        //setup initial location of window
        Rect track_window = new Rect(300, 200, 100, 50);

        // setup initial location of window
        roi = new Mat(m, track_window);
        Imgproc.cvtColor(roi, hsv_roi, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv_roi, new Scalar(0, 60, 32), new Scalar(180, 255, 255), mask);

        MatOfFloat range = new MatOfFloat(0, 256);
        Mat roi_hist = new Mat();
        MatOfInt histSize = new MatOfInt(180);
        MatOfInt channels = new MatOfInt(0);
        Imgproc.calcHist(Arrays.asList(hsv_roi), channels, mask, roi_hist, histSize, range);
        Core.normalize(roi_hist, roi_hist, 0, 255, Core.NORM_MINMAX);

        TermCriteria term_crit = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1);

        for (int i = 0; i < 5; i++) {
            Mat hsv = new Mat() , dst = new Mat();
            Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);
            Imgproc.calcBackProject(Arrays.asList(hsv), channels, roi_hist, dst, range, 1);

            // apply meanshift to get the new location
            Video.meanShift(dst, track_window, term_crit);

            // Obtener la nueva región rastreada
            Rect new_track_window = track_window.clone();
            Mat tracked_roi = new Mat(m, new_track_window);

            // Calcular el color medio dentro de la región rastreada
            Scalar meanColor = Core.mean(tracked_roi);

            System.out.println("Iteracion: " + i);
            System.out.println("Color promedio (RGB): R=" + meanColor.val[2] + ", G=" + meanColor.val[1] + ", B=" + meanColor.val[0]);

            // Dibuja el rectángulo de color medio
            Scalar rgbColor = new Scalar(meanColor.val[2], meanColor.val[1], meanColor.val[0]); // Swap R and B channels
            Imgproc.rectangle(m, new_track_window, rgbColor, 2);

            // Dibuja en la imagen
            Imgproc.rectangle(m, track_window, new Scalar(255, 0, 0), 2);
            HighGui.imshow("img2", m);

            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }
//        System.exit(0);
    }
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new MeanShift().run();
    }
}