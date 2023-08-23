package edu.upvictoria.fpoo;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.video.Video;
import org.opencv.core.Rect;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.Arrays;

public class App
{
    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void contornos(Mat m){
        int contador = 0;
        Mat gris = new Mat();
        Imgproc.cvtColor(m,gris,Imgproc.COLOR_RGB2GRAY);
        Mat thresh = new Mat();
        Imgproc.threshold(gris,thresh,0,255,Imgproc.THRESH_BINARY);
        ArrayList<MatOfPoint> contornos = new ArrayList<>();
        Mat hera = thresh;
        Imgproc.findContours(thresh,contornos,hera,1,2);
        for (int i = 0; i < contornos.size(); i++) {
            MatOfPoint contorno;
            contorno = contornos.get(i);
            if (Imgproc.contourArea(contorno) > 50) {
                Imgproc.drawContours(m,contornos,i,new Scalar(255,255,255));
                contador++;
            }
        }
        System.out.println("Hay "+contador+" en esta capa");
    }

    public static BufferedImage Mat2BufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1)
        {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b);
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public static void ventana(Mat draw, String titulo){
        BufferedImage bi = Mat2BufferedImage(draw);
        ImageIcon ico = new ImageIcon(bi);
        JFrame frame = new JFrame();
        JLabel lbl = new JLabel();
        lbl.setIcon(ico);
        frame.setTitle(titulo);
        frame.setSize(bi.getWidth()+2,bi.getHeight()+2);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(lbl);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    public static void kmeans(String dir, int k)
    {
        Mat m = Imgcodecs.imread(dir); // Matriz de la imagen dada
        Imgproc.resize(m,m,new Size(),0.5,0.5);
        m.convertTo(m,CvType.CV_32F); // Conversion a un Float32
        Mat labels = new Mat(), data = m.reshape(1,(int)m.total()), msal = new Mat();
        // kmeans
        Core.kmeans(data,k,labels,new TermCriteria(TermCriteria.EPS+TermCriteria.COUNT,10,1.0),5,Core.KMEANS_PP_CENTERS,msal);
        Mat draw = new Mat((int)m.total(),1, CvType.CV_32FC3); // Matriz a la que se le aplicaran los segmentos
        Mat colors = msal.reshape(3,k); // Lista de segmentos
        for (int i=0; i<k; i++) {
            Mat mask = new Mat(); // Una mascara por cada segmento
            Core.compare(labels, new Scalar(i), mask, Core.CMP_EQ); //Comparamos la mascara con los labels generados por el kmeans
            Mat col = colors.row(i);
            double d[] = col.get(0,0);
            //Sistema de colores BGR
            double cerca = (Math.abs(d[0]-d[1]) + Math.abs(d[0]-d[2])+Math.abs(d[1]-d[2]))/3;
            System.out.println("Marcara "+i+":"+cerca);
            if(cerca >= 35.1){
                draw.setTo(new Scalar(d[0],d[1],d[2]), mask);
                Mat colorMask = new Mat((int)m.total(),1, CvType.CV_32FC3);
                colorMask.setTo(new Scalar(d[0],d[1],d[2]), mask);
                colorMask = colorMask.reshape(3, m.rows());
                colorMask.convertTo(colorMask, CvType.CV_8U);
                // Realiza el tracking del Mean Shift en colorMask
                meanShiftTracking(colorMask, i);

                ventana(colorMask,"Capa "+i);
            }
        }
        draw = draw.reshape(3, m.rows());
        draw.convertTo(draw, CvType.CV_8U);
        ventana(draw,"KMeans Final");
    }

    public static void meanShiftTracking(Mat colorMask,  int iteration) {
        Rect track_window = new Rect(0, 0, colorMask.cols(), colorMask.rows()); // Initialize tracking window

        Mat hsv_roi = new Mat();
        Imgproc.cvtColor(colorMask, hsv_roi, Imgproc.COLOR_RGB2HSV);

        Mat mask = new Mat();
        Core.inRange(hsv_roi, new Scalar(0, 60, 32), new Scalar(180, 255, 255), mask);

        MatOfFloat range = new MatOfFloat(0, 256);
        Mat roi_hist = new Mat();
        MatOfInt histSize = new MatOfInt(180);
        MatOfInt channels = new MatOfInt(0);
        Imgproc.calcHist(Arrays.asList(hsv_roi), channels, mask, roi_hist, histSize, range);
        Core.normalize(roi_hist, roi_hist, 0, 255, Core.NORM_MINMAX);

        TermCriteria term_crit = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1);

        String[] colorNames = {"Rojo", "Verde", "Azul", "Amarillo", "Cyan", "Magenta"};
        Scalar[] predefinedColors = {
                new Scalar(0, 0, 255),    // Rojo
                new Scalar(0, 255, 0),    // Verde
                new Scalar(255, 0, 0),    // Azul
                new Scalar(0, 255, 255),  // Amarillo
                new Scalar(255, 255, 0),  // Cyan
                new Scalar(255, 0, 255)   // Magenta
        };

        for (int i = 0; i < 1; i++) {
            Mat hsv = new Mat(), dst = new Mat();
            Imgproc.cvtColor(colorMask, hsv, Imgproc.COLOR_RGB2HSV);
            Imgproc.calcBackProject(Arrays.asList(hsv), channels, roi_hist, dst, range, 1);

            Video.meanShift(dst, track_window, term_crit);

            // Dibujar el rectángulo de tracking en colorMask
            Scalar rgbColor = new Scalar(0, 0, 255); // color azul
            Imgproc.rectangle(colorMask, track_window, rgbColor, 2);

            // Imprime los valores RGB y encuentra el color más parecido
            Mat tracked_roi = new Mat(colorMask, track_window);
            Scalar meanColor = Core.mean(tracked_roi);
            System.out.println("Iteracion: " + iteration + ", RGB: R=" + //R = Red = Rojo
                    String.format("%.2f", meanColor.val[2]) + ", G=" + //G = Green = Verde
                    String.format("%.2f", meanColor.val[1]) + ", B=" + //B = Blue = Azul
                    String.format("%.2f", meanColor.val[0]));

            // Buscar el color predefinido más cercano en función de los valores RGB
            double minDistance = Double.MAX_VALUE;
            int closestColorIndex = -1;
            for (int j = 0; j < predefinedColors.length; j++) {
                double rDiff = predefinedColors[j].val[2] - meanColor.val[2];
                double gDiff = predefinedColors[j].val[1] - meanColor.val[1];
                double bDiff = predefinedColors[j].val[0] - meanColor.val[0];
                double distance = Math.sqrt(rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);

                if (distance < minDistance) {
                    minDistance = distance;
                    closestColorIndex = j;
                }
            }

            System.out.println("Color mas cercano: " + colorNames[closestColorIndex]);
        }
    }

    public static void main(String[] args){
        String archivo = "F:\\bloquesIMG\\legoblocks.jpg"; // Imagen que se segmentara
        int grupos = 4; // Numero de segmentos
        kmeans(archivo, grupos);
    }
}


// git checkout -b [nombre rama] | Crear rama nueva          --*----*----*--
// git checkout main | Regresar a main                      /                \
// git merge [nombre rama] | Se absorbe la rama indicada    *-----------------*