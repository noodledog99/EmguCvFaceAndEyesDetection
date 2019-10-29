using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace TestEmguCVDnn
{
    public partial class Form1 : Form
    {
        private VideoCapture camera;
        //private Image<Bgr, byte> frame;
        private int resolutionX = 640;
        private int resolutionY = 480;
        private int cameraIndex = 0;

        private int detectionSize = 300;
        private float xRate = 1.0f;
        private float yRate = 1.0f;
        private Net net;
        private string protoPath = @"E:\Programing\FaceDetection\DemoEmguCVDnn\TestEmguCVDnn\Models\deploy.prototxt";
        private string caffemodelPath = @"E:\Programing\FaceDetection\DemoEmguCVDnn\TestEmguCVDnn\Models\res10_300x300_ssd_iter_140000.caffemodel";
        private CascadeClassifier eyes_detect;

        private string text;
        public Form1()
        {
            InitializeComponent();
            try
            {
                eyes_detect = new CascadeClassifier(@"E:\Programing\FaceDetection\DemoEmguCVDnn\TestEmguCVDnn\haarcascade_eye.xml");

                xRate = resolutionX / (float)detectionSize;
                yRate = resolutionY / (float)detectionSize;
                net = DnnInvoke.ReadNetFromCaffe(protoPath, caffemodelPath);

                this.Width = resolutionX;
                this.Height = resolutionY;
                camera = new VideoCapture(cameraIndex);
                camera.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, resolutionX);
                camera.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, resolutionY);

            }
            catch (NullReferenceException ex)
            {
                throw;
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {

            using (Image<Bgr, byte> frame = camera.QueryFrame().ToImage<Bgr, byte>())
            {
                if (frame != null)
                {
                    Image<Gray, Byte> grayImage = frame.Convert<Gray, byte>();
                    var StoreEyes = eyes_detect.DetectMultiScale(grayImage);

                    //CvInvoke.Flip(frame, frame, Emgu.CV.CvEnum.FlipType.Horizontal);
                    Mat blobs = DnnInvoke.BlobFromImage(frame, 1.0, new System.Drawing.Size(detectionSize, detectionSize));
                    net.SetInput(blobs);
                    Mat detections = net.Forward();

                    float[,,,] detectionsArrayInFloats = detections.GetData() as float[,,,];

                    for (int i = 0; i < detectionsArrayInFloats.GetLength(2); i++)
                    {
                        if (Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 2], CultureInfo.InvariantCulture) > 0.4)
                        {
                            float Xstart = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 3],
                                CultureInfo.InvariantCulture) * detectionSize * xRate;
                            float Ystart = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 4],
                                CultureInfo.InvariantCulture) * detectionSize * yRate;
                            float Xend = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 5],
                                CultureInfo.InvariantCulture) * detectionSize * xRate;
                            float Yend = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 6],
                                CultureInfo.InvariantCulture) * detectionSize * yRate;

                            System.Drawing.Rectangle rect = new System.Drawing.Rectangle
                            {
                                X = (int)Xstart,
                                Y = (int)Ystart,
                                Height = (int)(Yend - Ystart),
                                Width = (int)(Xend - Xstart)
                            };
                            frame.Draw(rect, new Bgr(0, 255, 0), 2);
                            foreach (var hEye in StoreEyes)
                            {
                                //frame.Draw(hEye, new Bgr(0, double.MaxValue, 0), 3);
                                var avgEyes = StoreEyes?.Average(it => (it.Right + it.Left) / 2) ?? 0;
                                var turnLeft = (Xstart + Xend) * 0.52;
                                var turnRight = (Xstart + Xend) * 0.48;
                                Console.WriteLine($"Xstart in Eyes = {Xstart}");
                                Console.WriteLine($"Ystart in Eyes = {Ystart}");
                                Console.WriteLine($"turnLeft = {turnLeft}");
                                Console.WriteLine($"turnRight = {turnRight}");
                                Console.WriteLine($"avgEyes = {avgEyes}");
                                if (avgEyes > turnLeft)
                                {
                                    text = "Turn Left";
                                }
                                else if (avgEyes < turnRight)
                                {
                                    text = "Turn Right";
                                }
                                else
                                {
                                    text = "Default";
                                }
                            }
                            CvInvoke.PutText(frame, text, new Point(rect.X - 2, rect.Y - 2), Emgu.CV.CvEnum.FontFace.HersheySimplex, 1.0, new Bgr(Color.Red).MCvScalar);
                        }
                    }
                    imageBox1.Image = frame;
                }
            }
        }
    }
}
