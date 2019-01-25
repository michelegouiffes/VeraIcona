
//#include "stdafx.h"
//#define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#include <Windows.h>
#include <iostream>  
#include <fstream>  
#include <vector>
#include <ctime>

#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include <dlib/opencv.h>   
#include <dlib/image_processing/frontal_face_detector.h>    
#include <dlib/image_processing/render_face_detections.h>    
#include <dlib/image_processing.h>    
#include <dlib/gui_widgets.h>  

#define OSC_HOST_BIG_ENDIAN
#include "include/osc/OscOutboundPacketStream.h"
#include "include/ip/UdpSocket.h"

/*#include "OscOutboundPacketStream.h"
#include "NetworkingUtils.h"
#include "UdpTransmitPort.h"
//#include "Transmit.h"
#define IP_ISADORA  "localhost"
#define IP_MTU_SIZE 1536
#define PORT 1234	*/


using namespace std;
using namespace cv;

/**Global Variables**/
unsigned int  AAtime = 0, BBtime = 0;
int flag_detection =0;

#define SEND_ADDRESS "127.0.0.1"
#define PORT 9000
#define OUTPUT_BUFFER_SIZE 1024
#define VOLUME_MIN 0
#define VOLUME_MAX 0.85
#define TIME_SWAP 5


namespace osc {
	void sendInt(float value, int piste) {

		UdpTransmitSocket transmitSocket(IpEndpointName(SEND_ADDRESS, PORT));

		char buffer[OUTPUT_BUFFER_SIZE];
		osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE);

		if (piste == 1) {
			p << osc::BeginBundleImmediate
				<< osc::BeginMessage("/volume1")
				<<(float)value << osc::EndMessage
				<< osc::EndBundle;
		}
		if (piste == 4) {
			p << osc::BeginBundleImmediate
				<< osc::BeginMessage("/volume4")
				<<(float)value << osc::EndMessage
				<< osc::EndBundle;
		}
		/*if (piste == 3) {
			p << osc::BeginBundleImmediate
				<< osc::BeginMessage("/volume3")
				<< (float)value << osc::EndMessage
				
				<< osc::EndBundle;
		}*/

		/*p << osc::BeginBundleImmediate
			<< osc::BeginMessage("/test1")
			<< true << 23 << (float)3.1415 << "hello" << osc::EndMessage
			<< osc::BeginMessage("/test2")
			<< true << 24 << (float)10.8 << "world" << osc::EndMessage
			<< osc::EndBundle;*/

		transmitSocket.Send(p.Data(), p.Size());
	}
	void sendcmd() {

		UdpTransmitSocket transmitSocket(IpEndpointName(SEND_ADDRESS, PORT));

		char buffer[OUTPUT_BUFFER_SIZE];
		osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE);

		
			p << osc::BeginBundleImmediate
				<< osc::BeginMessage("/live/play/clip")
				<< (float)1.1 << osc::EndMessage
				<< osc::EndBundle;
		
		transmitSocket.Send(p.Data(), p.Size());
	}


}

//sample functions
void recordVid(const string output)
{
	VideoCapture vcap(0);
	if (!vcap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	int frame_width = static_cast<int> (vcap.get(CV_CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int> (vcap.get(CV_CAP_PROP_FRAME_HEIGHT));
	VideoWriter video(output + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);

	for (;;) {

		Mat frame;
		vcap >> frame;
		video.write(frame);
		imshow("Frame", frame);
		char c = (char)waitKey(33);
		if (c == 27) break;
	}
}

void readFromVideo(const string input)
{
	bool hasEnded = false;
	while (!hasEnded)
	{
		VideoCapture CCap(input);
		if (CCap.isOpened())
		{
			Mat frame;
			while (true)
			{
				CCap >> frame;
				if (frame.empty()) break;
				resize(frame, frame, Size(480, 270));
				imshow("Input Video", frame);
				if (waitKey(3) == 27)
				{
					hasEnded = true; break;
				}
			}
			frame.release();
		}
		CCap.release();
	}
}

void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri);
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri);
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2);
std::vector<cv::Point2f> getFacialLandmarks(dlib::frontal_face_detector& detector, dlib::shape_predictor& pose_model, dlib::cv_image<dlib::bgr_pixel>& imgVideo, cv::Point2i& min, cv::Point2i& max);
cv::Mat swapfaces(cv::Mat& img1, cv::Mat& img2, cv::Mat& img1Warped, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);
void grayPreprocess(cv::Mat& CCframe, cv::Mat& IIframe);
void addSaltPepperNoise(cv::Mat& source, float pa, float pb, RNG& rng, int param);



//final algo from(01/09/2016) without 2nd view camera integrated
void faceswap2(const string portrait_input, const int index = 0, bool isGray = false, bool hasNoise = false, const string output = "output")
{


	//init dlib detector
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model; //don't forget this file when exporting the project

	bool hasEnded = false;


	cv::RNG rng; //random gene

	
	VideoCapture IIcap(index); //camera
	VideoCapture CCcap(portrait_input); //portrait stream

	Size frameSZ = Size(static_cast<int>(IIcap.get(CV_CAP_PROP_FRAME_WIDTH)), static_cast<int>(IIcap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	cv::namedWindow("image", WINDOW_NORMAL);
	cv::resizeWindow("image", 1920, 1080);
	cv::moveWindow("image", -20, -50);

	CCcap.release();
	IIcap.release();

	Mat CCframe, CCframe1, IIframe, RRframe, RRframefinal; //portrait frame, cam frame, result from face swap, sole displayed frame, last result frame
	Mat CCframeWarped, CCtempframe;
	Mat CCface, IIface, RRface; //store croped frames

								//vectors where to store 68 facial landmarks
	std::vector<cv::Point2f> CCpoints, IIpoints;
	/*create a face vector*/
	std::vector<cv::Rect> detectedRect(2); //detectedRect[0] for faces on the portrait, detectedRect[1] on the camera
	cv::Rect CCRect0, IIRect0;
	while (1)
	{
		osc::sendInt((float)VOLUME_MIN, 1);
		osc::sendInt((float)VOLUME_MAX, 4);
		VideoCapture IIcap(index); //camera
		VideoCapture CCcap(portrait_input); //portrait stream
		clock_t timer;
		
		//osc::sendInt((float)VOLUME_MIN, 3);
		//osc::sendcmd();

		if (CCcap.isOpened() && IIcap.isOpened())
		{
			osc::sendInt((float)VOLUME_MIN, 1);
			osc::sendInt((float)VOLUME_MAX, 4);
			bool swapping = false;
			bool blurring = false;
			bool isStopped = false;

			const double t_swap = TIME_SWAP; //time in seconds

			timer = clock(); //init the timer

			bool okCCface = false, okIIface = false; //security for the faces both in position and in dimension (dimension is not implemented yet)

			const int offset = 20; //offset pixels to the frame border
			double alpha = 0; //alpha for fading/blending
			double alpha_inc = 0.2; //increment for alph
			hasEnded = 0;
			flag_detection = 0;
			//clock_t tv3;
			while (!hasEnded)
			{

				/*get frames from streams*/
				CCcap >> CCframe1; //img1
				CCframe1.copyTo(CCframe);
				IIcap >> IIframe; //img2

				/*if (flag_detection == 0) {
					osc::sendInt((float)VOLUME_MIN, 1);
					osc::sendInt((float)VOLUME_MAX, 4);
				}
		
				else {
					
					osc::sendInt((float)VOLUME_MIN, 4);
					osc::sendInt((float)VOLUME_MAX, 1);
				}*/
				
				if (!CCframe1.empty() && !IIframe.empty()) {
					resize(CCframe1, CCframe, frameSZ);
					//equalizeHist(CCframe1, CCframe);
					//equalizeHist(MMframe1, MMframe);
					//create gray images
					if (isGray)cvtColor(CCframe, CCframe, CV_BGR2GRAY);
					if (CCframe.channels() == 1)
					{
						grayPreprocess(CCframe, IIframe);
					}
					
					//if the portrait video or the mosaic one are empty, break the display loop
					if (CCframe.empty()) hasEnded = true;
					//if (CCframe.empty() ) break;
					
					CCframeWarped = IIframe.clone();
					RRframe = CCframe.clone();	//by default the result frame of the face swap is the portrait frame
												//RRframefinal = MMframe.clone();	//if no face is detected the displayed frame is the mosaic one
					RRframefinal = CCframe.clone();	//if no face is detected the displayed frame is the mosaic one
					
					/*create dlib frames for landmark detection*/
					dlib::cv_image<dlib::bgr_pixel> CCdlibframe(CCframe), IIdlibframe(IIframe);

					/*CC processing (video)*/
					//init min and max for minimal rectangle creation
					cv::Point2i CCmin(CCframe.size().width, CCframe.size().height);
					cv::Point2i CCmax(0, 0);

					/*face detection*/
					CCpoints = getFacialLandmarks(detector, pose_model, CCdlibframe, CCmin, CCmax);
					
					/*secure the face detection*/
					if (CCmin.x > offset && CCmin.y > offset && CCmax.x < CCframe.size().width - offset && CCmax.y < CCframe.size().height - offset)
					{
						CCRect0 = cv::Rect(CCmin, CCmax);//cv::rectangle(CCframe, CCRect0, cv::Scalar(0, 255, 0));
						detectedRect[0] = CCRect0;
						okCCface = true;
						
					}
					else
					{
						okCCface = false;
						
					}

					/*II processing (camera feed)*/
					//init min and max for minimal rectangle creation
					cv::Point2i IImin(IIframe.size().width, IIframe.size().height);
					cv::Point2i IImax(0, 0);
					/*face detection*/
					IIpoints = getFacialLandmarks(detector, pose_model, IIdlibframe, IImin, IImax);

					/*secure the face detection*/
					if (IImin.x > offset && IImin.y > offset && IImax.x < IIframe.size().width - offset && IImax.y < IIframe.size().height - offset)
					{
						IIRect0 = cv::Rect(IImin, IImax);//cv::rectangle(IIframe, IIRect0, cv::Scalar(0, 255, 0));
						detectedRect[1] = IIRect0;
						okIIface = true;
					}
					else
					{
						okIIface = false;
						
					}

					/*time to swap*/
					/*if (!swapping && t_swap < ((clock() - timer) / (double)CLOCKS_PER_SEC))
					{
						swapping = true;
						timer = clock(); //init the timer
					}*/


					//if (swapping)
					{
						
						if (okCCface && okIIface)
						{
							if (flag_detection == 0) {
								//tv3 = clock();
								//osc::sendInt((float)VOLUME_MAX, 1);
								//osc::sendInt((float)VOLUME_MIN, 4);
								//osc::sendInt((float)VOLUME_MAX, 3);
								osc::sendInt((float)VOLUME_MAX, 1);
								osc::sendInt((float)VOLUME_MIN, 4);
								
								flag_detection = 1;
							}
							//RunSend_int("localhost", 1234, 1, 1);
							CCtempframe = IIframe.clone();
							if (hasNoise) //set noise
							{
								IIface = IIframe(IIRect0);
								IIface.copyTo(CCtempframe(IIRect0));
							}

							//swap morphed face
							RRframe = swapfaces(CCtempframe, CCframe, CCframeWarped, IIpoints, CCpoints);
							//RRframefinal = RRframe;
							//alpha blending with rrframe and the displayed result
							alpha = 1;
							if (alpha > 1)
								alpha = 1;
							else if (alpha <= 1 - alpha_inc)
								alpha += alpha_inc;

							if (alpha <= 1) RRframefinal = alpha*RRframe + (1 - alpha)*RRframefinal;
						}
						else
						{

							if (flag_detection == 1) {
								flag_detection = 0;
								clock_t t0, t1;
								float val4 = VOLUME_MIN;
								//float val4 = VOLUME_MAX;
								float val1 = VOLUME_MAX;
								//float val1 = VOLUME_MIN;

								t0 = clock();
								osc::sendInt((float)(val1), 4);
								osc::sendInt((float)(val4), 1);
								t1 = clock();
								
								double pas = (t1 - t0) / (double)CLOCKS_PER_SEC;
								int n = 2 / pas;
								float dval = val1 / n;
								for (int tt = 0; tt < n; tt++) {
										osc::sendInt((float)(val4 + tt*dval), 4);
										osc::sendInt((float)(val1 - tt*dval), 1);
									}
								//if (0.4 >((clock() - tv3) / (double)CLOCKS_PER_SEC))
								//osc::sendInt((float)VOLUME_MIN, 3);
								osc::sendInt((float)VOLUME_MAX, 4);
								osc::sendInt((float)VOLUME_MIN, 1);
								
							}
						}
					
					}
					//addSaltPepperNoise(cloned, 0.1, 0.05, rng, 10);
					addSaltPepperNoise(RRframefinal, 1, 1, rng, 10);
					//osc::sendInt(0, 1);
					cvtColor(RRframefinal, RRframefinal, CV_GRAY2BGR);

					if (isGray) //transform the final result to gray
					{
						cvtColor(RRframe, RRframe, CV_BGR2GRAY); grayPreprocess(RRframe, IIframe);
					}

					cv::flip(RRframefinal, RRframefinal, 1);

					imshow("image", RRframefinal);
					

					switch (waitKey(1))
					{
					case 27:		//ESC to close the program
						hasEnded = true; isStopped = true; break;
					case 32:		//space
						swapping = !swapping; break;
					}
					if (isStopped) break;
					/*if (swapping)
					{
						timer = clock();
					}*/

				}
				else {
					hasEnded = true;
					//flag_detection = 0;
					//osc::sendInt((float)VOLUME_MIN, 1);
					//osc::sendInt((float)VOLUME_MAX, 4);
				}
				
			}
			
		}
			
		else
		{
			
			cout << "Camera or video stream Error while opening" << endl;
			
			break;
			//waitKey(0);
		}
		//timer = clock(); //timer re-init

		CCcap.release();
		IIcap.release();
		


	}
	CCframe.release();
	CCframe1.release();
	CCframeWarped.release();
	CCtempframe.release();
	IIframe.release();
	RRframe.release();
	RRframefinal.release();
	CCface.release();
	IIface.release();
	RRface.release();


}
//test salt&pepper
void test()
{
	RNG rng;
	cv::Mat frame = cv::imread("lena.jpg");
	cv::Mat cloned;
	cv::Rect rect(0, 0, frame.rows / 2, frame.cols / 2);
	cloned = frame(rect);
	addSaltPepperNoise(cloned, 0.1, 0.05, rng,10);
	cvtColor(cloned, cloned, CV_GRAY2BGR);
	cloned.copyTo(frame(rect));
	cv::imshow("original", frame); cv::imshow("cloned", cloned); cv::waitKey(0);
}

int main()
{
	cout << "Hello World" << endl;
	bool isGray = true, hasNoise = true;
	try
	{
		//faceswap2("estelleNBlarge2fade.mp4", 0, isGray, hasNoise, "estelleNBlarge2fade");
		faceswap2("testboucle.mp4", 0, isGray, hasNoise, "testboucle");
		
	}
	catch (exception& e)
	{
		cerr << e.what() << endl;
	}
	return 0;
}


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri) {

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Insert points into subdiv
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}

}

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// Offset points by left top corner of the respective rectangles
	vector<Point2f> t1Rect, t2Rect;
	vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

	}

	// Get mask by filling triangle
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;


}
//get 68 facial landmarks and pushes them into a vector<cv::Point2f>
std::vector<cv::Point2f> getFacialLandmarks(dlib::frontal_face_detector& detector, dlib::shape_predictor& pose_model, dlib::cv_image<dlib::bgr_pixel>& imgVideo, cv::Point2i& min, cv::Point2i& max)
{
	// Detect faces 
	std::vector<dlib::rectangle> faces = detector(imgVideo);
	// Find the pose of each face.
	std::vector<dlib::full_object_detection> shapes;
	std::vector<cv::Point2f> points;
	for (unsigned long i = 0; i < faces.size(); ++i)
		shapes.push_back(pose_model(imgVideo, faces[i]));
	if (!shapes.empty())
	{
		for (int i = 0; i < 68; i++)
		{
			points.push_back(cv::Point2f(shapes[0].part(i).x(), shapes[0].part(i).y()));

			if (shapes[0].part(i).x() < min.x) min.x = shapes[0].part(i).x();
			else if (shapes[0].part(i).x() > max.x) max.x = shapes[0].part(i).x();
			if (shapes[0].part(i).y() < min.y) min.y = shapes[0].part(i).y();
			else if (shapes[0].part(i).y() > max.y) max.y = shapes[0].part(i).y();
		}
	}
	else
	{
		min = cv::Point2i(0, 0); max = cv::Point2i(0, 0);
	}
	return points;
}
//warp cam face to portrait face using delaunay
cv::Mat swapfaces(cv::Mat& img1, cv::Mat& img2, cv::Mat& img1Warped, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2)
{
	//convert Mat to float data type
	img1.convertTo(img1, CV_32F);
	img1Warped.convertTo(img1Warped, CV_32F);


	// Find convex hull
	vector<Point2f> hull1;
	vector<Point2f> hull2;
	vector<int> hullIndex;

	convexHull(points2, hullIndex, false, false);

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);
		hull2.push_back(points2[hullIndex[i]]);
	}


	// Find delaunay triangulation for points on the convex hull
	vector< vector<int> > dt;
	Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
	calculateDelaunayTriangles(rect, hull2, dt);

	// Apply affine transformation to Delaunay triangles
	for (size_t i = 0; i < dt.size(); i++)
	{
		vector<Point2f> t1, t2;
		// Get points for img1, img2 corresponding to the triangles
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);
		}

		warpTriangle(img1, img1Warped, t1, t2);

	}

	// Calculate mask
	vector<Point> hull8U;
	for (int i = 0; i < hull2.size(); i++)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}

	Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));

	// Clone seamlessly.
	Rect r = boundingRect(hull2);
	Point center = (r.tl() + r.br()) / 2;

	Mat output;
	img1Warped.convertTo(img1Warped, CV_8UC3);
	seamlessClone(img1Warped, img2, mask, center, output, NORMAL_CLONE);

	return output;
}
//get 3 grayscale channels for CCframes and IIframes
void grayPreprocess(cv::Mat& CCframe, cv::Mat& IIframe)
{
	cvtColor(IIframe, IIframe, CV_BGR2GRAY);

	vector<Mat> CCchannels, IIchannels;
	CCchannels.push_back(CCframe);
	CCchannels.push_back(CCframe);
	CCchannels.push_back(CCframe);
	merge(CCchannels, CCframe);

	IIchannels.push_back(IIframe);
	IIchannels.push_back(IIframe);
	IIchannels.push_back(IIframe);
	merge(IIchannels, IIframe);
}
//add some noise to the source frame
void addSaltPepperNoise(cv::Mat& source, float pa, float pb, RNG& rng, int param)
{
	cvtColor(source, source, CV_BGR2GRAY);
	int amountA = source.rows*source.cols*pa;
	int amountB = source.rows*source.cols*pb;
	for (int i = 0; i < amountA; ++i)
	{
		int pixel = source.at<uchar>(i);
		pixel += rng.uniform(-param, param);
		if (pixel > 255) pixel = 255;
		if (pixel < 0) pixel = 0;
		if (pixel > 255) cout << pixel << endl;
		if (pixel <0) cout << pixel << endl;
		source.at<uchar>(i) = pixel;
	}
	for (int i = 0; i < amountB; ++i)
	{
		int pixel = source.at<uchar>(i);
		if (pixel > 255) pixel = 255;
		if (pixel < 0) pixel = 0;
		if (pixel > 255) cout << pixel << endl;
		if (pixel <0) cout << pixel << endl;
		source.at<uchar>(i) = pixel;
	}
}

