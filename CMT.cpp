#include "CMT.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "fast_test.cpp"





#include <time.h>

static int imax;
static int histSize = 4;
static int thr=10;
static vector<KeyPoint> keypoints_fg;
using namespace std;
//static Mat database;
namespace cmt {

void CMT::initialize(const Mat im_gray, const Rect rect)
{
    FILE_LOG(logDEBUG) << "CMT::initialize() call";
	
    //Remember initial size
    size_initial = rect.size();
	
    //Remember initial image
    im_prev = im_gray;

    //Compute center of rect
    Point2f center = Point2f(rect.x + rect.width/2.0, rect.y + rect.height/2.0);

    //Initialize rotated bounding box
    bb_rot = RotatedRect(center, size_initial, 0.0);

    //Initialize detector and descriptor
#if CV_MAJOR_VERSION > 2
    detector = cv::FastFeatureDetector::create();
	

    descriptor = cv::BRISK::create();
#else
    detector = FeatureDetector::create(str_detector);
    descriptor = DescriptorExtractor::create(str_descriptor);
#endif

    //Get initial keypoints in whole image and compute their descriptors
    vector<KeyPoint> keypoints;

	//Mat zero1(im_gray.rows, im_gray.cols, im_gray.depth(), Scalar(0));
	//Mat idk = zero1(rect);
	//im_gray(rect).convertTo(idk, idk.depth(), 1, 0);



	//detector->set("threshold", 10);
   // detector->detect(im_gray, keypoints);
	Mat hist;
	Mat coneruse = im_gray(rect);
	float range[] = { 0, 255 };
	const float* histRange = { range };
	double min1 = 0;
	double max1 = 0;
	//直方图统计
	bool uniform = true; bool accumulate = false;
	calcHist(&coneruse, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	minMaxIdx(hist, &min1, &max1);

	
	for (int i = 0; i<histSize; i++)
	{

		if (hist.at<float>(i) == max1)
			imax = i;

	}
	
	FAST_test<16>(im_gray, keypoints, thr, true, imax, histSize);
	
	//Mat img_keypoints_1;
	//drawKeypoints(im_gray, keypoints, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("kp", img_keypoints_1);
	//FAST(im_gray, keypoints, 20);
	
    //Divide keypoints into foreground and background keypoints according to selection
    
    vector<KeyPoint> keypoints_bg;

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint k = keypoints[i];
        Point2f pt = k.pt;

        if (pt.x > rect.x && pt.y > rect.y && pt.x < rect.br().x && pt.y < rect.br().y)
        {
            keypoints_fg.push_back(k);
        }

        else
        {
            keypoints_bg.push_back(k);
        }

    }

    //Create foreground classes
    vector<int> classes_fg;
    classes_fg.reserve(keypoints_fg.size());
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        classes_fg.push_back(i);
    }

    //Compute foreground/background features
    Mat descs_fg;
    Mat descs_bg;
    descriptor->compute(im_gray, keypoints_fg, descs_fg);
    descriptor->compute(im_gray, keypoints_bg, descs_bg);
	//Mat database(descs_bg.rows+descs_fg.rows*10 ,descs_bg.cols, descs_bg.type());
    //Only now is the right time to convert keypoints to points, as compute() might remove some keypoints
    vector<Point2f> points_fg;
    vector<Point2f> points_bg;

    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_fg.push_back(keypoints_fg[i].pt);
    }

    FILE_LOG(logDEBUG) << points_fg.size() << " foreground points.";

    for (size_t i = 0; i < keypoints_bg.size(); i++)
    {
        points_bg.push_back(keypoints_bg[i].pt);
    }

    //Create normalized points
    vector<Point2f> points_normalized;
    for (size_t i = 0; i < points_fg.size(); i++)
    {
        points_normalized.push_back(points_fg[i] - center);
    }

    //Initialize matcher
    matcher.initialize(points_normalized, descs_fg, classes_fg, descs_bg, center);

    //Initialize consensus
    consensus.initialize(points_normalized);

    //Create initial set of active keypoints
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_active.push_back(keypoints_fg[i].pt);
        classes_active = classes_fg;
    }

    FILE_LOG(logDEBUG) << "CMT::initialize() return";
}

void CMT::processFrame(Mat im_gray) {

    FILE_LOG(logDEBUG) << "CMT::processFrame() call";
	clock_t s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,s13;
    //Track keypoints
	s1 = clock();
    vector<Point2f> points_tracked;
    vector<unsigned char> status;
   tracker.track(im_prev, im_gray, points_active, points_tracked, status);


   int diff;
   for (int i = 0; i< points_tracked.size(); i++)
   {

	   int tx = points_tracked[i].x;
	   int ty = points_tracked[i].y;
	   int ax = points_active[i].x;
	   int ay = points_active[i].y;
	   if (status[i] != 0 && 0 < tx&&tx < im_gray.cols && 0 < ty&&ty < im_gray.rows && 0 < ax&&ax < im_prev.cols && 0 < ay&&ay < im_prev.rows)
	   {
		   uchar  datatrack = im_gray.ptr<uchar>(ty)[tx];
		   uchar  dataactive = im_prev.ptr<uchar>(ay)[ax];
		   // cout << i << endl;
		   diff = datatrack - dataactive;

		   if (-75 > diff || diff > 75)
		   {
			   points_tracked.erase(points_tracked.begin() + i);
			   status[i] = 0;

		   }
	   }
   }

    FILE_LOG(logDEBUG) << points_tracked.size() << " tracked points.";
	Mat showgray;
	showgray = im_gray.clone();
	for (size_t i = 0; i <points_tracked.size(); i++)
	{
		circle(showgray, points_tracked[i], 6, Scalar(0, 255, 0));
	}

	imshow("trackpoing", showgray);
    //keep only successful classes
    vector<int> classes_tracked;
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        if (status[i])
        {
            classes_tracked.push_back(classes_active[i]);
        }

    }
	s2 = clock();
	cout << "21" << endl;
    //Detect keypoints, compute descriptors
    vector<KeyPoint> keypoints;
	//detector->set("threshold", 10);
	//detector->detect(im_gray, keypoints);
	FAST_test<16>(im_gray, keypoints, thr, true, imax, histSize);
	//FAST(im_gray, keypoints, 20);
    FILE_LOG(logDEBUG) << keypoints.size() << " keypoints found.";
	s3 = clock();
    Mat descriptors;
    descriptor->compute(im_gray, keypoints, descriptors);
	s4 = clock();
    //Match keypoints globally
    vector<Point2f> points_matched_global;
    vector<int> classes_matched_global;
    matcher.matchGlobal(keypoints, descriptors, points_matched_global, classes_matched_global);
	cout << "22" << endl;
	Mat showgray1;
	showgray1 = im_gray.clone();
	for (size_t i = 0; i < points_matched_global.size(); i++)
	{
		circle(showgray1, points_matched_global[i], 7, Scalar(0, 0, 255));
	}
	imshow("matchpoint", showgray1);

    FILE_LOG(logDEBUG) << points_matched_global.size() << " points matched globally.";
	s5 = clock();
    //Fuse tracked and globally matched points
    vector<Point2f> points_fused;
    vector<int> classes_fused;
    fusion.preferFirst(points_tracked, classes_tracked, points_matched_global, classes_matched_global,
            points_fused, classes_fused);
	cout << "23" << endl;
    FILE_LOG(logDEBUG) << points_fused.size() << " points fused.";
	s6 = clock();
    //Estimate scale and rotation from the fused points
    float scale;
    float rotation;
    consensus.estimateScaleRotation(points_fused, classes_fused, scale, rotation);
//	consensus.estimateScaleRotation(points_matched_global, classes_matched_global, scale, rotation);
	cout << "24" << endl;

    FILE_LOG(logDEBUG) << "scale " << scale << ", " << "rotation " << rotation;
	s7 = clock();
    //Find inliers and the center of their votes
    Point2f center;
    vector<Point2f> points_inlier;
    vector<int> classes_inlier;
    consensus.findConsensus(points_fused, classes_fused, scale, rotation,
            center, points_inlier, classes_inlier);
	cout << "25" << endl;
    FILE_LOG(logDEBUG) << points_inlier.size() << " inlier points.";
    FILE_LOG(logDEBUG) << "center " << center;
	s8 = clock();
    //Match keypoints locally
    vector<Point2f> points_matched_local;
    vector<int> classes_matched_local;
    matcher.matchLocal(keypoints, descriptors, center, scale, rotation, points_matched_local, classes_matched_local);
	cout << "26" << endl;
    FILE_LOG(logDEBUG) << points_matched_local.size() << " points matched locally.";
	s9 = clock();
    //Clear active points
    points_active.clear();
    classes_active.clear();
	s10 = clock();
    //Fuse locally matched points and inliers
    fusion.preferFirst(points_matched_local, classes_matched_local, points_inlier, classes_inlier, points_active, classes_active);
 //   points_active = points_fused;
 //   classes_active = classes_fused;
	cout << "27" << endl;
    FILE_LOG(logDEBUG) << points_active.size() << " final fused points.";
	s11 = clock();
	//if(points_active.size()<6)
    //TODO: Use theta to suppress result
	
	if (points_active.size() <= 1|| scale>2.5)
	{
	//	bb_rot = RotatedRect(img_center, im_gray.size, 0);
		center.x = im_gray.cols / 2;
		center.y = im_gray.rows / 2;
		Size2f o_size = im_gray.size();
		bb_rot = RotatedRect(center, o_size, rotation / CV_PI * 180);
		//points_active.clear();
		//classes_active.clear();
	}
	else bb_rot = RotatedRect(center,  size_initial * scale, rotation/CV_PI * 180);




    //Remember current image
    im_prev = im_gray;
	s12 = clock();

    
    FILE_LOG(logDEBUG) << "CMT::processFrame() return";
}

} /* namespace CMT */

