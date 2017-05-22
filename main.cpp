#include "CMT.h"
#include "gui.h"
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#ifdef __GNUC__
#include <getopt.h>
#else
#include "getopt/getopt.h"
#endif

using cmt::CMT;
using cv::imread;
using cv::imshow;
using cv::namedWindow;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using cv::VideoWriter;
using std::cerr;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::min_element;
using std::max_element;
using std::endl;
using ::atof;

static string WIN_NAME = "CMT";
static string OUT_FILE_COL_HEADERS =
    "Frame,Timestamp (ms),Active points,"\
    "Bounding box centre X (px),Bounding box centre Y (px),"\
    "Bounding box width (px),Bounding box height (px),"\
    "Bounding box rotation (degrees),"\
    "Bounding box vertex 1 X (px),Bounding box vertex 1 Y (px),"\
    "Bounding box vertex 2 X (px),Bounding box vertex 2 Y (px),"\
    "Bounding box vertex 3 X (px),Bounding box vertex 3 Y (px),"\
    "Bounding box vertex 4 X (px),Bounding box vertex 4 Y (px)";

vector<float> getNextLineAndSplitIntoFloats(istream& str)
{
    vector<float>   result;
    string                line;
    getline(str,line);

    stringstream          lineStream(line);
    string                cell;
    while(getline(lineStream,cell,','))
    {
        result.push_back(atof(cell.c_str()));
    }
    return result;
}

int display(Mat im, CMT & cmt)
{
    //Visualize the output
    //It is ok to draw on im itself, as CMT only uses the grayscale image
    for(size_t i = 0; i < cmt.points_active.size(); i++)
    {
        circle(im, cmt.points_active[i], 2, Scalar(255,0,0));
    }

    Point2f vertices[4];
    cmt.bb_rot.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(im, vertices[i], vertices[(i+1)%4], Scalar(255,0,0));
    }

    imshow(WIN_NAME, im);

    return waitKey(5);
}

string write_rotated_rect(RotatedRect rect)
{
    Point2f verts[4];
    rect.points(verts);
    stringstream coords;

    coords << rect.center.x << "," << rect.center.y << ",";
    coords << rect.size.width << "," << rect.size.height << ",";
    coords << rect.angle << ",";

    for (int i = 0; i < 4; i++)
    {
        coords << verts[i].x << "," << verts[i].y;
        if (i != 3) coords << ",";
    }

    return coords.str();
}

int main(int argc, char **argv)
{
    //Create a CMT object
    CMT cmt;

    //Initialization bounding box
    Rect rect;

	clock_t s1, s2;
	s1 = clock();
    //Parse args
    int challenge_flag = 0;
    int loop_flag = 0;
    int verbose_flag = 0;
    int bbox_flag = 0;
    int skip_frames = 0;
    int skip_msecs = 0;
    int output_flag = 0;
    string input_path;
    string output_path;

    const int detector_cmd = 1000;
    const int descriptor_cmd = 1001;
    const int bbox_cmd = 1002;
    const int no_scale_cmd = 1003;
    const int with_rotation_cmd = 1004;
    const int skip_cmd = 1005;
    const int skip_msecs_cmd = 1006;
    const int output_file_cmd = 1007;

    struct option longopts[] =
    {
        //No-argument options
        {"challenge", no_argument, &challenge_flag, 1},
        {"loop", no_argument, &loop_flag, 1},
        {"verbose", no_argument, &verbose_flag, 1},
        {"no-scale", no_argument, 0, no_scale_cmd},
        {"with-rotation", no_argument, 0, with_rotation_cmd},
        //Argument options
        {"bbox", required_argument, 0, bbox_cmd},
        {"detector", required_argument, 0, detector_cmd},
        {"descriptor", required_argument, 0, descriptor_cmd},
        {"output-file", required_argument, 0, output_file_cmd},
        {"skip", required_argument, 0, skip_cmd},
        {"skip-msecs", required_argument, 0, skip_msecs_cmd},
        {0, 0, 0, 0}
    };

    int index = 0;
    int c;
    while((c = getopt_long(argc, argv, "v", longopts, &index)) != -1)
    {
        switch (c)
        {
            case 'v':
                verbose_flag = true;
                break;
            case bbox_cmd:
                {
                    //TODO: The following also accepts strings of the form %f,%f,%f,%fxyz...
                    string bbox_format = "%f,%f,%f,%f";
                    float x,y,w,h;
                    int ret = sscanf(optarg, bbox_format.c_str(), &x, &y, &w, &h);
                    if (ret != 4)
                    {
                        cerr << "bounding box must be given in format " << bbox_format << endl;
                        return 1;
                    }

                    bbox_flag = 1;
                    rect = Rect(x,y,w,h);
                }
                break;
            case detector_cmd:
                cmt.str_detector = optarg;
                break;
            case descriptor_cmd:
                cmt.str_descriptor = optarg;
                break;
            case output_file_cmd:
                output_path = optarg;
                output_flag = 1;
                break;
            case skip_cmd:
                {
                    int ret = sscanf(optarg, "%d", &skip_frames);
                    if (ret != 1)
                    {
                      skip_frames = 0;
                    }
                }
                break;
            case skip_msecs_cmd:
                {
                    int ret = sscanf(optarg, "%d", &skip_msecs);
                    if (ret != 1)
                    {
                      skip_msecs = 0;
                    }
                }
                break;
            case no_scale_cmd:
                cmt.consensus.estimate_scale = false;
                break;
            case with_rotation_cmd:
                cmt.consensus.estimate_rotation = true;
                break;
            case '?':
                return 1;
        }

    }

    // Can only skip frames or milliseconds, not both.
    if (skip_frames > 0 && skip_msecs > 0)
    {
      cerr << "You can only skip frames, or milliseconds, not both." << endl;
      return 1;
    }

    //One argument remains
    if (optind == argc - 1)
    {
        input_path = argv[optind];
    }

    else if (optind < argc - 1)
    {
        cerr << "Only one argument is allowed." << endl;
        return 1;
    }

    //Set up logging
    FILELog::ReportingLevel() = verbose_flag ? logDEBUG : logINFO;
    Output2FILE::Stream() = stdout; //Log to stdout

    //Challenge mode
    if (challenge_flag)
    {
        //Read list of images
        ifstream im_file("images.txt");
        vector<string> files;
        string line;
        while(getline(im_file, line ))
        {
            files.push_back(line);
        }

        //Read region
        ifstream region_file("region.txt");
        vector<float> coords = getNextLineAndSplitIntoFloats(region_file);

        if (coords.size() == 4) {
            rect = Rect(coords[0], coords[1], coords[2], coords[3]);
        }

        else if (coords.size() == 8)
        {
            //Split into x and y coordinates
            vector<float> xcoords;
            vector<float> ycoords;

            for (size_t i = 0; i < coords.size(); i++)
            {
                if (i % 2 == 0) xcoords.push_back(coords[i]);
                else ycoords.push_back(coords[i]);
            }

            float xmin = *min_element(xcoords.begin(), xcoords.end());
            float xmax = *max_element(xcoords.begin(), xcoords.end());
            float ymin = *min_element(ycoords.begin(), ycoords.end());
            float ymax = *max_element(ycoords.begin(), ycoords.end());

            rect = Rect(xmin, ymin, xmax-xmin, ymax-ymin);
            cout << "Found bounding box" << xmin << " " << ymin << " " <<  xmax-xmin << " " << ymax-ymin << endl;
        }

        else {
            cerr << "Invalid Bounding box format" << endl;
            return 0;
        }

        //Read first image
        Mat im0 = imread(files[0]);
        Mat im0_gray;
        cvtColor(im0, im0_gray, CV_BGR2GRAY);

		//Mat zero1(im0_gray.rows, im0_gray.cols, im0_gray.depth(), Scalar(0));
		//Rect r;
		//r = Rect((rect.x - rect.width / 2) - 25, (rect.y - rect.height / 2) - 25, rect.width+50, rect.height+50);
		//Mat idk = zero1(r);
		//im0_gray(r).convertTo(idk, idk.depth(), 1, 0);
		//Initialize cmt;
		//copyMakeBorder(im0_gray, im0_gray, 0, 1, 0, 1, IPL_BORDER_CONSTANT, Scalar(0));
		
        cmt.initialize(im0_gray, rect);

        //Write init region to output file
        ofstream output_file("output.txt");
        output_file << rect.x << ',' << rect.y << ',' << rect.width << ',' << rect.height << std::endl;

        //Process images, write output to file
        for (size_t i = 1; i < files.size(); i++)
        {
            FILE_LOG(logINFO) << "Processing frame " << i << "/" << files.size();
            Mat im = imread(files[i]);
            Mat im_gray;
            cvtColor(im, im_gray, CV_BGR2GRAY);
            cmt.processFrame(im_gray);
            if (verbose_flag)
            {
                display(im, cmt);
            }
            rect = cmt.bb_rot.boundingRect();
            output_file << rect.x << ',' << rect.y << ',' << rect.width << ',' << rect.height << std::endl;
        }

        output_file.close();

        return 0;
    }

    //Normal mode

    //Create window
    namedWindow(WIN_NAME);

	

    VideoCapture cap;

    bool show_preview = true;

    //If no input was specified
    if (input_path.length() == 0)
    {
        cap.open(0); //Open default camera device
    }

    //Else open the video specified by input_path
    else
    {
        cap.open(input_path);

        if (skip_frames > 0)
        {
          cap.set(CV_CAP_PROP_POS_FRAMES, skip_frames);
        }

        if (skip_msecs > 0)
        {
          cap.set(CV_CAP_PROP_POS_MSEC, skip_msecs);

          // Now which frame are we on?
          skip_frames = (int) cap.get(CV_CAP_PROP_POS_FRAMES);
        }

        show_preview = false;
    }

    //If it doesn't work, stop
    if(!cap.isOpened())
    {
        cerr << "Unable to open video capture." << endl;
        return -1;
    }

    //Show preview until key is pressed
    while (show_preview)
    {
        Mat preview;
        cap >> preview;

        screenLog(preview, "Press a key to start selecting an object.");
        imshow(WIN_NAME, preview);

        char k = waitKey(10);
        if (k != -1) {
            show_preview = false;
        }
    }

    //Get initial image
    Mat im0;
    cap >> im0;

	//change
	//ifstream gtrect("groundtruth.txt");
	//vector<float> coords = getNextLineAndSplitIntoFloats(gtrect);

	//if (coords.size() == 4) {
	//	rect = Rect(coords[0], coords[1], coords[2], coords[3]);
	//}

	//bbox_flag = 1;
    //If no bounding was specified, get it from user
    if (!bbox_flag)
    {
        rect = getRect(im0, WIN_NAME);
    }

    FILE_LOG(logINFO) << "Using " << rect.x << "," << rect.y << "," << rect.width << "," << rect.height
        << " as initial bounding box.";

    //Convert im0 to grayscale
    Mat im0_gray;
    if (im0.channels() > 1) {
        cvtColor(im0, im0_gray, CV_BGR2GRAY);
    } else {
        im0_gray = im0;
    }





    //Initialize CMT
    cmt.initialize(im0_gray, rect);



    int frame = skip_frames;

    //Open output file.
    ofstream output_file;

    if (output_flag)
    {
        int msecs = (int) cap.get(CV_CAP_PROP_POS_MSEC);

        output_file.open(output_path.c_str());
        output_file << OUT_FILE_COL_HEADERS << endl;
        output_file << frame << "," << msecs << ",";
        output_file << cmt.points_active.size() << ",";
        output_file << write_rotated_rect(cmt.bb_rot) << endl;
    }

	ofstream outfile("crossing.txt");
	//VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, im0_gray.size());
    //Main loop
    while (true)
    {
        frame++;

        Mat im;

        //If loop flag is set, reuse initial image (for debugging purposes)
        if (loop_flag) im0.copyTo(im);
        else cap >> im; //Else use next image in stream

        if (im.empty()) break; //Exit at end of video stream

        Mat im_gray;
        if (im.channels() > 1) {
            cvtColor(im, im_gray, CV_BGR2GRAY);
        } else {
            im_gray = im;
        }


//选取前一帧bbox加长加宽后，作为下一帧判断图像
		//Mat zero(im_gray.rows, im_gray.cols, im_gray.depth(), Scalar(0));
		////Mat zero_prev;
		//float w = cmt.bb_rot.size.width;
		//float h = cmt.bb_rot.size.height;
		//float cx = cmt.bb_rot.center.x;
		//float cy = cmt.bb_rot.center.y;
		//float rx = (cx - w / 2) - 25;
		//float ry = (cy - h / 2) - 25;
		//float rw = w + 50;
		//float rh = h + 50;
		//if (rx <= 0 || ry <= 0 || (rx + rw ) >= im_gray.cols || (ry + rh) >= im_gray.rows)
		//{
		//	if (rx <= 0) rx = 0;
		//	if (ry <= 0) ry = 0;
		//	if ((rx + rw) >= im_gray.cols) rw = im_gray.cols - rx;
		//	if ((ry + rh) >= im_gray.rows) rh = im_gray.rows - ry;
		//}

		////Mat img1, img2;
		///*cout << "imgray.rows" << im_gray.rows << endl;
		//cout << "imgray.cols" << im_gray.cols << endl;
		//cout << "rx" << rx << endl;
		//cout << "rw" << rw << endl;*/
		///*Mat contours;
		//Canny(im_gray, contours, 125, 350);
		//threshold(contours, contours, 128, 255, CV_THRESH_BINARY);*/
		////copyMakeBorder(zero, img1, 1, 0, 1, 0, IPL_BORDER_CONSTANT, Scalar(0));

		////copyMakeBorder(zero, img2, 0, 1, 0, 1, IPL_BORDER_CONSTANT, Scalar(0));
		////框出ROI
		//Rect r;
		//r = Rect(rx, ry, rw, rh);
		//Mat idk = zero(r);
		//im_gray(r).convertTo(idk, idk.depth(), 1, 0);

		//zero = (img1 - img2) + (img2 - img1);
		//输出与groundtruth比较的数据
		//outfile << cx << " "
		//	<< cy << " " << w << " "
		//	<< h << endl;


	//	imshow("zero", im_gray);
		
		
		//copyMakeBorder(zero_prev, gray, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(0, 0, 0));

        //Let CMT process the frame
        cmt.processFrame(im_gray);
		
		
		
		//zero.copyTo(zero_prev);
        //Output.
        if (output_flag)
        {
            int msecs = (int) cap.get(CV_CAP_PROP_POS_MSEC);
            output_file << frame << "," << msecs << ",";
            output_file << cmt.points_active.size() << ",";
            output_file << write_rotated_rect(cmt.bb_rot) << endl;
        }
        else
        {
            //TODO: Provide meaningful output
            FILE_LOG(logINFO) << "#" << frame << " active: " << cmt.points_active.size();
        }

        //Display image and then quit if requested.
        char key = display(im, cmt);
        if(key == 'q') break;
	//	writer << im;
		//string qq = "VideoTest.avi";

    }
	s2 = clock();
	
	cout << "final time:" << s2 - s1 << endl;
	cout << "each time:" << (s2 - s1) / frame << endl;
    //Close output file.
    if (output_flag) output_file.close();

    return 0;
}
