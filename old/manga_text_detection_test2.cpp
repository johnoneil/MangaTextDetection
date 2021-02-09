/*
 # vim: set ts=2 expandtab:

g++ -o opencvtest opencvtest.cpp `pkg-config opencv --cflags --libs`

*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <list>
#include <algorithm>

#include "StrokeWidthTransform.hpp"


using namespace cv;
using std::cout;
using std::endl;
using std::list;


int main( int argc, char* argv[] )
{
  if(argc!=2)
  {
    cout<<"USAGE: opencvtest <input image filename>"<<endl;
    exit(-1);
  }

  const std::string infile_name(argv[1]);

  ///1) Load indicated image
  Mat src = imread(infile_name.c_str());
  if(!src.data){ cout<<"Error loading src1"<<endl; exit(-1); }

  //2) Get image dimensions
  const int w = src.cols;
  const int h = src.rows;
  cout<<"Input image is "<<w<<"px wide by "<<h<<"px high."<<endl;


  //3) Create a grayscale version of input image
  Mat gray;
  cvtColor(src,gray,CV_BGR2GRAY);
  //imshow( "gray", gray );

  //4) To a gaussian pyramid operations to remove basic tones moire patterns
  Mat scaled = gray;
  pyrDown( scaled, scaled, Size(w/2,h/2));
  pyrUp( scaled, scaled, Size(w,h));
  //imshow( "scaled", scaled );

  //5) Get Canny edges of grayscale image
  Mat edges;
  Canny(scaled, edges, 128.0, 200.0,3);
  
  //6) Threshold Canny edges to binary
  Mat bin_edges;
  threshold(edges,bin_edges, 32, 255, THRESH_BINARY);
  //imshow( "bin_edges", bin_edges );

  //7) Get contours (or connected components?)
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(bin_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  
  //find the stroke width transform image
  Mat swt = StrokeWidthTransform(scaled);
  imshow( "swt", swt );
  double min_stroke=0.0f,max_stroke=0.0f;
  minMaxLoc(swt, &min_stroke, &max_stroke);
  //cout<<"Stoke width min = "<<min_stroke<<" and max is "<<max_stroke<<endl;
  Mat swt_visualization;
  //swt.convertTo(swt_visualization, CV_8U, 255.0/(max_stroke - min_stroke), -min_stroke * 255.0/(max_stroke - min_stroke));
  swt.convertTo(swt_visualization, CV_8U, 255.0/(max_stroke), -min_stroke * 255.0/(max_stroke));
  //cout<<"swt converted is "<<swt_visualization.rows<<"x"<<swt_visualization.cols<<" element "<<swt_visualization.elemSize()<<endl;
  //imshow( "swt_visualization", swt_visualization);
  Mat color_swt;
  cvtColor(swt_visualization,color_swt,CV_GRAY2BGR);
  //imwrite( "swt_visualization.jpg", swt_visualization);

  //8) Draw contours
  Mat blobs = Mat::zeros(bin_edges.size(), CV_8UC3 );
  RNG rng(12345);
  for( int i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    Scalar green = Scalar(0,255,0);
    drawContours( color_swt, contours, i, green, 1);// 8, hierarchy, 0, Point() );
  }
  //imshow( "color_swt", color_swt );
  //imwrite( "color_swt.jpg", color_swt);

  //9)Find bounding boxes of contours and draw lines between contours which don't overlap
  //and are within max(w,h) * 0.5 pixels
  for( int i = 0; i< contours.size();++i)
  {
    Scalar red = Scalar(0,0,255);
    //drawContours( blobs, contours, i, red, 2, 8, hierarchy, 0, Point() );
    //Moments m = moments(contours[i], true );
    //double a_x = m.m10/m.m00;
    //double a_y = m.m01/m.m00;

    Rect a_r = boundingRect(contours[i]);
    double a_x = a_r.x + a_r.width/2;
    double a_y = a_r.y + a_r.height/2;

    list< list< int > > candidate_lists;
    
    for(int j=0;j<contours.size();++j)
    {
      if(j==i){continue;}

      //make sure bounding boxes don't overlap
      Rect b_r = boundingRect(contours[j]);
      double b_x = b_r.x + b_r.width/2;
      double b_y = b_r.y + b_r.height/2;
      if (a_r.x < b_r.x+b_r.width && a_r.x + a_r.width > b_r.x &&
        a_r.y < b_r.y + b_r.width && a_r.y+a_r.width > b_r.y)
      {
        continue;
      }

      //only link boxes of roughly similar minimum dimension
      double a_max = std::max(a_r.width,a_r.height);
      double b_max = std::max(b_r.width,b_r.height);
      //double d_min = std::abs(a_min-b_min);
      if(a_max > 1.5*b_max || b_max > 1.5*a_max){continue;}
      

      //Moments n = moments(contours[j], true );
      //double b_x = n.m10/n.m00;
      //double b_y = n.m01/n.m00;
      double d_x = a_x - b_x;
      double d_y = a_y - b_y;
      double d = sqrt(d_x*d_x + d_y*d_y);

      //double a_wh = std::max(a_r.width,a_r.height);
      if(d < ( 1.5 * 0.5 * (a_max+b_max) ) )
      {
        line(src, Point(a_x,a_y), Point(b_x,b_y), red);
        //link these two
        bool found = false;
        for(list< list<int> >::iterator l = candidate_lists.begin(); l!=candidate_lists.end();++l)
        {
          if( std::find(l->begin(),l->end(),i) !=l->end() )
          {
            if(std::find(l->begin(),l->end(),j) !=l->end() )
            {
              found = true;
              break;
            }else{
              l->push_back(j);
              found = true;
              break;
            }
          }
          else if(std::find(l->begin(),l->end(),j) !=l->end() )
          {
              l->push_back(i);
              found = true;
              break;
          }
        }
        if(!found)
        {
          list<int> newlist;
          newlist.push_back(i);
          newlist.push_back(j);
          candidate_lists.push_back(newlist);
        }
      }
      
    }
  }

  //imshow( "blobs", blobs );
  // show the image
  imshow( "Input", src );

  
  // Wait until user press some key
  waitKey(0);
  return 0;
}
