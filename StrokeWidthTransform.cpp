/*
 # vim: set ts=2 expandtab:

FILE: StrokeWidthTranform.hpp
AUTHOR: John O'Neil
DATE: Saturday, July 13th 2013
EMAIL: oneil.john@gmail.com
DESC: Implementation of stroke width tranform algorithm as described
in http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf

I'm trying to separate later steps (connected component and geometry
operations out of the basic algorithm described). This is really
only just the core computation of stroke rays from edges,gradients.

This uses the opencv2 C++ api

*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "StrokeWidthTransform.hpp"

const double PI = 3.1415926;

using namespace cv;
using std::cout;
using std::endl;

void EstimatePixelStrokeWidth(const int row,const int col,const Mat& edges,
  const Mat& gradient_x,const Mat& gradient_y, Mat& output);

/*
  Computes first-pass basic stroke width tranform on an input image.
  Returns image where each pixel is either zero (pixes outside edges)
  or the initial (not averaged, not connected component) estimate
  for the stroke width.
*/
Mat StrokeWidthTransform(const Mat& image)
{
  //1)Do edge detection
  Mat edges = Mat::zeros(image.rows, image.cols, CV_64F);
  Canny(image, edges, 128.0, 200.0,3);
  threshold(edges, edges, 32, 255, THRESH_BINARY);
  //imshow( "swt_edges", edges );

  Mat smoothed;
  //resize(image, smoothed, Size(), 1.0/255.0, 1.0/255.0 );
  GaussianBlur( image, smoothed, Size( 5, 5 ), 0, 0 );
  //imshow( "smoothed", smoothed );

  //2)Get gradients in x,y directions
  Mat gradientX = Mat::zeros(image.rows, image.cols, CV_64F);
  Sobel(smoothed, gradientX,
    CV_64F,1/*1st x derivative*/, 0);
  GaussianBlur( gradientX, gradientX, Size( 3, 3 ), 0, 0 );

  Mat gradientY = Mat::zeros(image.rows, image.cols, CV_64F);
  Sobel(smoothed, gradientY,
    CV_64F,0,1/*1st y derivative*/);
  GaussianBlur( gradientY, gradientY, Size( 3, 3 ), 0, 0 );

  //3)Compute first pass stroke width estimates
  Mat stroke_width = Mat::zeros(edges.rows, edges.cols, CV_64F);
  edges.convertTo(edges, CV_64F, 1.0/255.0);
  for(int row=0;row<stroke_width.rows;++row)
  {
    for(int col=0;col<stroke_width.cols;++col)
    {
      if(edges.at<double>(row, col) > 0.0)
      {
        EstimatePixelStrokeWidth(row,col,edges,gradientX,gradientY,stroke_width);
      }
    }    
  }

  return stroke_width;
};

void EstimatePixelStrokeWidth(const int row,const int col,const Mat& edges,
  const Mat& gradient_x,const Mat& gradient_y, Mat& output)
{
  //1) get gradient for this pixel
  //We're flipping gradients here to hopefully handle white text on dark backgrounds
  //more gracefully
  double gx = -1.0 * gradient_x.at<double>(row,col);
  double gy = -1.0 * gradient_y.at<double>(row,col);
  //if(dx<0.0f){gx*=-1.0;}
  //if(dx<0.0f){gy*=-1.0;}
  const double mag = sqrt(gx*gx+gy*gy);
  Point2d g(gx/mag,gy/mag);

  //In the unlikely case where there's no gradient direction, bail.
  if(gx==0.0 && gy==0.0)
  {
    return;
  }

  //2) Shoot a ray from this pixel along the gradient until we hit another edge.
  //  We take the image bounds to be edges for this, allowing us to process text
  // that may be 'cut off' at the image boundary.
  const Point2d p(static_cast<double>(col)+0.5,static_cast<double>(row)+0.5);
  Point2d q=p;
  int n=1;
  int delta_x=0;
  int delta_y=0;
  Point prev_qi(col,row);
  for(n=1;;++n)
  {
    q=p+g*(static_cast<double>(n)*0.05);
    Point qi(static_cast<int>(q.x),static_cast<int>(q.y));
    if(qi==prev_qi)
    {
      continue;
    }

    if(qi.x >= edges.cols || qi.y >= edges.rows || qi.x < 0 || qi.y < 0)
    {
      break;
    }

    prev_qi = qi;

    if(edges.at<double>(qi.y,qi.x)>0)
    {
      break;
    }
  }
  
  //We've now got a ray along the gradient starting at the edge pixel, and
  //a point where it intercepts an "opposite" edge. Compute the angle
  //between the gradients at these points and filter out inappropritate
  //values (i.e. bail if the angle between is >PI/6)
  const double gx_f =  1.0 * gradient_x.at<double>(prev_qi.y,prev_qi.x);
  const double gy_f = 1.0 * gradient_y.at<double>(prev_qi.y,prev_qi.x);
  const double mag_f = sqrt(gx_f*gx_f+gy_f*gy_f);
  const Point2d gf(gx_f/mag_f,gy_f/mag_f);
  const double dp = g.x*gf.x + g.y*gf.y;
  const double a_rad = acos(dp);
  if(a_rad > PI/4.0)
  {
    return;
  }
  //and Filter out any obviously too large rays
  if(n*0.05 >200)
  {
    return;
  }

  //Fill in the pixel values along the found ray with the width value
  //(really the minimum of their current value and the width value).
  q=p;
  prev_qi = Point(col,row);
  for(int m=1;m<=n;++m)
  {
    q=p+g*(static_cast<double>(m)*0.05);
    Point qi(static_cast<int>(q.x),static_cast<int>(q.y));

    if(qi == prev_qi)
    {
      continue;
    }
    const double current_value = output.at<double>(qi.y,qi.x);
    const double calculated_width = static_cast<double>(n)*0.05;
    if(current_value==0.0)
    {
      output.at<double>(qi.y,qi.x) = calculated_width;
    }else{
      output.at<double>(qi.y,qi.x) = std::min(current_value,calculated_width);
    }

    prev_qi = qi;
  }
}
