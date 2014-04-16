MangaTextDetection
==================

Experiments in text localization and detection in raw manga scans. Mostly using OpenCV python API.


Overview
--------
This repository holds some experiments I did in summer 2013 during a sudden interest in text detection in images. It uses some standard techniques (run length smoothing, connected component analysis) and some experimental stuff. Overal, I was able to get in the neighborhood of where I wanted to be, but the results are very processing intensive and not terribly reliable.

State
-----
I haven't bothered to form this into a python library. It's just a series of scripts each trying out various things, such as:
* Isolating bounding boxes for text areas on a raw manga page.
* Identifying ares of furigana text (pronunciation guide, which can screw up OCR) in text bounding boxes.
* Preparing identified text areas for basic OCR.


Text Location Example
---------------------
Here's an example run of a page from Weekly Young Magazine #31 2013. The input image is as follows (jpg).
![Input image](https://github.com/johnoneil/MangaTextDetection/blob/master/test/194.jpg?raw=true)

An initial estimate of text locations can be found by the 'LocateText.py' script:
'''
 ../LocateText.py '週刊ヤングマガジン31号194.jpg' -o 194_text_locations.png

'''
With the results as follows (estimated text marked with red boxes):

![locate text output](https://github.com/johnoneil/MangaTextDetection/blob/master/test/194_text_locations_thumb.png?raw=true)

Text Segmentation Example
-------------------------
To more easily separate text from background you can also segment the image, with text areas and non text being separated into different (RGB) color channels. This easily allows you to remove estimated text from image entirely or vice-versa.
Use the command: 



