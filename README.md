MangaTextDetection
==================

Experiments in text localization and detection in raw manga scans. Mostly using OpenCV python API.


Overview
--------
This repository holds some experiments I did in summer 2013 during a sudden interest in text detection in images. It uses some standard techniques (run length smoothing, connected component analysis) and some experimental stuff. Overall, I was able to get in the neighborhood of where I wanted to be, but the results are very processing intensive and not terribly reliable.

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

```
 ../LocateText.py '週刊ヤングマガジン31号194.jpg' -o 194_text_locations.png
```

With the results as follows (estimated text marked with red boxes):

![locate text output](https://github.com/johnoneil/MangaTextDetection/blob/master/test/194_text_locations_thumb.png?raw=true)

Note that in the output above you see several of the implementation deficiencies. For example, there are several small false positives scattered around, and some major false positives on the girl's sleeve and eyes in panels 2 and 3.
Also note that many large areas of text were not detected (false negatives). Despite how pleased I was with the results (and I was more pleased than you could possibly believe) significant improvements are needed.

Text Segmentation Example
-------------------------
To more easily separate text from background you can also segment the image, with text areas and non text being separated into different (RGB) color channels. This easily allows you to remove estimated text from image entirely or vice-versa.
Use the command:
```
./segmentation.py '週刊ヤングマガジン31号194.jpg' -o 194_segmentation.png
```
The results follow:

![Input image](https://github.com/johnoneil/MangaTextDetection/blob/master/test/194_segmentation_thumb.png?raw=true)

OCR and Html Generation
-----------------------
I did take the time to run simple OCR on some of the located text regions, with mixed results. I used the python tesseract package (pytesser) but found the results were not generally good for vertical text, among other issues.
The script ocr.py should run ocr on detected text regions, and output the results to the command line.
```
../ocr.py '週刊ヤングマガジン31号194.jpg'
Test blob assigned to no row on pass 2
Test blob assigned to no row on pass 3
0,0 1294x2020 71% :ぅん'・ 結局
玉子かけご飯が
一 番ぉぃしぃと

从
胤
赫
囃
包
け
H」
の
も
側
鵬

はフィクショ穴ぁり、 登場する人物

※この物語

```
You can see some fragmented positives, but in all the results for this page are abysmal.

I also embedded those results in an HTML output, allowing "readers" to hover on Japanese Text, revealing the OCR output, which can be edited/copied/pasted. This is via the script MangaDetectText. A (more successful) example of this can be seen below:

![locate text output](https://github.com/johnoneil/MangaTextDetection/blob/master/test/example.png?raw=true)

Dependencies
-----------------------
You should be able to install most of the dependencies via pip, or you could use your operating systems package manager (e.g. Mac OS X http://brew.sh/)

### Python 2.7+

https://www.python.org/

Install as per OS instructions.

### Pip

http://pip.readthedocs.org/en/latest/index.html

Install as per OS instructions.

### Numpy

http://www.numpy.org/

```
pip install numpy
```

### Scipy

http://www.scipy.org/index.html

```
pip install scipy
```

### Matplotlib (contains PyLab)

http://matplotlib.org/

```
pip install matplotlib
```

### Pillow

http://pillow.readthedocs.org/en/latest/

```
pip install Pillow
```

### OpenCV

http://opencv.org/

```
Install as per OS instructions, this should also include the python bindings.
```

### Tesseract

https://code.google.com/p/tesseract-ocr/

Install as per OS instructions, then use pip to install the python bindings.
Don't forget to include your target language's trained data sets.

```
pip install python-tesseract
```
