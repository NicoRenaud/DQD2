# DQD2

The Delft Quantum Dot Detector (DQD2) processes TEM images of nanostructures to extract their size distribution. 
The program consists of a single python script that heavily relies on the library sci-kit. The code
only needs as input the image to be processed and its dimension. The image must be square so far.

To launch the code simply type

python imageProcess.py image_file size

Many more options are available and can be printed by typing

python imageProcess.py --help

The code will generate a summary of the image processing similar to the one represented below

![Alt text](image1.png?raw=true "Title")

In addition the code will output an histogram of the size and aspect ratio distribution of the dots

![Alt text](image2.png?raw=true "Title")

The code is far from bullet proof but has been proven to be quite versatile thanks to 
many options available
