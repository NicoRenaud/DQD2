# DQD2

The Delft Quantum Dot Detector (DQD2) processes TEM images of nanostructures to extract their size distribution. 
The program consists of a single python script that heavily relies on the library sci-kit. The code
only needs as input the image to be processed and its dimension. The image must be square so far.

To launch the code simply type

python imageProcess.py image_file size

Many more options are available and can be printed by typing

python imageProcess.py --help



