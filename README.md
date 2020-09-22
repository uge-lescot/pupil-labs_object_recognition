# pupil-labs_object_recognition
Real-Time Object Recognition plugin for pupil capture app

## Plugin for Pupil Capture, it provides : 
- detection of objects in the world video
- recognition of which object you are looking at, based on gaze data and Yolo coordinates.
- streaming of the video with objects detected on tcp, with compatibility with RTMaps (contact me for this)
- exporting data of which objects are in focus in a custom topic named "objects" in the IPC Backbone
- saving local .avi video and data of which objects are in focus in JSON

The plugin uses already trained Yolo models, but it is easy to add a custom model that can recognize other things than objects like facial emotion or whatever you need to.

## Installation steps
- Python 3.7
- Installing compilation tools
  - CMake
  - Visual Studio Community 2019
- Updating Graphics Drivers
- Installing CUDA
- Installing cuDNN
- Installing Open CV
- Configuring CMake
- Compiling OpenCV with Visual Studio
- Compiling Darknet
- Installing Pupil Capture
- Adding the plugin
Take a look at the documentation folder

## Requirements
- python 3.7
- cuda 10.2
- numpy
- cudnn 7.6.5 for cuda 10.2
- opencv 4.1.0

## For more information about Yolo and Pupil Capture, check out :
- https://github.com/AlexeyAB/darknet
- https://docs.pupil-labs.com/developer/

## Contact
Contact us on discord : baptonx#2813 | Fab#6442 ; or by email : b.broyer@free.fr | fabien.moreau@univ-eiffel.fr

Developped by Baptiste BROYER under the supervision of Fabien Moreau at Gustave Eiffel University Lyon-Bron campus
© 2020
