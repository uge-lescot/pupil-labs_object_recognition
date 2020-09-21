'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs
 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import json
import string
from pydoc import replace
from typing import List

import cv2
import numpy as np
import zmq
import msgpack
from pyglui import ui
from pyglui.cygl.utils import draw_polyline, draw_points, RGBA, draw_gl_texture
import gl_utils
from glfw import *
from imagezmq import imagezmq
import argparse
import socket
import time
import os

from plugin import Plugin
# logging
import logging

#from pupil.capture_settings.plugins.darknet.build.darknet.x64.darknet import *
#from pupil.capture_settings.plugins.darknet.build.darknet.x64.darknet_video import *

# you should copy your darknet folder next to detection_plugin.py in the plugin folder
from darknet.build.darknet.x64.darknet import *
from darknet.build.darknet.x64.darknet_video import *

logger = logging.getLogger(__name__)
USER = os.getlogin()

OBJ_DATA = "object"

class Yolov4_Detection_Plugin(Plugin):
    """Describe your plugin here

    """
    def __init__(self, g_pool, my_persistent_var=10.0, ):
        super(Yolov4_Detection_Plugin, self).__init__(g_pool)
        # order (0-1) determines if your plugin should run before other plugins or after
        self.order = .1

        # all markers that are detected in the most recent frame
        self.my_var = my_persistent_var
        # attribute for the UI
        self.window = None
        self.menu = None
        self.img = None
        self.color = {
            "blue": [172, 183, 114],  # color : #5AA333 colors are in BGR format
            "red": [12, 15, 145]  # color : #9C2118
            }
        # booleans used in switches
        self.activation_state = False  # determine if object detection is active or not
        self.detect_mode = "Yolo 608 (best precision)"  # determine the model to use
        self.new_window = False  # determine if we open a new window for object detection
        self.model_loaded = False  # determine if we have loaded a model
        self.sight_only = False
        self.export = False
        self.is_streaming = False

        # streaming attributes
        self.SERVER_IP = "127.0.0.1"
        self.SERVER_PORT = 12001 # if you get errors while streaming, try different port number
        self.HOST_NAME = "localhost"
        self.sender = imagezmq.ImageSender(connect_to="tcp://{}:{}".format(self.SERVER_IP, self.SERVER_PORT))

        # we need to create a darknet image then use it for every detection
        self.darknet_image = None
        self.network_image_size = 608
        self.fps = 29.97

        self.gaze_position_2d = []
        self.focus_object_index=-1
        self.data = None
        self.width = 1280
        self.height = 720
        self.codec = cv2.VideoWriter_fourcc(*'DIVX') # codec use to create .avi video file
        # stream definition is HD by default but you can change it
        self.out = None
    def init_ui(self):
        # lets make a menu entry in the sidebar
        self.add_menu()
        self.menu.label = 'Yolo v4 Detection'
        self.menu.append(
            ui.Info_Text(
                "Classifies and labels objects seen by the user in the world camera video. Boxes will appear around objects, with their name and the confidence score at the top. When an object is being focused it will be exported to IPC Backbone in the topic \"object\""
            )
        )
        # here we add the selector for precision or speed
        detect_mode = ["Yolo 608 (best precision)", "Yolo 512", "Yolo 416", "Yolo 320", "Yolo Tiny (fastest)", "Yolo Mish (RTX 2070)","Yolo SAM Mish 512","Yolo Leaky (GTX 1080ti)", "Custom"]
        self.menu.append(
            ui.Selector(
                "detect_mode", self, selection=detect_mode, label="Optimize for"
            )
        )
        self.menu.append(
            ui.Info_Text(
                "Yolo 608/512 for precision, Yolo Tiny for speed, Yolo 416/320 for a good compromise."
            )
        )
        self.menu.append(
            ui.Info_Text(
                "Yolo Mish is optimized for RTX 2070 (8GB), Yolo Leaky is optimized for GTX 1080ti (11GB). Yolo SAM Mish should be used with 16GB card minimum."
            )
        )

        # here we add the switches and the "load model" button
        self.menu.append(ui.Button("Load Current Model", self.load_model))
        self.menu.append(ui.Switch("new_window", self, label="Open detection in new window"))
        self.menu.append(ui.Switch("sight_only", self, label="Only detect focused object"))
        self.menu.append(ui.Switch("is_streaming", self, label="Stream at tcp://127.0.0.1:12001"))
        self.menu.append(ui.Switch("export", self, label="Export data to local JSON and AVI"))
        self.menu.append(ui.Info_Text(
            "File will be saved at pupil_capture_settings/plugin/data."
        ))
        self.menu.append(ui.Switch("activation_state", self, label="Launch object recognition"))
        self.menu.append(ui.Info_Text(str(self.fps)+" fps"
       	))   
        
    # window calback
    def on_resize(self, window, w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfwMakeContextCurrent(active_window)

    def deinit_ui(self):
        self.remove_menu()

    
    def load_model(self):
    	# load the model based on which parameter we want.
    	# there are many models available, check AlexeyAB/darknet on github for more

        # global variables used to access darknet parameters
        global metaMain, netMain, altNames
        netMain = None
        metaMain = None
        altNames = None
        if self.detect_mode=="Yolo 608 (best precision)":

            # here you need to specify the path to darknet/x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            # configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4.cfg"
            # weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            # metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 608
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)
            # 608 is the image size needed for image detections with yolov4
            # the right way to do it is :
            # self.darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
            # but it raise an access violation for us

        elif self.detect_mode=="Yolo 512":

            # here you need to specify the path to darknet\x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4_512.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 512
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        elif self.detect_mode == "Yolo 416":

            # here you need to specify the path to darknet/x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-416.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 416
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)


        elif self.detect_mode == "Yolo 320":

            # here you need to specify the path to darknet/x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-320.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 320
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        elif self.detect_mode=="Yolo Tiny (fastest)":

            # here you need to specify the path to darknet\x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-tiny.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4-tiny.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 416
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)
            # 416 is the image size needed for image detections with yolov4-tiny
            # the right way to do it is :
            # self.darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
            # but it raise an access violation for us

        elif self.detect_mode=="Yolo Mish (RTX 2070)":

            # here you need to specify the path to darknet\x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-mish-416.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4-mish-416.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 416
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        elif self.detect_mode=="Yolo SAM Mish 512":

            # here you need to specify the path to darknet\x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-sam-mish.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4-sam-mish.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 512
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        elif self.detect_mode=="Yolo Leaky (GTX 1080ti)":

            # here you need to specify the path to darknet\x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-leaky-416.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4-leaky-416.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 416
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        elif self.detect_mode=="Custom":
            # here you need to specify the path to darknet/x64 directory and cfg, weight, data files
            configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4-self.cfg"
            weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4-self_13000.weights"
            metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/self.data"

            # configPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/yolov4.cfg"
            # weightPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/yolov4.weights"
            # metaPath = "C:/Users/"+USER+"/pupil_capture_settings/plugins/darknet/build/darknet/x64/cfg/coco.data"

            self.network_image_size = 608
            self.darknet_image = make_image(self.network_image_size, self.network_image_size, 3)

        else:
            print("Error : No model loaded.")
            raise ValueError("No model loaded.")

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if netMain is None:
            netMain = load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = load_meta(metaPath.encode("ascii"))
        if altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        self.model_loaded = True

    def cvDrawBoxes(self, detections, img):
        '''
        draw boxes and labels around objects
        :param detections: lists of objects detected, each object is also a list of class name,
         x position, y position, width and height of the box around it.
        :param img: the current img after detection
        :return: modified image
        '''
        self.focus_object_index=-1
        for detection in detections:
            x, y, w, h = detection[2][0], \
                         detection[2][1], \
                         detection[2][2], \
                         detection[2][3]

            # compute the coordinates of the box from yx,y,w and h
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))

            # rescale the size of the box from the size of the darknet image to the world camera image
            self.width, self.height = self.g_pool.capture.frame_size
            xmin = int(xmin * self.width / self.network_image_size)
            xmax = int(xmax * self.width / self.network_image_size)
            ymax = int(ymax * self.height / self.network_image_size)
            ymin = int(ymin * self.height / self.network_image_size)
            #print("x : "+str(xmin)+", "+str(xmax))
            #print("y : " + str(ymin) + ", " + str(ymax))
            #print(self.gaze_position_2d)
            if len(self.gaze_position_2d) != 0:
                if self.sight_only:
                    if xmin < self.gaze_position_2d[0] < xmax and ymin < self.gaze_position_2d[1] < ymax:
                        # create points to draw the rectangle and the text only if they are in sight and in red
                        pt1 = (xmin, ymin)
                        pt2 = (xmax, ymax)
                        cv2.rectangle(img, pt1, pt2, self.color["red"], 1)
                        cv2.putText(img,
                                    detection[0].decode() +
                                    " [" + str(round(detection[1] * 100, 2)) + "]",
                                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    self.color["red"], 2)
                        self.focus_object_index = detections.index(detection)
                else:
                    if xmin < self.gaze_position_2d[0] < xmax and ymin < self.gaze_position_2d[1] < ymax:
                        # create points to draw the rectangle and the text in red if they are in sight
                        pt1 = (xmin, ymin)
                        pt2 = (xmax, ymax)
                        cv2.rectangle(img, pt1, pt2, self.color["red"], 1)
                        cv2.putText(img,
                                    detection[0].decode() +
                                    " [" + str(round(detection[1] * 100, 2)) + "]",
                                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    self.color["red"], 2)
                        # store the index of the object in focus
                        self.focus_object_index = detections.index(detection)
                    else:
                        # create points to draw the rectangle and the text in green if not in sight
                        pt1 = (xmin, ymin)
                        pt2 = (xmax, ymax)
                        cv2.rectangle(img, pt1, pt2, self.color["blue"], 1)
                        cv2.putText(img,
                                    detection[0].decode() +
                                    " [" + str(round(detection[1] * 100, 2)) + "]",
                                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    self.color["blue"], 2)
            else:
                    # create points to draw the rectangle and the text in green(#5AA333) if no gaze data is gathered
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    cv2.rectangle(img, pt1, pt2, self.color["blue"], 1)
                    cv2.putText(img,
                                detection[0].decode() +
                                " [" + str(round(detection[1] * 100, 2)) + "]",
                                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                self.color["blue"], 2)
                    #print("No Gaze data, try calibrating to get focused object")
        return img



    def format_json(self, obj_datum, detections):
        data = {}
        if self.focus_object_index > 0:
            obj = detections[self.focus_object_index]
            name = obj[0].decode('utf-8')
            #print(name)
            data[name] = obj[2]
            obj_datum["objects"] = data
        return obj_datum

    def export_data_json(self, datum):
        """
      save JSON file to plugins/data/object_data.json
      :param datum: JSON file
      """
        if self.focus_object_index > 0:
            with open('C:/Users/' + USER + '/pupil_capture_settings/plugins/data/object_data.json', 'a',
                  newline='') as file:
                jstr = json.dumps(datum, indent=4) + '\n'
                file.write(jstr)

    def recent_events(self, events):
        prev_time = time.time()  # used to show fps
        events["objects"] = [] # creates the topic "objects" that will be used to export recognition data

        if 'frame' in events:
            frame = events['frame']
        else:
            return
        # get current frame
        self.img = frame.img

        # get the gaze data from Pupil Core
        gaze = events.get("gaze", [])
        try:
        	self.gaze_position_2d = list(gaze[0].get('norm_pos')) # "norm_pos" returns the gaze position in the normalized plane of the image.
        	self.gaze_position_2d[0] = self.gaze_position_2d[0] * self.width
        	self.gaze_position_2d[1] = (1-self.gaze_position_2d[1]) * self.height
        except IndexError: # if you don't calibrate the Pupil Core, gaze will be empty and it raises an IndexError.
        	self.gaze_position_2d = []
        
        # print(self.gaze_position_2d)
        # the vector base of this plane is the same as Yolo's vector base except that the y axis is inverted
        # therefore we need to convert back the normalized position and then rescale it with our image dimensions
        
        if self.activation_state:
            if self.model_loaded:
                # resize frame to right size for prediction
                frame_resized = cv2.resize(self.img,
                                           (network_width(netMain),
                                            network_height(netMain)),
                                           interpolation=cv2.INTER_LINEAR)
                copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())  # transform the current frame ina darknet compatible format
                detections = detect_image(netMain, metaMain, self.darknet_image, thresh=0.25)  # Yolo is used here to detect objects
                image = self.cvDrawBoxes(detections=detections, img=self.img)   # draw boxes around objects
                self.img = image  # display in Pupil Capture
                '''
                Export objects classes and box coordinates with their associated timestamps and coordinates in the topic "objects"
                We only export recognition data when the subject is focusing on an object. if no object is being watch, nothing happends.
                '''
                if self.focus_object_index > 0:
                    obj_entry = {
                        'topic': "objects",
                        'timestamp': self.g_pool.get_timestamp(),
                        'mode': self.detect_mode
                    }

                    obj_entry = self.format_json(obj_entry, detections=detections) # create JSON format string
                    events["objects"].append(obj_entry) # add the object data to "objects" topic
                    self.data = events["objects"] # getting back the data to use it, it allow to check if everything is fine

        # local .avi video exportation
        if self.export:
            if self.out is None:
                self.out = cv2.VideoWriter('C:/Users/' + USER + '/pupil_capture_settings/plugins/data/output.avi', self.codec, 24.0, (1280, 720))
            self.out.write(self.img)  # write local .avi video
            self.export_data_json(datum=self.data) # write local JSON file
        else:
            if self.out is not None:
                self.out.release()

        '''
        Stream the current frame at tcp://127.0.0.1:12001 but you can change ip and port in self.__init__().
        It sends JPG images but you can change the format if needed. 
        '''
        if self.is_streaming:
            if self.sender is None:
                self.sender = imagezmq.ImageSender(connect_to="tcp://{}:{}".format(self.SERVER_IP, self.SERVER_PORT), REQ_REP=False)
                print('Connect the ImageSender object to {}:{}'.format(self.SERVER_IP, self.SERVER_PORT))
                print('Host name is {}'.format(self.HOST_NAME))
            _, jpg_buffer = cv2.imencode(".jpg", self.img)
            self.sender.send_jpg(self.HOST_NAME, jpg_buffer)
        else:
            if self.sender is not None:
                self.sender.close() # close the stream when the switch is off
            self.sender = None

        # detection fps indicator
        total_time = time.time()-prev_time
        self.menu.elements.pop()
        self.fps = int((1 / total_time)*100)/100
        self.menu.append(ui.Info_Text(str(self.fps) + " fps"
        ))
        
        if self.new_window: # if you prefer to open the detection in a new window
            if self.img is not None:
                cv2.imshow('Demo', self.img)
                cv2.waitKey(3)

    def gl_display(self):
        """
        This is where we can draw to any gl surface
        by default this is the main window, below we change that
        """

        # active our window
        #active_window = glfwGetCurrentContext()
        #glfwMakeContextCurrent(self.window)

        # start drawing things:
        gl_utils.clear_gl_screen()
        # set coordinate system to be between 0 and 1 of the extents of the window
        gl_utils.make_coord_system_norm_based()
        # draw the image
        try:
        	draw_gl_texture(self.img)
        	gl_utils.make_coord_system_pixel_based(self.img.shape)
        except AttributeError:
        	pass
        # make coordinate system identical to the img pixel coordinate system

        # draw some points on top of the image
        # notice how these show up in our window but not in the main window
        # draw_points([(200, 400), (600, 400)], color=RGBA(0., 4., .8, .8), size=self.my_var)
        # draw_polyline([(200, 400), (600, 400)], color=RGBA(0., 4., .8, .8), thickness=3)

        # since this is our own window we need to swap buffers in the plugin
        #glfwSwapBuffers(self.window)

        # and finally reactive the main window
        #glfwMakeContextCurrent(active_window)

    def get_init_dict(self):
        # anything vars we want to be persistent accross sessions need to show up in the __init__
        # and identically as a dict entry below:
        return {'my_persistent_var': self.my_var}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        if self.out is not None:
            self.out.release()
        self.export_data_json(datum=self.data)
        #glfwDestroyWindow(self.window)