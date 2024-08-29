import cv2
from ultralytics import YOLO
import numpy as np
from djitellopy import Tello


try:
    tello = Tello()
    #initialising Tello drone object
except:
    print("No Tello instance created")

file = open("objects.txt", "r")
data = file.read()
object_list = data.split("\n")
file.close()

model = YOLO('yolov8n.pt', 'v8')

cap = cv2.VideoCapture(0)

def main():
    tello.connect()
    print("Battery level is", tello.get_battery())

    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0



def objectDetectionLoop():

    tello.streamoff()
    tello.streamon()
    frame_read = tello.get_frame_read()

    while True:
        
        frame = frame_read.frame

        # ret, frame= cap.read()
        
        results = model.track(frame, persist = True)
        detect_params = model.predict(source = frame, conf = 0.45, save = False)


        DP = detect_params[0].numpy()

        frame_ = results[0].plot()


        cv2.imshow("video", frame_)

    
        if len(DP) != 0:

            for i in range (len(detect_params[0])):
           
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]

                print("CLASS Id", clsID)
            
                if (clsID == 67):
                    print("phone found")
                    route(box)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    releaseVideo()



def releaseVideo():
    cap.release()
    cv2.destroyAllWindows()
            

def route(box):
    # box = boxes[i]
    itemConf, itemCoords =  box.conf.numpy()[0], box.xyxy.numpy()[0]

    #code to find the middle point of the phone if confidence is over 95%
    #turn towards so that i'm facing the middle point
    #fly towards the item



    


if __name__ == "__main__":
    main()

    while True:
        inp = input("1 to run object detection, to to run camera test")
        if inp == "1":
            objectDetectionLoop()
                
                     
        elif inp == "0":
            break
