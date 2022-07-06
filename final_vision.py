from tkinter.tix import Tree
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torchvision
from torchvision import models
from datetime import datetime
import time
import os
import psutil
import socket

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
load_weights = "./DeepLabv3_100_640640.pth"



if __name__ == "__main__":
    #load the model
    net = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    net.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 4)
    
    #load the model from host path
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(load_weights).items()})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device ：", device)
    net = net.eval()
    net.to(device)


def img_transform_tensor (frame, color_mean, color_std, square_px):
    #img_ = img_.resize((600, 600) , Image.BICUBIC)
    img_ = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_ = img_.resize((square_px, square_px), Image.BICUBIC)
    img_ = transforms.functional.to_tensor(img_)
    img_ = transforms.functional.normalize(img_, color_mean, color_std)
    data = img_.unsqueeze(0)
    return data



font = cv2.FONT_HERSHEY_SIMPLEX
skin = False
def visualization (model, image_in,  tensor_img, frame_height, frame_width, alpha, p_palette, device):
    global skin
    over = image_in.copy()
    tensor_img=tensor_img.to(device)
    outputs = model(tensor_img)
    y = outputs['out']  
    y = y[0].detach().cpu().numpy()
    y = np.argmax(y, axis=0)
    visual = Image.fromarray(np.uint8(y), mode="P")
    visual = visual.resize((frame_width, frame_height), Image.NEAREST)
    visual.putpalette(p_palette)
    visual=visual.convert('RGBA')
    open_cv_image = np.array(visual) 
    open_cv_image = cv2.cvtColor (open_cv_image, cv2.COLOR_RGB2BGR)
    open_cv_images = open_cv_image.copy()
    imgray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imgray, 0, 255, 0)
    ctr, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lst_intensities = []
    text =''
    #skin = False
    jumlah = len (ctr)
    for i in range (len(ctr)):
        c =  ctr[i] #max(ctr, key = cv2.contourArea)
        cimg=open_cv_image.copy()
        
        text = 'defect'
        cv2.drawContours(image_in, ctr, i, (0,0,128), 3)
        cv2.drawContours(over, ctr, i,(0,0,128), -1)
  
        M = cv2.moments(c)
        ax = M['m10']
        ay = M['m01']
        az = M['m00']
        print (text)
        if ax != 0 and ay != 0 and az != 0:
            cx = int (ax/az)
            cy = int (ay/az)
            data_x = 'X= '+str(cx)
            data_y = 'Y= '+str(cy)
            cv2.putText(image_in, text, (cx, cy), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image_in, data_x, (cx, cy+15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image_in, data_y, (cx, cy+25), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
            area = cv2.contourArea(c)
            print (area)
        else:
            continue
    outimg = cv2.addWeighted(over, alpha, image_in, 1 - alpha, 0)
    return outimg, mask, jumlah

def callback(x):
    pass
   

PATH_VIDEO = './Untitled.mp4'
frame_height = 1080
frame_width = 1920
square_px= 500
palate_mat = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 
64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 
128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 
128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]

if __name__ == "__main__":
    try:
        frist = 0
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW) 
        
        cap.set(3, frame_width)
        cap.set(4, frame_height)
        cv2.namedWindow('Segmentation', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('SetColors', cv2.WINDOW_AUTOSIZE)

        ilowH = 0
        ihighH = 103
        ilowS = 0
        ihighS = 255
        ilowV = 0
        ihighV = 255

        # create trackbars for color change
        cv2.createTrackbar('lowH','SetColors',ilowH,179,callback)
        cv2.createTrackbar('highH','SetColors',ihighH,179,callback)

        cv2.createTrackbar('lowS','SetColors',ilowS,255,callback)
        cv2.createTrackbar('highS','SetColors',ihighS,255,callback)

        cv2.createTrackbar('lowV','SetColors',ilowV,255,callback)
        cv2.createTrackbar('highV','SetColors',ihighV,255,callback)


        width = int (cap.get(3))
        height = int (cap.get(4))
        counter = 0
        prev_frame_time = 0
        new_frame_time = 0
        next = 0
        update_fps=0
        fps = 0
        prev_frame_time_ml= 0
        new_frame_time_ml = 0
        fps_ml = 0
        crop_size = 960
        cx=0;cy=0

        shoot_tolerant = 8
        shoot_point = 550
        counter_shoot = 0
        local_counter = 0
        get_slices = False
        last_shoot_counter=0
        area = 0.0
        send_bridge = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conveyor = 0
        string_message = "{:04d}{:d}{:d}".format(0,0,0)
        send_bridge.sendto(bytes(string_message, 'utf-8'), ("127.0.0.1", 1111))
        detect_quality=0
        auto_mode = True
        last_mode = True
        with torch.no_grad():
            while True:
                new_frame_time = time.time()
                load1, load5, load15 = psutil.getloadavg()
                cpu_usage = (load15/os.cpu_count()) * 100
                _, frame = cap.read()
                ilowH = cv2.getTrackbarPos('lowH', 'SetColors')
                ihighH = cv2.getTrackbarPos('highH', 'SetColors')
                ilowS = cv2.getTrackbarPos('lowS', 'SetColors')
                ihighS = cv2.getTrackbarPos('highS', 'SetColors')
                ilowV = cv2.getTrackbarPos('lowV', 'SetColors')
                ihighV = cv2.getTrackbarPos('highV', 'SetColors')

                update_fps=update_fps+1
                x0 = int ((width/2)-(height/2))
                proces_frame = frame.copy()
                y0 = int ((height/2) - (crop_size/2))
                x0 = int ((width/2) - (crop_size/2))+110
                crop_img = proces_frame[y0:y0+crop_size, x0:x0+crop_size] # cv2.imread("./1_cam_12222021_152342.jpg")
                save_crop = crop_img.copy()
                hsv_crop = crop_img.copy()
                hsv_crop = cv2.cvtColor(hsv_crop, cv2.COLOR_BGR2HSV)
                lower_hsv = np.array([ilowH, ilowS, ilowV])
                higher_hsv = np.array([ihighH, ihighS, ihighV])
                masks = cv2.inRange(hsv_crop, lower_hsv, higher_hsv)
                masks = (255 - masks)
                ctr, h = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if ctr:
                    for i in range (len(ctr)):
                        c = ctr[i]
                        area = cv2.contourArea(c)
                        if area>70000:
                            get_slices=True
                            M = cv2.moments(c)
                            ax = M['m10']
                            ay = M['m01']
                            az = M['m00']
                            if ax != 0 and ay != 0 and az != 0:
                                cx = int (ax/az)
                                cy = int (ay/az)
                        else:
                            continue
                else:
                    get_slices=False

                gcx = cx + x0
                gcy = cy + y0

                if auto_mode:
                    if (gcy <= (shoot_point+shoot_tolerant) and gcy >= (shoot_point-shoot_tolerant)):
                        local_counter+=1
                        if local_counter==2:
                            counter_shoot += 1
                        cv2.line(frame, (x0+30,int (height/2)), (x0+crop_size-30, int(height/2)), (3, 3, 138), 10)
                    else:
                        local_counter = 0
                        cv2.line(frame, (x0+30,int (height/2)), (x0+crop_size-30, int(height/2)), (51, 165, 50), 10)
                
                    
                fps_Dis = " | FPS = "+str(fps)
                cps = " | CPU Load = "+str(cpu_usage)
                apple_flow = str (counter_shoot)
                cv2.rectangle(frame, (0, 0), (1989, 35), (255, 10, 10), -1)
                cv2.putText(frame, fps_Dis, (300, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Slices Count :", (50, 180), font, 1, (3, 3, 138), 1, cv2.LINE_AA)
                cv2.putText(frame, apple_flow, (100, 300), font, 3, (3, 3, 138), 1, cv2.LINE_AA)
                cv2.putText(frame, "Auto Mode :", (50, 500), font, 1, (3, 3, 138), 1, cv2.LINE_AA)
                cv2.putText(frame, str(auto_mode), (100, 650), font, 3, (3, 3, 138), 1, cv2.LINE_AA)
                cv2.putText(frame, cps, (600, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Camera", (10, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x0, y0), (x0+crop_size, y0+crop_size), (0, 211, 255), 1)

                if get_slices:
                    cv2.circle(frame, (gcx,gcy), 10, (0,165,255), -1)
                display = cv2.resize(frame , (int(width*0.75), int(height*0.75)))

                #print ("CX = ", gcx)
                #print ("CY = ", gcy)
                
                tensor_img = img_transform_tensor (crop_img, color_mean, color_std, square_px)
                vis_frame, mask, sums =visualization(net, crop_img, tensor_img, crop_size, crop_size, 0.5,palate_mat,device)
                Save_img = vis_frame.copy()
                if (sums != 0):
                    detect_quality = 1
                else:
                    detect_quality = 0

                new_frame_time_ml = time.time()
                timer = round ((new_frame_time_ml-prev_frame_time_ml)*1000,3)
                times = "| Time Process = " + str (timer) + " ms"
                detects = "| Slices Count = " + str (counter_shoot) 
                cv2.rectangle(vis_frame, (0, 0), (1079, 35), (184,53,255), -1)
                cv2.putText(vis_frame, times, (200, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(vis_frame, detects, (680, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(vis_frame, "Inference", (10, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                display_ml = cv2.resize(vis_frame , (int(crop_size*0.75), int(crop_size*0.75)))
                #string_message = "{:04d}{:d}{:d}".format(counter_shoot,detect_quality,conveyor)
                #send_bridge.sendto(bytes(string_message, 'utf-8'), ("127.0.0.1", 1111))
                cv2.imshow('Segmentation', display_ml)

               
                key = cv2.waitKey(1)
                '''
                prev_frame_time_ml=time.time()
                if key == ord('s') or key == ord ('S') or counter_shoot != last_shoot_counter:
                    #detect_counter+=1
                    tensor_img = img_transform_tensor (crop_img, color_mean, color_std, square_px)
                    vis_frame, mask, sums =visualization(net, crop_img, tensor_img, crop_size, crop_size, 0.5,palate_mat,device)
                    Save_img = vis_frame.copy()
                    if (sums != 0):
                        detect_quality = 1
                    else:
                        detect_quality = 0

                    new_frame_time_ml = time.time()
                    timer = round ((new_frame_time_ml-prev_frame_time_ml)*1000,3)
                    times = "| Time Process = " + str (timer) + " ms"
                    detects = "| Slices Count = " + str (counter_shoot) 
                    cv2.rectangle(vis_frame, (0, 0), (1079, 35), (184,53,255), -1)
                    cv2.putText(vis_frame, times, (200, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(vis_frame, detects, (680, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(vis_frame, "Inference", (10, 25), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
                    display_ml = cv2.resize(vis_frame , (int(crop_size*0.75), int(crop_size*0.75)))
                    string_message = "{:04d}{:d}{:d}".format(counter_shoot,detect_quality,conveyor)
                    send_bridge.sendto(bytes(string_message, 'utf-8'), ("127.0.0.1", 1111))
                    cv2.imshow('Segmentation', display_ml)
                    nows = datetime.now()
                    date = nows.strftime("%d%m%Y%H%M%S")
                    cv2.imwrite('./data_log/inference_'+date+'_'+str(counter_shoot)+'.png', Save_img)
                    cv2.imwrite('./data_log/groundTht_'+date+'_'+str(counter_shoot)+'.png', save_crop)
                '''
                
                if key == ord ("q") or key == ord ("Q"):
                    break

                if key == ord('P') or key == ord ('p'):
                    cap.set(cv2.CAP_PROP_SETTINGS,1)
                
                if key == ord('A') or key == ord ('a'):
                    auto_mode = True
                
                if key == ord('M') or key == ord ('m'):
                    auto_mode = False

                if auto_mode != last_mode:
                    if auto_mode:
                        conveyor=1
                        string_message = "{:04d}{:d}{:d}".format(counter_shoot,detect_quality,1)
                        send_bridge.sendto(bytes(string_message, 'utf-8'), ("127.0.0.1", 1111))
                    else:
                        conveyor=0
                        string_message = "{:04d}{:d}{:d}".format(counter_shoot,detect_quality,0)
                        send_bridge.sendto(bytes(string_message, 'utf-8'), ("127.0.0.1", 1111))

               
                
                
              
                    
                cv2.imshow('Camera', display)
                #cv2.imshow('AutoTake', masks)
                #cv2.imshow('Camera 2', vis2)
                print ("Detect count =", counter_shoot)
                print ("Detect Qlty  =", detect_quality)
                print ("Auto Mode    =", auto_mode)
                if (update_fps>=5):
                    fps = round (1/(new_frame_time-prev_frame_time),3)
                    update_fps=0
                
                
                prev_frame_time = new_frame_time
                last_shoot_counter = counter_shoot
                last_mode = auto_mode

            
    finally:
        cv2.destroyAllWindows()
        cap.release()
        send_bridge.close()
        

            
        
        
