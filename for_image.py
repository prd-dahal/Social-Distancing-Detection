from flask import Flask, render_template, Response
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

def eucledian_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    distance = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return distance
    
app = Flask(__name__)

@app.route('/')
def index():
    app.logger.info('Hello ')
    return render_template('index.html')




net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []

#coco dataset contains 80 classes
with open('coco.names', 'r') as f:
    classes = [ line.strip() for line in f.readlines()]
    

#get the layer names
layer_names = net.getLayerNames()

#get o/p layer
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#color for each class labels
color_red = (0, 0,255)
color_green =(0,255,0)

#print(output_layers)

#load image
#img = cv2.imread('gb.jpg', 1)
#img = cv2.resize(img, None, fx = 0.5, fy = 0.5)

#start camera
def get_frame():
    cap = cv2.imread('test_images/test4.jpg')


    # We convert the resolutions from float to integer.
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
  
    
    
    # Capture frame-by-frame
    frame = cap
    
          
        # Display the resulting frame
    img = frame
     
        
        #img = cv2.resize(img, (420, 640))
    height, width, n_channels = img.shape
        
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #get blob from img..img, scaleFactor, size, means of channel, RGB?
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False) 
        
        #see blob of each channel
        #for b in blob:
         #   for n, img_b in enumerate(b):
                #cv2.imshow(str(n), img_b)
            
        # send image to input layer
    net.setInput(blob)
    outs = net.forward(output_layers)        
        #print(outs)
        #print(outs.shape)
    class_ids = []
    boxes = []
    confidences = []
        
        
        #showing info
        
        # loop through all outputs it contains, center coo., height, width, class ids, prediction scores
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
                
            if confidence > 0.5:
                    #print(classes[class_id], ' detected.')
                    
                cx = int(det[0] * width)
                cy = int(det[1] * height)
                    
                w = int(det[2] * width)
                h = int(det[3] * height)
                    
                    #rectangle_coo  
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                    
                    
                    # add bounding box, confidences, class ids to array
                boxes.append([x, y, w, h,cx,cy])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                    
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
                    #cv2.circle(img, (cx, cy), 10, 2)
                    
        
        
        # print(len(boxes))
        
    n_det = len(boxes)
        
        # NMS is used to remove alike boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #removes boxes those are alike
    font = cv2.FONT_HERSHEY_PLAIN
    center_of_persons =[]    
    for i in range(n_det):
        if i in indexes:
            x, y, w, h,cx,cy = boxes[i]
            label = str(classes[class_ids[i]])
            color = color_red
            if(label=='person'):
                center_of_persons.append([cx,cy,x,y,w,h])
                #cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                #cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
                
    distance_between_centers = []
    close_rectangles = []
    further_rectangles =[]
    for i in range(len(center_of_persons)):
        for j in range(i+1, len(center_of_persons)):
            dis = eucledian_distance(center_of_persons[i],center_of_persons[j])/width
            if(dis<0.07):
                print('here')
                close_rectangles.append(center_of_persons[i])
                close_rectangles.append(center_of_persons[j])       
            distance_between_centers.append(dis)
    for i in (set(tuple(element) for element in close_rectangles)):
        cv2.rectangle(img, (i[2],i[3]), (i[2]+i[5] ,i[3]+i[4]), color_red,1)
        cv2.putText(img, "Danger", (i[2], i[3] + 30), font, 1, color_red, 2)
    for i in center_of_persons:
        if i not in close_rectangles:
            cv2.rectangle(img,(i[2],i[3]), (i[2]+i[5] ,i[3]+i[4]), color_green,1)
            cv2.putText(img, "Safe",(i[2], i[3] + 30), font, 1, color_green, 2)
    print(distance_between_centers)
    imgencode=cv2.imencode('.jpg',img)[1]
    stringData=imgencode.tostring()
    yield(b'--frame\r\n'
        b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
      # Break the loop
     
@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', debug=False, threaded=True)
