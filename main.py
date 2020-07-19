from flask import Flask, render_template, Response
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    cap = cv2.VideoCapture('pedestrain.png')


    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
  
    while(cap.isOpened()):
    
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
          
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
        
            for i in range(n_det):
                if i in indexes:
                    x, y, w, h,cx,cy = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = color_red
                    if(label=='person'):
                    	print("{} and {}".format(cx,cy))
                    	cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                    	cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
            
        imgencode=cv2.imencode('.jpg',img)[1]
        stringData=imgencode.tostring()
        yield(b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(cap) 
      # Break the loop
     
@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', debug=False, threaded=True)
