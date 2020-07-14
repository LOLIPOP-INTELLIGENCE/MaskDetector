import tensorflow as tf
import cv2
import numpy as np
import urllib

#Add url path with the /shot.jpg at the end
url='http://192.168.1.21:8080/shot.jpg'


interpreter = tf.lite.Interpreter(model_path="/Users/blackhole/Desktop/Mask Detection Model/masks-final1.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



interpreter.allocate_tensors()

vid = cv2.VideoCapture(0)

i=0


def draw_rect(image, box,detected_class):
    y_min = int(max(1, (box[0] * 512)))
    x_min = int(max(1, (box[1] * 512)))
    y_max = int(min(512, (box[2] * 512)))
    x_max = int(min(512, (box[3] * 512)))

    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    if(detected_class==0):
        cv2.putText(image, 'No Mask', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        global i
        #provide a path to a folder(In my case it is my downloads folder)
        path='/Users/blackhole/Downloads/'+str(i)+'.jpg'
        cv2.imwrite(path, image)
        i=i+1
        #This will save 100 images in the above path when it detceted no mask
        if(i==100):
            exit(0)

    else:
        cv2.putText(image, 'Mask', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

while(True):
    #Code to process image
    img=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(img.read()),dtype=np.uint8)
    new_img=cv2.imdecode(imgNp,-1)
    new_img=cv2.resize(new_img,(512,512))
    # ret, frame = vid.read()
    # new_img=cv2.resize(frame,(512,512))
    #new_img=cv2.resize(imgNp,(512,512))
    interpreter.set_tensor(input_details[0]['index'], [new_img])

    interpreter.invoke()
    rects = interpreter.get_tensor(
            output_details[0]['index'])

    scores = interpreter.get_tensor(
            output_details[2]['index'])


    detection_classes = interpreter.get_tensor(output_details[1]['index'])

    for index, score in enumerate(scores[0]):
        #print(score)
        if score > 0.1:
            draw_rect(new_img, rects[0][index],detection_classes[0][index])

    cv2.imshow("image", new_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
