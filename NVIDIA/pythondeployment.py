import caffe
import cv2
import sys
import matplotlib.pyplot as plt
#import Image

def deploy(img_path):

    caffe.set_mode_gpu()
    MODEL_JOB_DIR = '/dli/data/digits/20180301-185638-e918'
    DATASET_JOB_DIR = '/dli/data/digits/20180222-165843-ada0'
    ARCHITECTURE = MODEL_JOB_DIR + '/deploy.prototxt'
    WEIGHTS = MODEL_JOB_DIR + '/snapshot_iter_735.caffemodel'
    
    # Initialize the Caffe model using the model trained in DIGITS.
    net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       
    # Create an input that the network expects. 
    input_image= caffe.io.load_image(img_path)
    test_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image(DATASET_JOB_DIR + '/mean.jpg')
    test_image = test_image-mean_image

 
    prediction = net.predict([test_image])
    
    #print("Input Image:")
    #plt.imshow(sys.argv[1])
    #plt.show()
    #Image.open(input_image).show()
    print(prediction)
    ##Create a useful output
    print("Output:")
    if prediction.argmax()==0:
        print "Sorry cat:( https://media.giphy.com/media/jb8aFEQk3tADS/giphy.gif"
    else:
        print "Welcome dog! https://www.flickr.com/photos/aidras/5379402670"
   

    
##Ignore this part    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))