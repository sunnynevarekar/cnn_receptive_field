#Inspired from https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51 and extended the code
#to handle different height and width and sliding window visualization for receptive fileld

#usage: python receptivefield.py [-i IMAGE] [-s1 IMGHEIGHT] [-s2 IMGWIDTH] [-l LAYER]
 
import argparse
import math
import cv2
import matplotlib.pyplot as plt


#Each layer i requires the following parameters to be fully represented: 
    # - n_i: number of features in the (output) feature map (n_0_h = image_height, n_0_w = image_width)
    # - j_i: distance (projected to image pixel distance) between center of two adjacent features
    # - r_i: receptive field of a feature in layer i
    # - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

def outFromIn(conv, layerIn):
    n_in_h, n_in_w = layerIn[0]
    j_in_h, j_in_w = layerIn[1]
    r_in_h, r_in_w = layerIn[2]
    start_in_x, start_in_y = layerIn[3]
    k_h, k_w = conv[0]
    s_h, s_w = conv[1]
    p_h, p_w = conv[2]

    n_out_w = math.floor((n_in_w - k_w + 2*p_w)/s_w) + 1
    actualP_w = (n_out_w-1)*s_w - n_in_w + k_w 
    pR_w = math.ceil(actualP_w/2)
    pL_w = math.floor(actualP_w/2)
    
    n_out_h = math.floor((n_in_h - k_h + 2*p_h)/s_h) + 1
    actualP_h = (n_out_h-1)*s_h - n_in_h + k_h 
    pR_h = math.ceil(actualP_h/2)
    pL_h = math.floor(actualP_h/2)
    
    j_out_w = j_in_w * s_w
    j_out_h = j_in_h * s_h

    r_out_w = r_in_w + (k_w - 1)*j_in_w
    r_out_h = r_in_h + (k_h - 1)*j_in_h
    
    start_out_x = start_in_x + ((k_w-1)/2 - pL_w)*j_in_w
    start_out_y = start_in_y + ((k_h-1)/2 - pL_h)*j_in_h
    
    return (n_out_h, n_out_w), (j_out_h, j_out_w), (r_out_h, r_out_w), (start_out_x, start_out_y)
  
def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t feature map size: {} \n \t effective stride: {} \n \t receptive size: {} \t start: {} ".format(layer[0], layer[1], layer[2], layer[3]))
    print()  

def calculate_receiptive_field(imsize, convnet, layer_names):
    layerInfos = []
    #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and (start_0_x, start_0_y) = (0.5, 0.5)
    currentLayer = [imsize, (1, 1), (1, 1), (0.5, 0.5)]
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
    return layerInfos

def get_receptive_field_info(layerInfos, layer_names, layer_name, idx_x=0, idx_y=0):
    
    layer_idx = layer_names.index(layer_name)
    n_h, n_w = layerInfos[layer_idx][0]
    j_h, j_w = layerInfos[layer_idx][1]
    r_h, r_w = layerInfos[layer_idx][2]
    start_x, start_y = layerInfos[layer_idx][3]
    assert(idx_x < n_w)
    assert(idx_y < n_h)
    return (n_h, n_w), (r_h, r_w), (j_h, j_w), (start_x+idx_x*j_w, start_y+idx_y*j_h)


if __name__=="__main__":
    #contruct argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_path", required=True, help="path to input image", type=str)
    parser.add_argument("-s1", "--img_height", help="height of the input image", type=int, default=224)
    parser.add_argument("-s2", "--img_width", help="width of the input image", type=int, default=224)
    parser.add_argument("-l", "--layer_name", required=False, help="layer name to get receptive field for", default="down5_conv2")
    
    args = parser.parse_args()
    
    img_path = args.img_path
    imsize = (args.img_height, args.img_width)
    layer_name = args.layer_name
    
            

    #Define convnet as a list of layers, each layer define as 
    # [(kernel_height, kernel_width), (stride_vertical, stride_horizontal), (padding_vertical, padding_horizontal)]
    #unet
    unet = [[(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)], [(2, 2), (2, 2), (0, 0)], [(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)], [(2, 2), (2, 2), (0, 0)], [(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)], [(2, 2), (2, 2), (0, 0)], [(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)], [(2, 2), (2, 2), (0, 0)], [(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)], [(2, 2), (2, 2), (0, 0)], [(3, 3), (1, 1), (1, 1)], [(3, 3), (1, 1), (1, 1)]]
    unet_layers = ['inc1', 'inc2', 'down1_maxpool', 'down1_conv1', 'down1_conv2', 'down2_maxpool', 'down2_conv1', 'down2_conv2', 'down3_maxpool', 'down3_conv1', 'down3_conv2', 'down4_maxpool', 'down4_conv1', 'down4_conv2', 'down5_maxpool', 'down5_conv1', 'down5_conv2']
    net = unet
    layer_names = unet_layers
    

    layerInfos = calculate_receiptive_field(imsize, net, layer_names)

    
    #print details of each layer
    print("Layer info:")

    for i in range(len(layerInfos)):
        printLayer(layerInfos[i], layer_names[i])
        
    #get receptive field info for last conv layer
    feature_map_size, receptive_field_size, effective_stride, center= get_receptive_field_info(layerInfos, layer_names, layer_name, 0, 0)
   
    receptive_field = receptive_field_size
    img_height, img_width, = imsize[0], imsize[1]

    num_rows= feature_map_size[0]
    num_cols= feature_map_size[1]
    
    #visualize receptive field for layer name specified
    #read image and resize to given input size
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imsize[1], imsize[0]))

    im = plt.imshow(img)

    for i in range(num_rows):
        for j in range(num_cols):
            new_center = center[0]+effective_stride[1]*j, center[1]+effective_stride[0]*i

            x1 = new_center[0] -  receptive_field[1]/2
            y1 = new_center[1] -  receptive_field[0]/2

            x2 = new_center[0] +  receptive_field[1]/2
            y2 = new_center[1] +  receptive_field[0]/2

            x1 = int(x1 if x1>0 else 0)
            y1 = int(y1 if y1>0 else 0)

            x2 = int(x2 if x2 < img_width else img_width)
            y2 = int(y2 if y2 < img_height else img_height)

            pt1 = (x1, y1)
            pt2 = (x2, y2)
            color = (255, 0, 0)
            img_sliding_window = cv2.rectangle(img.copy(), pt1, pt2, color, 2)
            im.set_data(img_sliding_window)
            plt.axis('off')
            plt.pause(0.1)
            
    plt.show()           
            