#
# this module converts a floating point caffe model to the model conv.dat and fc.dat
# conv.dat -- run on TEE comput stick.
# fc.dat   -- runs on host device
# Note: the input model has to be trained as 1bit/3bits network.
#
# Run:
# python convt_cnnsvic.py teeNet1.caffemodel teeNet1.prototxt teeNet1.json test.jpg ./output
# Result is saved as conv.dat and fc.dat
#

# import modules:
import numpy as np
import sys
import struct

import os
os.environ['GLOG_minloglevel'] = '2'

import caffe
import cv2
import argparse
import os

def NetworkSurgery(net_module, net_weights, net_config, output_model_file):
    print("Load caffe model:")
    net = caffe.Net(net_module, net_weights, caffe.TEST)

    print("Parse JSON file")
    # parse JSON file
    with open(net_config, 'r') as f:
        inlines = f.readlines()
    
    # find sublayer numbers and coef bits
    sublayer_bits = np.zeros(100, int)   # upto 100 sublayers   
    sub = '"sublayer'
    coef = '"coef'
    cnt_sub = 0
    for line in inlines:
        ln = line.split()
        if any(sub in string for string in ln):
            sub_char =  line[line.index(':')+1:line.index(',')]
            sublayers = int(sub_char)
        if any(coef in string for string in ln):
            coef_char = line[line.index(':')+1:line.index('\n')]
            for kk in range(sublayers):
                sublayer_bits[cnt_sub] = int(coef_char)
                cnt_sub += 1

    print('parsed layer bits:'+ str(sublayer_bits[0:cnt_sub]) )

    count = -1
    for layer in net.params.keys():       
        if (layer.find('conv') == 0 or layer.find('conv') == 1) and count < cnt_sub-1:
            count += 1
            coef = net.params[layer][0].data
            for output_channel in range(coef.shape[0]):
                for input_channel in range(coef.shape[1]):                
                    coef3x3 = coef[output_channel][input_channel][...]
                    var = np.sum(abs(coef3x3))/9.0
                    
                    QFactor = 4.0/(var+0.0001)
                                     
                    if sublayer_bits[count] == 3:
                        for i in range(0,3):
                            for j in range(0,3):
                                abs_coefInt = int(abs(coef3x3[i,j]) * QFactor)
                                if abs_coefInt > 2:
                                    abs_coefInt = 4
                                abs_coef = abs_coefInt/QFactor;
                                if coef3x3[i,j] >= 0:
                                    coef3x3[i,j] = abs_coef
                                else:
                                    coef3x3[i,j] = -abs_coef
                    else:
                        coef3x3[coef3x3>=0] = var
                        coef3x3[coef3x3<0] = -var

    net.save(output_model_file)

def outputNetworkWithoutBackbone(net_model, net_weights, output_fc_Coef):
     # create net
    caffe.set_mode_cpu()
    net = caffe.Net(net_model, net_weights, caffe.TEST)
  
    # create output file
    fpout = open(output_fc_Coef, 'wb')
    # write headers
    
    fclayers = []
    for layer in net._layer_names:
        if layer[:2] == 'fc':
            fclayers.append(layer)
    print('Layer list:'+ str(fclayers) )
        
    for fcname in fclayers:
        inlen = net.params[fcname][0].data.shape[1]
        outlen = net.params[fcname][1].data.shape[0]
        fpout.write(struct.pack('<i', inlen))
        fpout.write(struct.pack('<i', outlen))
        
    for fcname in fclayers:
        fpout.write(net.params[fcname][0].data)
        fpout.write(net.params[fcname][1].data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights",  \
	                    default='teeNet1.caffemodel',  \
	                    help=".caffemodel file")
						
    parser.add_argument("--network",   \
	                    default='teeNet1.prototxt',   \
	                    help=".prototxt file")

    parser.add_argument("--config",   \
	                    default="teeNet1.json",   \
	                    help=".prototxt file")

    parser.add_argument("--testdata",   \
	                    default="test.jpg",   \
	                    help="an image file for test")

    parser.add_argument("--output",   \
	                    default=".",   \
	                    help="conversion output direction")
    args = parser.parse_args()

    img = cv2.imread(args.testdata)
    if img is None:
        print("Error: input image " + args.testdata + " not exist!")
        sys.exit()
        
    image = cv2.resize(img, (224, 224))
    with open('test.bin','wb') as f:
        f.write(image)
        f.close()

    if os.path.isfile("test.bin") == False:
        print("Error: File test.bin does not exist!")
        sys.exit()

    #step 1: get the sliced model
    output_sliced_model = args.output + '/network_sliced.caffemodel'
    
    NetworkSurgery(args.network, args.weights, args.config, output_sliced_model)

    #step 2: get the output_conv.dat
    if os.path.isfile(output_sliced_model) == False:
        print("Error: File %s does not exist!" % (output_sliced_model))
        sys.exit()

    command = './cnnconvt '
    command_param = args.network + ' ' + output_sliced_model + ' ' + args.config + ' test.bin ' + args.output
    print(command + command_param)
    os.system(command + command_param)

    #step 3: get the weight part without backbone network weights
    outputNetworkWithoutBackbone(args.network, args.weights, 'fc.dat')
    os.remove(output_sliced_model)
    os.remove("test.bin")
    print("Convert quantizated caffemodel to conv.dat and fc.dat sucessful!")





