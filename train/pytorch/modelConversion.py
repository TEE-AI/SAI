import os
import sys
import struct
import numpy as np
import torch
from models import BinarizeFn, TernarizeFn, Threebits

from ctypes import *
import platform

if(platform.system() == "Linux"):
    dllFilename=os.path.join(os.path.dirname(__file__) + '/lib', "libTEEConvertor.so")
    teeConvertor = CDLL(dllFilename)
else:
    dllFilename=os.path.join(os.path.dirname(__file__) + '/lib', "libTEEConvertor.dll")
    teeConvertor = CDLL(dllFilename)

teeConvertor.TeeConvertNN.argtypes = (c_char_p,c_char_p,c_char_p,c_char_p)

def netSlicingBatchnormAbsorptionType1(input_checkpoint, output_checkpoint, mask_bits):

    print('==> Slicing network..')
    #checkpoint = torch.load(input_checkpoint,map_location=lambda storage, loc: storage)
    checkpoint = torch.load(input_checkpoint)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    net_state = net.state_dict()
    mask_params = [p for p in net_state if 'mask_val' in p]

    maskLayers = [int(item) for item in mask_bits.split(',')]
    mask_num = len(maskLayers)
    layer_num = 0 
    for mask in maskLayers:
        assert mask >= 0, "Only support 1, 2, 3 bit masks"
    
    pre_layer = []
    for mask_layer in mask_params:
       cur_layer = mask_layer.split('.')[0]
       if cur_layer != pre_layer: 
          layer_num +=1

       mask = dict(net_state)[mask_layer]
       mask_layer_bit = maskLayers[layer_num-1]
       #print(layer_num, mask_layer, mask_layer_bit, cur_layer, pre_layer) 
       if mask_layer_bit==1:
          dict(net_state)[mask_layer].data.copy_(BinarizeFn.apply(mask.data))    # 1-bit
       elif mask_layer_bit==2:
          dict(net_state)[mask_layer].data.copy_(TernarizeFn.apply(mask.data))   # 2-bit
       elif mask_layer_bit==3:
          dict(net_state)[mask_layer].data.copy_(Threebits.apply(mask.data))     # 3-bit
       else:
          sys.exit("Only support 1-bit and 3-bit quantization in filter weights!") 
       
       pre_layer = cur_layer
    assert layer_num==mask_num, "mask_bit length does not match!"    

    '''    
    # For debug
    state = {
        'net': net,
        'acc': best_acc,
        'epoch': start_epoch,
    }
    torch.save(state, os.path.join("checkpoint", 'teeNet_sliced_debug.t7'))
    '''
    print('==> Absorbing batch normalization..')
    parameters = net.named_parameters
    quant_params = [p[0] for p in parameters() if '.scale_coef' in p[0]]
    mask_params = [p[0] for p in parameters() if '.mask_val' in p[0]]
    coef_params = [p[0] for p in parameters() if '.coef' in p[0]]
    bias_params = [p[0] for p in parameters() if '.bias' in p[0]]
    fc_params = [p[0] for p in parameters() if 'classifier' in p[0]]
    parametersDict = dict(parameters())
    
    ''' Copy scale_coef (Relu alpha) '''
    for quant_layer in quant_params:
    
        scale_coef = dict(net_state)[quant_layer]
        parametersDict[quant_layer].data.copy_(scale_coef)
    
    ''' Copy mask '''
    for mask_layer in mask_params:
    
        mask = dict(net_state)[mask_layer]
        parametersDict[mask_layer].data.copy_(mask)
        
    
    ''' Modify coef  '''
    for coef_layer in coef_params:
        
        bn_runningmean_layer = coef_layer[:-9] + 'bn.running_mean'
        bn_runningmean = dict(net_state)[bn_runningmean_layer]
        bn_runningvar_layer = coef_layer[:-9] + 'bn.running_var'
        bn_runningvar = dict(net_state)[bn_runningvar_layer]
        
        invstd = bn_runningvar.clone().add_(np.finfo(np.float).eps).pow_(-0.5)
        coef = dict(net_state)[coef_layer]
        coef.mul_(invstd.view(coef.size(0), 1, 1, 1).expand_as(coef))
        
        parametersDict[coef_layer].data.copy_(coef)
        
    
    ''' Modify bias  '''
    for bias_layer in bias_params:
        if 'classifier' in bias_layer:
            continue
        
        bn_runningmean_layer = bias_layer[:-4] + 'bn.running_mean'
        bn_runningmean = dict(net_state)[bn_runningmean_layer]
        bn_runningvar_layer = bias_layer[:-4] + 'bn.running_var'
        bn_runningvar = dict(net_state)[bn_runningvar_layer]
        
        invstd = bn_runningvar.clone().add_(np.finfo(np.float).eps).pow_(-0.5)
        bias = dict(net_state)[bias_layer]
        
        bias.add_(-bn_runningmean.view(bias.shape).mul_(invstd.view(bias.shape))) ## "Correct" one
        parametersDict[bias_layer].data.copy_(bias)
    
    ''' Copy fc layer '''
    for fc_layer in fc_params: 
        fc_coef= dict(net_state)[fc_layer]
        parametersDict[fc_layer].data.copy_(fc_coef)
    
    state = {
        'net': net,
        'acc': best_acc,
        'epoch': start_epoch,
    }
    torch.save(state, output_checkpoint)


def netSurgeryDumpType1(arch, input_checkpoint, filter_path, bias_path, fc_path):
        
    #checkpoint = torch.load(input_checkpoint,map_location='cpu')
    checkpoint = torch.load(input_checkpoint)
    net = checkpoint['net']
    
    parameters = net.named_parameters
    quant_params = [p[0] for p in parameters() if '.scale_coef' in p[0]]
    mask_params = [p[0] for p in parameters() if '.mask_val' in p[0]]
    coef_params = [p[0] for p in parameters() if '.coef' in p[0]]
    bias_params = [p[0] for p in parameters() if '.bias' in p[0]]
    parametersDict = dict(parameters())
    
    quant_gain = np.array([])
    for quant_layer in quant_params:
        quant = parametersDict[quant_layer].data.cpu()
        quant_gain = np.concatenate((quant_gain, np.array(quant.data)), axis=0)

    print('==> Merging gain..')
    max_ReLu = 31.5
    layer_count = 0
    layer_num = len(quant_gain)
    for layer_count in range(layer_num):
        
        if layer_count == 0:
           gain_weight = max_ReLu/quant_gain[layer_count]
        else:
           gain_weight = quant_gain[layer_count-1]/quant_gain[layer_count]
         
        gain_bias = max_ReLu/quant_gain[layer_count]
    
        coef_layer = coef_params[layer_count]
        bias_layer = bias_params[layer_count]
        quant_layer = quant_params[layer_count]
    
        parametersDict[coef_layer].data.mul_(gain_weight) 
        parametersDict[bias_layer].data.mul_(gain_bias)    
        parametersDict[quant_layer].data.mul_(gain_bias)      

    if arch != 'teeNet3': 
       fc_params = [p[0] for p in parameters() if 'classifier' in p[0]]
       for fc_layer in fc_params:
           if '.bias' in fc_layer:
              parametersDict[fc_layer].data.mul_(gain_bias)      
   
    print('==> Dumping CNN parameters into .txt files.. ')
    flt = np.array([])
    bias = np.array([])
    
    layer_num = 0 
    outch_pre = 0
    
    if arch == 'teeNet3':
       last_major_outch = 256
       last_major_layerIdx = 14
    elif arch == 'teeNet2':
       last_major_outch = 512
       last_major_layerIdx = 15
    elif arch == 'teeNet1':
       last_major_outch = 512
       last_major_layerIdx = 11

    for mask_layer in mask_params:
        layer_num += 1
        coef_layer = mask_layer[:-8] + 'coef'
        mask = parametersDict[mask_layer].data
        coef = parametersDict[coef_layer].data
        weight = mask*coef
        ## Network surgery if necessary
        (outch_old, inch_old, kh, kw) = weight.size() 
        if layer_num < last_major_layerIdx:
           updated_weight = weight
           outch_new = outch_old 
        else: 
           outch_new = last_major_outch
           inch_new = outch_pre  # input ch# equals the output ch# of the previous layer
           
           if outch_old == outch_new:
              updated_weight = weight
           else:
              print('    TEE Model %s: output channel of layer %d extended from %d to %d' % (arch, layer_num, outch_old, outch_new))
              updated_weight = torch.FloatTensor(np.zeros((outch_new, inch_new, kh, kw)))
              for i in range(0, outch_new):
                if i < outch_old:
                  for j in range(0, inch_new):
                    if j < inch_old:
                      updated_weight[i][j][...] = weight[i][j][...]
                    else:
                      updated_weight[i][j][...] = torch.FloatTensor(np.zeros(weight[0][0].size()))
                else:
                  for j in range(0, inch_new):
                    updated_weight[i][j][...] = torch.FloatTensor(np.zeros(weight[0][0].size())) 
        
        outch_pre = outch_new   
        updated_weight = np.array(updated_weight.data.cpu()).flatten()
        flt = np.concatenate((flt, updated_weight), axis=0)
    
    layer_num = 0
    for bias_layer in bias_params:
        if 'classifier' in bias_layer:
            continue

        layer_num += 1
        bias_vec = parametersDict[bias_layer].data.cpu()
        outch_old = len(np.array(bias_vec).flatten())
        if layer_num < last_major_layerIdx:
           updated_bias = bias_vec 
        else: 
           outch_new = last_major_outch
          
           if outch_old == outch_new:
              updated_bias = bias_vec
           else: 
              updated_bias = torch.FloatTensor(np.zeros((outch_new,))) 
              for i in range(0, outch_new):
                if i < outch_old:
                  updated_bias[i] = bias_vec[0][i][0][0]
                else:
                  updated_bias[i] = torch.FloatTensor(np.zeros(bias_vec[0][0][0][0].size()))      
        updated_bias = np.array(updated_bias.data).flatten() 
        bias = np.concatenate((bias, updated_bias), axis=0)
    
    np.savetxt(filter_path, flt, fmt = '%.16e', delimiter = '\n')
    np.savetxt(bias_path, bias, fmt = '%.16e', delimiter = '\n')
    
    if arch != 'teeNet3': 
       print('==> Dumping FC parameters into .bin file.. ')
       fpout = open(fc_path, 'wb')
       for p in parameters():
           if 'classifier' not in p[0]:
              continue 
           if 'weight' in p[0]:
              (outlen, inlen) = parametersDict[p[0]].shape
              fpout.write(struct.pack('<i', inlen))
              fpout.write(struct.pack('<i', outlen))
              #print('Layer', p[0], 'in:', inlen, 'out:', outlen)
    
       for p in parameters():
           if 'classifier' not in p[0]:
              continue
           fpout.write(parametersDict[p[0]].cpu().detach().numpy())
    
       fpout.close()



def netSlicingBatchnormAbsorptionType2(input_checkpoint, output_checkpoint, mask_bits):

    print('==> Slicing network..')
    #checkpoint = torch.load(input_checkpoint,map_location='cpu')
    checkpoint = torch.load(input_checkpoint)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    net_state = net.state_dict()
    mask_params = [p for p in net_state if 'mask_val' in p]

    maskLayers = [int(item) for item in mask_bits.split(',')]
    mask_num = len(maskLayers)
    layer_num = 0 
    for mask in maskLayers:
        assert mask >= 0, "Only support 1, 2, 3 bit masks"
    
    pre_layer = []
    for mask_layer in mask_params:
       cur_layer = mask_layer.split('.')[0]
       if cur_layer != pre_layer: 
          layer_num +=1

       mask = dict(net_state)[mask_layer]
       mask_layer_bit = maskLayers[layer_num-1]
       #print(layer_num, mask_layer, mask_layer_bit, cur_layer, pre_layer) 
       if mask_layer_bit==1:
          dict(net_state)[mask_layer].data.copy_(BinarizeFn.apply(mask.data))    # 1-bit
       elif mask_layer_bit==2:
          dict(net_state)[mask_layer].data.copy_(TernarizeFn.apply(mask.data))   # 2-bit
       elif mask_layer_bit==3:
          dict(net_state)[mask_layer].data.copy_(Threebits.apply(mask.data))     # 3-bit
       else:
          sys.exit("Only support 1-bit and 3-bit quantization in filter weights!") 
       
       pre_layer = cur_layer
    assert layer_num==mask_num, "mask_bit length does not match!"    

    print('==> Absorbing batch normalization..')
    parameters = net.named_parameters
    quant_params = [p[0] for p in parameters() if '.scale_coef' in p[0]]
    mask_params = [p[0] for p in parameters() if '.mask_val' in p[0]]
    coef_params = [p[0] for p in parameters() if '.coef' in p[0]]
    bias_params = [p[0] for p in parameters() if 'conv.bias' in p[0]]
    fc_params = [p[0] for p in parameters() if 'classifier' in p[0]]
    parametersDict = dict(parameters())
    
    ''' Copy scale_coef (Relu alpha) '''
    for quant_layer in quant_params:
    
        scale_coef = dict(net_state)[quant_layer]
        parametersDict[quant_layer].data.copy_(scale_coef)
    
    ''' Copy mask '''
    for mask_layer in mask_params:
    
        mask = dict(net_state)[mask_layer]
        parametersDict[mask_layer].data.copy_(mask)
        
    
    ''' Modify coef  '''
    for coef_layer in coef_params:
        
        bn_runningvar_layer = coef_layer[:-9] + 'bn.running_var'
        bn_runningvar = dict(net_state)[bn_runningvar_layer]
        invstd = bn_runningvar.clone().add_(np.finfo(np.float).eps).pow_(-0.5)
        
        bn_weight_layer = coef_layer[:-9] + 'bn.weight'
        bn_weight = dict(net_state)[bn_weight_layer]  # gamma 
        gamma_invstd =invstd*bn_weight 

        coef = dict(net_state)[coef_layer]
        coef.mul_(gamma_invstd.view(coef.size(0), 1, 1, 1).expand_as(coef))
        
        parametersDict[coef_layer].data.copy_(coef)
        
    
    ''' Modify bias  '''
    for bias_layer in bias_params:
        if 'classifier' in bias_layer:
            continue
        bn_runningmean_layer = bias_layer[:-9] + 'bn.running_mean'
        bn_runningmean = dict(net_state)[bn_runningmean_layer]
        bn_runningvar_layer = bias_layer[:-9] + 'bn.running_var'
        bn_runningvar = dict(net_state)[bn_runningvar_layer]
        invstd = bn_runningvar.clone().add_(np.finfo(np.float).eps).pow_(-0.5)
        
        bn_weight_layer = bias_layer[:-9] + 'bn.weight'
        bn_weight = dict(net_state)[bn_weight_layer]  # gamma 
        bn_bias_layer = bias_layer[:-9] + 'bn.bias'
        bn_bias = dict(net_state)[bn_bias_layer]      # beta 

        bias = dict(net_state)[bias_layer]
        gamma_invstd =invstd*bn_weight 
        gamma_invstd_mean = -bn_runningmean*gamma_invstd
        #print(bias_layer, bn_weight.shape, invstd.shape, gamma_invstd.shape)

        bias.mul_(gamma_invstd.view(bias.shape)).add_(bn_bias).add_(gamma_invstd_mean.view(bias.shape))
        parametersDict[bias_layer].data.copy_(bias)
    
    ''' Copy fc layer '''
    for fc_layer in fc_params: 
        fc_coef= dict(net_state)[fc_layer]
        parametersDict[fc_layer].data.copy_(fc_coef)
    
    state = {
        'net': net,
        'acc': best_acc,
        'epoch': start_epoch,
    }
    torch.save(state, output_checkpoint)


def netSurgeryDumpType2(arch, input_checkpoint, filter_path, bias_path, fc_path):
        
    #checkpoint = torch.load(input_checkpoint,map_location='cpu')
    checkpoint = torch.load(input_checkpoint)
    net = checkpoint['net']
    
    parameters = net.named_parameters
    quant_params = [p[0] for p in parameters() if '.scale_coef' in p[0]]
    mask_params = [p[0] for p in parameters() if '.mask_val' in p[0]]
    coef_params = [p[0] for p in parameters() if '.coef' in p[0]]
    bias_params = [p[0] for p in parameters() if 'conv.bias' in p[0]]
    parametersDict = dict(parameters())

    quant_gain = np.array([])
    for quant_layer in quant_params:
        quant = parametersDict[quant_layer].data.cpu()
        quant_gain = np.concatenate((quant_gain, np.array(quant.data)), axis=0)

    print('==> Merging gain..')
    max_ReLu = 31.5
    layer_count = 0
    layer_num = len(quant_gain)
    for layer_count in range(layer_num):
        
        if layer_count == 0:
           gain_weight = max_ReLu/quant_gain[layer_count]
        else:
           gain_weight = quant_gain[layer_count-1]/quant_gain[layer_count]
         
        gain_bias = max_ReLu/quant_gain[layer_count]
    
        coef_layer = coef_params[layer_count]
        bias_layer = bias_params[layer_count]
        quant_layer = quant_params[layer_count]
    
        parametersDict[coef_layer].data.mul_(gain_weight) 
        parametersDict[bias_layer].data.mul_(gain_bias)    
        parametersDict[quant_layer].data.mul_(gain_bias)      

    if arch != 'teeNet3': 
       fc_params = [p[0] for p in parameters() if 'classifier' in p[0]]
       for fc_layer in fc_params:
           if '.bias' in fc_layer:
              parametersDict[fc_layer].data.mul_(gain_bias)      
   
    print('==> Dumping CNN parameters into .txt files.. ')
    flt = np.array([])
    bias = np.array([])
    
    layer_num = 0 
    outch_pre = 0
    
    if arch == 'teeNet3':
       last_major_outch = 256
       last_major_layerIdx = 14
    elif arch == 'teeNet2':
       last_major_outch = 512
       last_major_layerIdx = 15
    elif arch == 'teeNet1':
       last_major_outch = 512
       last_major_layerIdx = 11

    for mask_layer in mask_params:
        layer_num += 1
        coef_layer = mask_layer[:-8] + 'coef'
        mask = parametersDict[mask_layer].data
        coef = parametersDict[coef_layer].data
        weight = mask*coef
        ## Network surgery if necessary
        (outch_old, inch_old, kh, kw) = weight.size() 
        if layer_num < last_major_layerIdx:
           updated_weight = weight
           outch_new = outch_old 
        else: 
           outch_new = last_major_outch
           inch_new = outch_pre  # input ch# equals the output ch# of the previous layer
           
           if outch_old == outch_new:
              updated_weight = weight
           else:
              print('    TEE Model %s: output channel of layer %d extended from %d to %d' % (arch, layer_num, outch_old, outch_new))
              updated_weight = torch.FloatTensor(np.zeros((outch_new, inch_new, kh, kw)))
              for i in range(0, outch_new):
                if i < outch_old:
                  for j in range(0, inch_new):
                    if j < inch_old:
                      updated_weight[i][j][...] = weight[i][j][...]
                    else:
                      updated_weight[i][j][...] = torch.FloatTensor(np.zeros(weight[0][0].size()))
                else:
                  for j in range(0, inch_new):
                    updated_weight[i][j][...] = torch.FloatTensor(np.zeros(weight[0][0].size())) 
        
        outch_pre = outch_new   
        updated_weight = np.array(updated_weight.data.cpu()).flatten()
        flt = np.concatenate((flt, updated_weight), axis=0)
    
    layer_num = 0
    for bias_layer in bias_params:
        if 'classifier' in bias_layer:
            continue

        layer_num += 1
        bias_vec = parametersDict[bias_layer].data.cpu()
        outch_old = len(np.array(bias_vec).flatten())
        if layer_num < last_major_layerIdx:
           updated_bias = bias_vec 
        else: 
           outch_new = last_major_outch
          
           if outch_old == outch_new:
              updated_bias = bias_vec
           else: 
              updated_bias = torch.FloatTensor(np.zeros((outch_new,))) 
              for i in range(0, outch_new):
                if i < outch_old:
                  updated_bias[i] = bias_vec[i]
                else:
                  updated_bias[i] = torch.FloatTensor(np.zeros(bias_vec[0].size()))      
        updated_bias = np.array(updated_bias.data).flatten() 
        bias = np.concatenate((bias, updated_bias), axis=0)
    
    np.savetxt(filter_path, flt, fmt = '%.16e', delimiter = '\n')
    np.savetxt(bias_path, bias, fmt = '%.16e', delimiter = '\n')
    
    if arch != 'teeNet3': 
       print('==> Dumping FC parameters into .bin file.. ')
       fpout = open(fc_path, 'wb')
       for p in parameters():
           if 'classifier' not in p[0]:
              continue 
           if 'weight' in p[0]:
              (outlen, inlen) = parametersDict[p[0]].shape
              fpout.write(struct.pack('<i', inlen))
              fpout.write(struct.pack('<i', outlen))
              #print('Layer', p[0], 'in:', inlen, 'out:', outlen)
    
       for p in parameters():
           if 'classifier' not in p[0]:
              continue
           fpout.write(parametersDict[p[0]].cpu().detach().numpy())
    
       fpout.close()



def modelConversion(model_type, module_type, mask_bits, checkpoint_dir, checkpoint_filename = 'model_teeNet1.tee', netJson_in='netJson_teeNet1.json', conv_out ='conv.dat',fc_out = 'fc.dat'):
    
    arch = 'teeNet3' 
    if model_type == 3:
       arch = 'teeNet3'
    elif model_type == 2:
       arch = 'teeNet2'
    elif model_type == 1:
       arch = 'teeNet1'
    else:
       sys.exit("Only support teeNet3, teeNet2, teeNet1! 'model_type' should be within 0 to 2")   
    
    checkpoint_origin = os.path.join(checkpoint_dir , checkpoint_filename)
    checkpoint_mid = checkpoint_origin[:-3] + '_mid.tee'
    
    filter_out = str('filter.txt')
    bias_out = str('bias.txt')

    if module_type==0: # Type 1 arch: Conv(w/o bias) + bn + bias
       # Slice the network and absorbe batchnorm layer
       netSlicingBatchnormAbsorptionType1(checkpoint_origin, checkpoint_mid, mask_bits)
       # Network gain editing and dump filter/bias and fc parameters
       netSurgeryDumpType1(arch, checkpoint_mid, filter_out, bias_out, fc_out)
    elif module_type==1: # Type 2 arch: Conv(w/ bias) + bn
       netSlicingBatchnormAbsorptionType2(checkpoint_origin, checkpoint_mid, mask_bits)
       netSurgeryDumpType2(arch, checkpoint_mid, filter_out, bias_out, fc_out)
    
    teeConvertor.TeeConvertNN(netJson_in.encode('ascii'),filter_out.encode('ascii'),bias_out.encode('ascii'),conv_out.encode('ascii'))
    os.remove(filter_out)
    os.remove(bias_out)