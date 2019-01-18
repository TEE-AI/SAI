# * SAI: for release use 
# * SAI: Contains model training, on-chip file conversion
# * Support 3 kinds of Network: 1. teeNet1 -- Standard VGG (13 conv layers with 3 fc layers)
# *                             2. teeNet2 -- VGG18 (18 conv layers with GAP )
# *                             3. teeNet3 -- VGG layers without fc (16 conv layers)
#*********************************************************************************************

import json
import os
import sys
import torch

SAI_ROOT=os.getcwd()
sys.path.append(os.path.join(SAI_ROOT,"models"))

from modelTraining import modelTraining
from modelConversion import modelConversion

class SAI():
    def __init__(self, num_classes=2, data_dir=os.path.join(SAI_ROOT,'data'), checkpoint_dir=os.path.join(SAI_ROOT,'checkpoint'), model_type=3, module_type=0, device_type=0):

        ## Model training parameters [default]
        self.num_classes = num_classes
        if self.num_classes <=5:   
           model_type_desired = 0 
           self.model_type = 3                 
        elif self.num_classes <=20:
           model_type_desired = 2
        else: 
           model_type_desired = 1

        if model_type > 3:               # 1: teeNet1,  2: teeNet2,  3: teeNet3
           self.model_type = model_type_desired
        else:
           self.model_type = model_type

        if self.model_type == 3:
           self.net_arch = 'teeNet3'
           self.mask_bits = '3,3,1,1,1,1'
           self.act_bits = '5,5,5,5,5,5'
        elif self.model_type == 2:
           self.net_arch = 'teeNet2'
           self.mask_bits = '3,3,3,3,1'
           self.act_bits = '5,5,5,5,5'
        elif self.model_type == 1:
           self.net_arch = 'teeNet1'
           self.mask_bits = '3,3,1,1,1'
           self.act_bits = '5,5,5,5,5'
        else:
           sys.exit("Only support teeNet3, teeNet2, teeNet1! 'model_type' should be within 0 to 2")   
     
        # 0 : Type 1 arch, Conv(w/o bias) + bn + bias# 1 : Type 2 arch, Conv(w/ bias) + bn
        self.module_type=module_type

        self.max_epoch = 160
        self.learning_rate = 5e-2
        self.train_batch_size = 48
        self.test_batch_size = 24  
        self.resume = False
        self.finetune= False
        self.full = False
      
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = 'model_%s.tee' % self.net_arch

        self.device_type = device_type  # ftdi: 0, emmc: 1
        if device_type==0:
           self.device_name = '0'
        else: 
           self.device_name = '/dev/sg3'
       
        return

    def teeModelTraining(self, train_params_json):
        
        if os.path.isfile(train_params_json):       
           with open(train_params_json) as trainParamsJson:
              parsed_params = json.load(trainParamsJson)
           if 'num_classes' in parsed_params:
             self.num_classes = parsed_params['num_classes']
           if 'max_epoch' in parsed_params:
             self.max_epoch = parsed_params['max_epoch']        
           if 'learning_rate' in parsed_params:
             self.learning_rate = parsed_params['learning_rate']
           if 'train_batch_size' in parsed_params:
             self.train_batch_size = parsed_params['train_batch_size']
           if 'test_batch_size' in parsed_params:
             self.test_batch_size = parsed_params['test_batch_size']
           if 'mask_bits' in parsed_params:
            self.mask_bits = parsed_params['mask_bits']
           if 'act_bits' in parsed_params:
            self.act_bits= parsed_params['act_bits']
           if 'resume' in parsed_params:
            self.resume = parsed_params['resume']
           if 'finetune' in parsed_params:
            self.finetune= parsed_params['finetune']
           if 'full' in parsed_params:
            self.full = parsed_params['full']

        modelTraining(self.model_type, self.module_type, self.train_batch_size, self.test_batch_size, 
                      self.data_dir, self.mask_bits, self.act_bits, self.num_classes, 
                      self. max_epoch, self.learning_rate, 
                      self.resume, self.finetune, self.full)
        
    def teeModelConversion(self,net_dir):
        netJson_in = os.path.join(net_dir, str(self.net_arch) + '.json')
        fc_out = 'fc.dat'
        conv_out = 'conv.dat'
        modelConversion(self.model_type, self.module_type, self.mask_bits, self.checkpoint_dir, self.checkpoint, netJson_in, conv_out, fc_out)    

if __name__=='__main__':

   data_dir = os.path.join(SAI_ROOT,'data')
   checkpoint_dir = os.path.join(SAI_ROOT,'checkpoint')

   tee_SAI = SAI(num_classes=2, data_dir=data_dir, checkpoint_dir=checkpoint_dir, model_type=1, module_type=1, device_type=1)
   
   tee_SAI.teeModelTraining('training.json')
   
   maskLayers = [int(item) for item in tee_SAI.mask_bits.split(',')]
   # Check if it is a floating model
   for mask_bit in maskLayers:
       if mask_bit ==0 or mask_bit>3:
          sys.exit( "This seems to be a pure floating model. 1, 2, or 3-bit quantization is required")
   
   net_dir = os.path.join(SAI_ROOT,'nets')
   tee_SAI.teeModelConversion(net_dir)

