'''Train Quantized Networks with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import sys
import random
from models import *
from torch.autograd import Variable
from torch.optim import lr_scheduler
from shutil import copyfile

def dataPreparation(dir_input, dir_output):
    
   print('==> Preparing data..')

   #print('Calculate image count of each class')
   img_count_dict = {}
   folder_list = [os.path.join(dir_input, folder) for folder in os.listdir(dir_input) if os.path.isdir(os.path.join(dir_input, folder))]
   for folder in folder_list:
       imgs = [img for img in os.listdir(folder) if (os.path.isfile(os.path.join(folder, img)) and img != 'train' and img != 'val' and img != 'test')]
       img_count = len(imgs)
       cls_name = os.path.split(folder)[-1].upper()
       img_count_dict[folder] = img_count 
   
   min_img_count = min(img_count_dict.values()) # least img count of all classes
   
   # check if output folder exist 
   dir_output_train = os.path.join(dir_output, 'train')
   dir_output_val = os.path.join(dir_output, 'val')
   if (not os.path.exists(dir_output_train)) and (not os.path.exists(dir_output_val)):
       #print 'Create data folders'
       cls_name = os.path.split(folder)[-1].upper()
       os.system('mkdir ' + dir_output_train)
       os.system('mkdir ' + dir_output_val)
       for folder in img_count_dict.keys():
           cls_name = os.path.split(folder)[-1].upper()
           os.system('mkdir ' + os.path.join(dir_output_train, cls_name))
           os.system('mkdir ' + os.path.join(dir_output_val, cls_name))
   
       #print('Prepare data for training and validation data')                  	 
       for folder in img_count_dict.keys():
           cls_name = os.path.split(folder)[-1].upper()
           imgs = [os.path.join(folder, img) for img in os.listdir(folder) if os.path.isfile(os.path.join(folder, img))] 
           random.shuffle(imgs)
           for idx, img_path in enumerate(imgs):
               if idx < min_img_count:
                   if idx%10!=0:
                       copyfile(img_path, os.path.join(dir_output_train, cls_name, os.path.split(img_path)[-1]))
                       #os.system('cp '+img_path+' '+os.path.join(dir_output_train, cls_name))                                                                 
                   if idx%10==0:                              
                       copyfile(img_path, os.path.join(dir_output_val, cls_name, os.path.split(img_path)[-1]))    
                       #os.system('cp '+img_path+' '+os.path.join(dir_output_val, cls_name))    
			       
               else:
           	       #os.system('cp '+img_path+' '+os.path.join(dir_output_train, cls_name))    
           	       copyfile(img_path, os.path.join(dir_output_train, cls_name, os.path.split(img_path)[-1]))    
     
   else:
       print('Training and validation data exist already')

   
   # Generate a pic_label.txt file
   valdir = os.path.join(dir_input, 'val')
   classes = [d for d in os.listdir(valdir) if os.path.isdir(os.path.join(valdir, d))]
   #classes.sort(key=str.lower)
   classes.sort()
   class_to_idx = {classes[i]: i for i in range(len(classes))}
   #print(classes, class_to_idx, class_to_idx[classes[0]])

   labelFile = os.path.join(dir_input, 'pic_label.txt')
   label_file = open(labelFile, "w")
   for subfolder in classes:
       class_idx = class_to_idx[subfolder]
       label_file.write('{0} {1}\n'.format(subfolder, class_idx))

   return len(classes)

def dataLoader(data_dir, num_classes=2, batch_size=48, batch_size_test=24, num_workers=1):

   transform_train = transforms.Compose([
       transforms.Resize((256,256)),
       transforms.RandomCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
   ])
   
   transform_test = transforms.Compose([
       transforms.Resize((256,256)),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
   ])
   
   traindir = os.path.join(data_dir, 'train')
   valdir = os.path.join(data_dir, 'val')
   trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
   testset = torchvision.datasets.ImageFolder(valdir, transform_test)
   
   trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)
   testloader = torch.utils.data.DataLoader(testset, batch_size_test, shuffle=False, num_workers=num_workers)
   
   return trainloader, testloader

#copy weights from net2 into net1, assumes finetune path is a model trained using full precision
def finetune(net1, net2):
    for x in net1.state_dict():
        if x in net2.state_dict().keys() and dict(net1.state_dict())[x].data.numel() == dict(net2.state_dict())[x].data.numel():
            dict(net1.state_dict())[x].data.copy_(dict(net2.state_dict())[x])
        else:
            totalparams = dict(net1.state_dict())

            if 'mask_val' in x:
                float_layer_name = x[:-8] + 'weight'
                mask = dict(net2.named_parameters())[float_layer_name]
                totalparams[x].data.copy_(Threebits.apply(mask.data))
            elif 'conv.coef' in x:
                float_layer_name = x[:-4] + 'weight'
                weight = dict(net2.named_parameters())[float_layer_name]
                out_counter = 0
                for in_channel_float in weight:
                    in_counter = 0
                    for out_channel_float in in_channel_float:
                        totalparams[x].data[out_counter][in_counter][0][0] = out_channel_float.abs().mean()
                        in_counter += 1
                    out_counter += 1
    return net1

def netLoaderFinetune(all_models, arch, finetune_path, module_type, num_classes, maskLayers, quantLayers):

    print('==> Finetuning from checkpoint.. %s' % finetune_path)
    assert os.path.isfile(finetune_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(finetune_path)
    cls = all_models[arch]
    net = cls(module_type=module_type, num_classes=num_classes, mask_bits=maskLayers, act_bits=quantLayers)
    net = finetune(net, checkpoint['net'])

    return net

def netLoaderResume(path):

    print('==> Resuming from checkpoint.. %s' % path)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch

def netLoaderScratch(all_models, arch, module_type, num_classes, maskLayers, quantLayers, full):

    print('==> Building model from scratch..')
    cls = all_models[arch]
    net = cls(module_type=module_type, num_classes=num_classes, mask_bits=maskLayers, act_bits=quantLayers)

    return net

def train(net, criterion, epoch, optimizer, trainloader, use_cuda):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

def test(net, criterion, epoch, testloader, use_cuda, path, best_acc):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)

    if acc >= best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, path)
        print('* Saved checkpoint to %s' % path)
        best_acc = acc
    return best_acc

def modelTraining(model_type, module_type, batch_size, batch_size_test, data_dir, mask_bits="1,1,1,1,1", act_bits="5,5,5,5,5", num_classes=2, max_epoch =160, lr=0.01, resume=False, finetune=False, full=False):    
    
    maskLayers = [int(item) for item in mask_bits.split(',')]
    quantLayers = [int(item) for item in act_bits.split(',')]

    for mask in maskLayers:
        assert mask >= 0, "Only support 1, 2, 3 bit masks!"
        if mask != 0:
           full = False
    for quant in quantLayers:
        assert quant >= 0, "Activation must be positive!"
        if quant != 0:
           full = False
    
    dataPreparation(data_dir, data_dir)
    trainloader, testloader = dataLoader(data_dir, num_classes, batch_size, batch_size_test, num_workers=1)

    if model_type == 3:
       arch = 'teeNet3'
    elif model_type == 2:
       arch = 'teeNet2'
    elif model_type == 1:
       arch = 'teeNet1'

    suffix = 'model'
    suffix += '_%s' % arch
    path = './checkpoint/%s.tee' % suffix

    finetune_path = path  # By default

    all_models = {
        'teeNet3': TEE_VGG16_NoFC,
        'teeNet2': TEE_VGG18,
        'teeNet1': TEE_VGG16,
    }

    best_acc = 0     # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if finetune:
       net = netLoaderFinetune(all_models, arch, finetune_path, module_type, num_classes, maskLayers, quantLayers)
    elif resume:
       net, best_acc, start_epoch = netLoaderResume(path)
    else:
       net = netLoaderScratch(all_models, arch, module_type, num_classes, maskLayers, quantLayers, full)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        #print( "number of gpu: " + str(torch.cuda.device_count()))
    
    criterion = nn.CrossEntropyLoss()
    parameters = net.named_parameters
    mask_params = [p for p in parameters() if 'mask_val' in p[0]]
    other_params = [p for p in parameters() if p not in mask_params]

    optimizer = optim.SGD(
        [{'params':[p[1] for p in mask_params], 'weight_decay':0},
         {'params':[p[1] for p in other_params]}
        ],
        lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, max_epoch):
        if epoch % 25 == 0 or resume:
           scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 25)
        train(net, criterion, epoch, optimizer, trainloader, use_cuda)
        scheduler.step()
        best_acc = test(net, criterion, epoch, testloader, use_cuda, path, best_acc)
