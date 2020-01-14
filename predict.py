import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
import random
from utils import load_checkpoint, load_cat_names
def parse_args():
    par = argparse.ArgumentParser()
    par.add_argument('checkpoint', action='store', default='checkpoint.pth')
    par.add_argument('--top_k', dest='top_k', default='3')
    par.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg')  
    par.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    par.add_argument('--gpu', action='store', default='gpu')
    return par.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image) 
   
    adj = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = adj(img)
    
    return image

def predict(image_path, model, topk=3, gpu='gpu'):
    ''' Get probability values (indeces) and respective flower classes. 
    '''
    # TODO: Implement the code to predict the class from an image file
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    prob = F.softmax(output.data,dim=1) 
    
    probs = np.array(prob.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()} # from reviewer advice
    top_classes = [np.int(index_to_class[each]) for each in np.array(prob.topk(topk)[1][0])]
    
    return probs, top_classes

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    print(labels)
    print(probability)
    
    i=0 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 

if __name__ == "__main__":
    main()