"""
Created on Sun Oct 30 22:15:35 2022

@author: -G-
"""

import torch;
import torch.nn as nn;
import torch.optim as optim;
import torchvision.transforms as transforms;

from PIL import Image;

from torchmetrics import PeakSignalNoiseRatio;

import random;


DEVICE = torch.device("cpu");
numberOfSamples = 500;
Width = -1;
Height = -1;
width = -1;
height = -1;
baseDataFolder = "Dataset (128px)/";
baseTestDataFolder = "Test Dataset (128px)/";
transform = transforms.Compose(
                [transforms.ToTensor()])
transformNNormalize = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transformNNormalizePAN = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))])
deNormalize = transforms.Compose(
                [transforms.Normalize((-1, -1, -1), (2, 2, 2))])
normalizeImg = True;
after = True;

# This is a common function used to upsample or downsample the image using bicubic interpolation
def bicubicInterpolation(img, w, h):
    return nn.functional.interpolate(img, size=[w, h], scale_factor=None, mode='bicubic');

def loadData(trainingDatas, trainingTargetData, baseDataFolder):
    global Width, Height, width, height;
    print("\nLoading training LRMS, PAN and target HRMS images.")
    for index in range(1, (numberOfSamples + 1)):
        filename = "Sample" + str(index) + ".png"
        hrmsImg = Image.open(r""+baseDataFolder+"HRMS/" + filename);
        Width = hrmsImg.width;
        Height = hrmsImg.height;
        transformedHrmsImg = transform(hrmsImg) if not normalizeImg else transformNNormalize(hrmsImg);
        if transformedHrmsImg.shape[0] == 3:
            transformedHrmsImg = transformedHrmsImg.unsqueeze(0);
            trainingTargetData.append(transformedHrmsImg);
            
            trainingData = [];
            lrmsImg = Image.open(r""+baseDataFolder+"LRMS/" + filename);
            width = lrmsImg.width;
            height = lrmsImg.height;
            transformedLrmsImg = transform(lrmsImg) if not normalizeImg else transformNNormalize(lrmsImg);
            trainingData.append(transformedLrmsImg);
            transformedLrmsImg = transformedLrmsImg.unsqueeze_(0);
            upSamplesLrmsImg = bicubicInterpolation(transformedLrmsImg, Width, Height);
            upSamplesLrmsImg.clamp(min=0, max=255);
            trainingData.append(upSamplesLrmsImg);
            panImg = Image.open(r""+baseDataFolder+"PAN/" + filename);
            trainingData.append(transform(panImg) if not normalizeImg else transformNNormalizePAN(panImg));
            trainingDatas.append(trainingData);
            
            lenOfData = len(trainingTargetData);
            if lenOfData % 50 == 0 and lenOfData > 0:
                print("Loaded " + str(index) + " images...");
        else:
            print("Image " + filename + " has " + str(transformedHrmsImg.shape[0]) + " bands");
       
class Net(nn.Module):
    def __init__(self, B, C, b):
        super(Net, self).__init__();
        self.BCB = nn.Sequential(
            nn.Conv2d(in_channels=B, out_channels=C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=B, kernel_size=3, padding=1)
        );
        self.BCb = nn.Sequential(
            nn.Conv2d(in_channels=B, out_channels=C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=b, kernel_size=3, padding=1)
        );
        self.bCB = nn.Sequential(
            nn.Conv2d(in_channels=b, out_channels=C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=B, kernel_size=3, padding=1)
        );

    def forward(self, lrms, lrmsDeepPrior, pan):
        global after;
        for index in range(K):
            #print("Processing layer " + str(index + 1) + " of GPPNN layer...");
            lrmsDeepPriorUS = None;
            if not after:
                lrmsDeepPriorUS = bicubicInterpolation(lrmsDeepPrior, width, height);
            Lt = self.BCB(lrmsDeepPrior if lrmsDeepPriorUS == None else lrmsDeepPriorUS);
            LtUS = None;
            if after:
                LtUS = bicubicInterpolation(Lt, width, height);
            Rlt = lrms - (Lt if LtUS == None else LtUS);
            RltUS = None;
            if not after:
                RltUS = bicubicInterpolation(Rlt, Width, Height);
            Rht = self.BCB(Rlt if RltUS == None else RltUS);
            RhtUS = None;
            if after:
                RhtUS = bicubicInterpolation(Rht, Width, Height);
            Hlt = self.BCB(lrmsDeepPrior + (Rht if RhtUS == None else RhtUS));
            
            Pt = self.BCb(Hlt);
            Rpt = pan - Pt;
            Rht = self.bCB(Rpt);
            lrmsDeepPrior = self.BCB(Hlt + Rht);
        return lrmsDeepPrior;
    
trainingData = [];
trainingTargetData = [];
lr = 0.0005;
B = 3;
C = 30;                      # no. of features
b = 1;
K = 3;
numOfEpochs = 10;
criterion = nn.L1Loss();
net = Net(B, C, b);
optimizer = optim.Adam(net.parameters(), lr=lr);
psnr = PeakSignalNoiseRatio();
averageForBatch = 50;
    
# To perform training
trainCommand = input("Would you like to train the model? Enter Y for yes and N for no > ");
if trainCommand.lower() == "y":
    saveCmd = input("Would you like to save the trained model, later? Enter filename (without extension) if yes else press n for no > ");
    loadData(trainingData, trainingTargetData, baseDataFolder);
    for epoch in range(numOfEpochs):
        print("\n#################################### Epoch #", (epoch + 1));
        numOfTrainingData = len(trainingTargetData);
        # Randomize samples after each epoch
        #random.shuffle(trainingTargetData);
        predictedNTargetSum = 0.0;
        predictedEpochTargetSum = 0.0;
        for index in range(numOfTrainingData):
            lrms = trainingData[index][0];
            lrmsPrior = trainingData[index][1];
            pan = trainingData[index][2];
            H = trainingTargetData[index];
            
            optimizer.zero_grad();
            Ht = net(lrms, lrmsPrior, pan);
            loss = criterion(Ht, H);
            loss.backward();
            optimizer.step();
            psnrValue = float(psnr(Ht, H).item())
            if ((index + 1) % averageForBatch) == 0:
                predictedEpochTargetSum += psnrValue;
                predictedNTargetSum += psnrValue;
                print("Processed " + str(index + 1) + " training samples.\t\t\tAverage PSNR = {:.2f}".format(predictedNTargetSum/averageForBatch));
                predictedNTargetSum = 0;
            else:
                predictedNTargetSum += psnrValue;
                predictedEpochTargetSum += psnrValue;
        print("\nAverage PSNR for the Epoch with entire training samples = {:.2f}\n".format(predictedEpochTargetSum/numOfTrainingData));
        predictedEpochTargetSum = 0.0;
    if saveCmd.lower() != "n":
        torch.save(net.state_dict(), saveCmd + ".pt");
        
# To perform tests on test datasets.
testCmd = input("Would you like to test the trained model to measure performance? Enter filename (without extension) if yes else press n for no > ");
if testCmd.lower() != "n":
    try:
        isPresent = testCmd.index("128By64");
        baseTestDataFolder = "Test Dataset (128px)/";
    except:
        baseTestDataFolder = "Test Dataset (200px)/";
    settings = testCmd.split("(")[1].split(")")[0].split(",");
    C = int(settings[1]);
    K = int(settings[3]);
    numberOfSamples = 300;
    after = (True if settings[5] == "A" else False);
    net = Net(B, C, b);
    net.load_state_dict(torch.load(testCmd + ".pt"));
    net.eval()
    normalizeImg = (True if settings[4] == "T" else False);
    testData = [];
    testTargetData = [];
    loadData(testData, testTargetData, baseTestDataFolder);
    numOfTestData = len(testTargetData);
    # Randomize samples after each epoch
    #random.shuffle(testTargetData);
    predictedNTargetSum = 0.0;
    psnrList = [];
    predictedEpochTargetSum = 0.0;
    for index in range(numOfTestData):
        lrms = testData[index][0];
        lrmsPrior = testData[index][1];
        pan = testData[index][2];
        H = testTargetData[index];
        
        optimizer.zero_grad();
        Ht = net(lrms, lrmsPrior, pan);
        loss = criterion(Ht, H);
        loss.backward();
        optimizer.step();
        psnrValue = float(psnr(Ht, H).item());
        psnrList.append(psnrValue);
        if ((index + 1) % averageForBatch) == 0:
            predictedNTargetSum += psnrValue;
            predictedEpochTargetSum += psnrValue;
            print("Processed " + str(index + 1) + " testing samples.\t\t\tAverage PSNR = {:.2f}".format(predictedNTargetSum/averageForBatch));
            predictedNTargetSum = 0;
        else:
            predictedNTargetSum += psnrValue;
            predictedEpochTargetSum += psnrValue;
    print("\nAverage PSNR for the entire testing samples = {:.2f}\n".format(predictedEpochTargetSum/numOfTestData));
    predictedEpochTargetSum = 0.0;
    
# To demo the output image produced by the model
testCmd = input("Would you like to test your trained model on an image? Enter trained model filename (without extension) if yes else press n for no > ");
if testCmd != "n":
    print("\nTesting model '" + testCmd + ".pt'...");
    try:
        isPresent = testCmd.index("128By64");
        baseTestDataFolder = "Test Dataset (128px)/";
    except:
        baseTestDataFolder = "Test Dataset (200px)/";
    settings = testCmd.split("(")[1].split(")")[0].split(",");
    numberOfSamples = int(settings[0]);
    C = int(settings[1]);
    K = int(settings[3]);
    after = (True if settings[5] == "A" else False);
    net = Net(B, C, b);
    net.load_state_dict(torch.load(testCmd + ".pt"));
    net.eval()
    normalizeImg = (True if settings[4] == "T" else False);
    testImage = input("Enter the image name (without extension) you want to test > ");
    lrmsImg = Image.open(r""+baseTestDataFolder+"LRMS/"+testImage+".png");
    width = lrmsImg.width;
    height = lrmsImg.height;
    panImg = Image.open(r""+baseTestDataFolder+"PAN/"+testImage+".png");
    Width = panImg.width;
    Height = panImg.height;
    transformedLrmsImg = transform(lrmsImg) if not normalizeImg else transformNNormalize(lrmsImg);
    transformedLrmsImgSqueezed = transformedLrmsImg.unsqueeze_(0);
    upSamplesLrmsImg = bicubicInterpolation(transformedLrmsImgSqueezed, Width, Height);
    upSamplesLrmsImg.clamp(min=0, max=255);
    panImg = transform(panImg) if not normalizeImg else transformNNormalizePAN(panImg)
    output = net(transformedLrmsImg, upSamplesLrmsImg, panImg);
    output = output.squeeze_(0);
    output = output if not normalizeImg else deNormalize(output);
    pilImage = transforms.ToPILImage()(output);
    print("Saving output for the test image as '" +testImage+"_Output.png'");
    pilImage.save(testImage+"_Output.png");