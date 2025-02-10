import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim

import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
parser.add_argument('--dataset', type=int, default=0, metavar='N',
                        help='0: UWA30, 1:UCB,2: DHA')                 
parser.add_argument('--epochs', type=int, default=10001, metavar='N',
                        help='number of epochs to train (default: 10001)')
parser.add_argument('--beta',type = float,default=0.7, metavar='N')
args = parser.parse_args()

class PreprocessDataset(Dataset):
    def __init__(self,device,dataName = 'UWA30',isTrain = True):
        super(PreprocessDataset,self).__init__()
        self.dataName = dataName
        self.isTrain = 'train' if isTrain else 'test'
        
        # Depth feature -> 110-dimension
        # RGB feature -> 3x2048 dimension
        self.depthNum = 110
        self.rgbNum = 3 * 2048
        self.totalNum = self.depthNum + self.rgbNum
        self.device = device

        _start = time.time()
        self.datas = self._readFile()
        self.length = len(self.datas)
        print("[INFO] Data name: %s, Is train: %s, Total: %d, Time: %.2fs" % 
              (self.dataName,self.isTrain,self.length,time.time()-_start))
        
    def _readFile(self):
        datas = []
        with open('./data/' + '%s_total_%s.csv' % (self.dataName,self.isTrain),'r') as file:
            for index in file:
                _row = [float(value) for value in index.rstrip().split(',')[:-1]]
                
                _depthFeature = torch.tensor(_row[0:self.depthNum]).to(self.device)
                _rgbFeature = torch.tensor(_row[self.depthNum:self.totalNum]).to(self.device)
                _label = torch.tensor(_row[self.totalNum:])
                _label = torch.argmax(_label, dim = 0).to(self.device)
                _data = {
                    'depth': _depthFeature,
                    'rgb': _rgbFeature,
                    'label': _label
                }
                
                datas.append(_data)
                
        return datas
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        data = self.datas[index]
        
        depthFeature = data['depth']
        rgbFeature = data['rgb']
        label = data['label']
    
        return depthFeature,rgbFeature,label

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
            
    def next(self):
        data = self.next_data
        self.preload()
        return data

class Encoder(nn.Sequential):
    def __init__(self,inputDim,outputDim,hiddenDim = 512):
        super().__init__(
            nn.Linear(inputDim,hiddenDim),
            nn.ReLU(inplace = True),
            nn.Linear(hiddenDim,outputDim)
        )
        
class Decoder(nn.Sequential):
    def __init__(self,inputDim,classNum,hiddenDim = 256):
        super().__init__(
            nn.Linear(inputDim,hiddenDim),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(hiddenDim,classNum)
        )

class Classifier(nn.Sequential):
    def __init__(self,classNum):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(classNum * 2,classNum)
        )


    def forward(self,rgb,depth):
        att = self.attention(torch.cat([rgb,depth],dim = 1))
        out = att * (rgb + depth) 
        return out

class Generator(nn.Sequential):
    def __init__(self,inputDim,outputDim,hiddenDim):
        super().__init__(
            nn.Linear(inputDim,hiddenDim),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(hiddenDim,hiddenDim),
            nn.LeakyReLU(inplace = True),
            nn.Linear(hiddenDim,outputDim)
        )
        
class Discriminator(nn.Sequential):
    def __init__(self,inputDim,hiddenDim):
        super().__init__(
            nn.Linear(inputDim,hiddenDim),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(inplace = True),
            nn.Linear(hiddenDim,1),
            nn.Sigmoid()
        )

class NetworkStructure(nn.Module):
    def __init__(self,rgbFeatureNum,dFeatureNum,classNum,rgbDim = 128,depDim = 128):
        super().__init__()
        self.rgbDim = rgbDim
        self.depDim = depDim
        
        self.encoder = nn.ModuleDict({
            'RGB': Encoder(rgbFeatureNum,self.rgbDim),
            'Depth':Encoder(dFeatureNum,self.depDim)
        })
        
        self.decoder = nn.ModuleDict({
            'RGB': Decoder(self.rgbDim,classNum),
            'Depth': Decoder(self.depDim,classNum)
        })
        
        self.generator = nn.ModuleDict({
            'RGB2Dep': Generator(self.rgbDim,self.depDim,512),
            'Dep2RGB': Generator(self.depDim,self.rgbDim,512)
        })
        
        self.discriminator = nn.ModuleDict({
            'RGB': Discriminator(self.rgbDim + self.depDim,256),
            'Depth': Discriminator(self.rgbDim + self.depDim,256)
        })
        
        self.classifier = Classifier(classNum)
        
    
        
    def forward(self,rgb,depth,isGAN = False):
        results = dict()
        realRgb = self.encoder['RGB'](rgb)
        realDepth = self.encoder['Depth'](depth)
        
        fakeRgb = self.generator['Dep2RGB'](realDepth)
        fakeDepth = self.generator['RGB2Dep'](realRgb)
        
        results['realRgbPred'] = self.decoder['RGB'](realRgb)
        results['fakeRgbPred'] = self.decoder['RGB'](fakeRgb)
        
        results['realDepthPred'] = self.decoder['Depth'](realDepth)
        results['fakeDepthPred'] = self.decoder['Depth'](fakeDepth)

        results['pred'] = self.classifier(results['realRgbPred'],
                                        results['realDepthPred'])
        results['frrdPred'] = self.classifier(results['fakeRgbPred'],
                                        results['realDepthPred'])
        results['rrfdPred'] = self.classifier(results['realRgbPred'],
                                        results['fakeDepthPred'])
        return results
    


def training(epoch,beta = 0.3): #0.8 80.62
    net.train(True)

    prefetcher = DataPrefetcher(trainData)
    depth,rgb,labels = prefetcher.next()

    for step in range(len(trainData) - 1):
        realRgb = net.encoder['RGB'](rgb)
        realDepth = net.encoder['Depth'](depth)
        fakeRgb = net.generator['Dep2RGB'](realDepth)
        fakeDepth = net.generator['RGB2Dep'](realRgb)

        realRgbPred = net.decoder['RGB'](realRgb.detach())
        fakeRgbPred = net.decoder['RGB'](fakeRgb.detach())
        
        realDepthPred = net.decoder['Depth'](realDepth.detach())
        fakeDepthPred = net.decoder['Depth'](fakeDepth.detach())
        
        #Classifier  
        net.classifier.zero_grad()
        pred = net.classifier(realRgbPred.detach(),realDepthPred.detach())

        loss = 0.5 * criteria(pred,labels) 
        loss.backward()
        optimizers['classifier'].step()

        net.classifier.zero_grad()
        frrdPred = net.classifier(fakeRgbPred.detach(),realDepthPred.detach())
        loss =  0.25 * criteria(frrdPred,labels)
        loss.backward()
        optimizers['classifier'].step()

        net.classifier.zero_grad()
        rrfdPred = net.classifier(realRgbPred.detach(),fakeDepthPred.detach())
        loss = 0.25 * criteria(rrfdPred,labels)
        loss.backward()
        optimizers['classifier'].step()

        #Decoder
        net.decoder['RGB'].zero_grad()
        realRgbPred = net.decoder['RGB'](realRgb.detach())
        loss = beta * criteria(realRgbPred,labels) 
        loss.backward()
        optimizers['rgbDecoder'].step()

        net.decoder['RGB'].zero_grad()
        fakeRgbPred = net.decoder['RGB'](fakeRgb.detach())
        loss = (1-beta) * criteria(fakeRgbPred,labels)
        loss.backward()
        optimizers['rgbDecoder'].step()


        net.decoder['Depth'].zero_grad()
        realDepthPred = net.decoder['Depth'](realDepth.detach())
        
        loss = beta * criteria(realDepthPred,labels)
        loss.backward()
        optimizers['depDecoder'].step()

        net.decoder['Depth'].zero_grad()
        fakeDepthPred = net.decoder['Depth'](fakeDepth.detach())
        loss = (1-beta) * criteria(fakeDepthPred,labels)
        loss.backward()
        optimizers['depDecoder'].step()

        #Encoder
        net.encoder['Depth'].zero_grad()
        realDepthPred = net.decoder['Depth'](realDepth)
        loss = criteria(realDepthPred,labels)
        loss.backward()
        optimizers['depEncoder'].step()
        
        net.encoder['RGB'].zero_grad()
        realRgbPred = net.decoder['RGB'](realRgb)
        loss = criteria(realRgbPred,labels)
        loss.backward()
        optimizers['rgbEncoder'].step()
        

        #GAN
        realRgb = net.encoder['RGB'](rgb)
        realDepth = net.encoder['Depth'](depth)
        fakeRgb = net.generator['Dep2RGB'](realDepth)
        fakeDepth = net.generator['RGB2Dep'](realRgb)

        realPairs = torch.cat([realRgb,realDepth],dim = 1)
        fakeRgbPairs = torch.cat([fakeRgb,realDepth],dim = 1)
        fakeDepPairs = torch.cat([realRgb,fakeDepth],dim = 1)

        net.discriminator['RGB'].zero_grad()
        disRealRgb = net.discriminator['RGB'](realPairs.detach())
        rgbLoss = disCriteria(disRealRgb,torch.ones_like(disRealRgb))
        rgbLoss.backward()
        optimizers['rgbDis'].step()

        net.discriminator['RGB'].zero_grad()
        disFakeRgb = net.discriminator['RGB'](fakeRgbPairs.detach())
        rgbLoss = disCriteria(disFakeRgb,torch.zeros_like(disFakeRgb))
        rgbLoss.backward()
        optimizers['rgbDis'].step()


        net.discriminator['Depth'].zero_grad()
        disFakeDep = net.discriminator['Depth'](fakeDepPairs.detach())
        depthLoss = disCriteria(disFakeDep,torch.zeros_like(disFakeDep)) 
        depthLoss.backward()
        optimizers['depDis'].step()

        net.discriminator['Depth'].zero_grad()
        disRealDep = net.discriminator['Depth'](realPairs.detach())
        depthLoss = disCriteria(disRealDep,torch.ones_like(disRealDep))
        depthLoss.backward()
        optimizers['depDis'].step()
            
        net.generator['Dep2RGB'].zero_grad()
        realRgb = net.encoder['RGB'](rgb)
        realDepth = net.encoder['Depth'](depth)
        fakeRgb = net.generator['Dep2RGB'](realDepth.detach())
        fakeRgbPairs = torch.cat([fakeRgb,realDepth.detach()],dim = 1)
        disFakeRgb = net.discriminator['RGB'](fakeRgbPairs)
        rgbLoss = disCriteria(disFakeRgb,torch.ones_like(disFakeRgb)) + \
                            100 * L1Criteria(realRgb.detach(),fakeRgb)
        rgbLoss.backward()
        optimizers['Dep2RGB'].step()


        net.generator['RGB2Dep'].zero_grad()
        fakeDepth = net.generator['RGB2Dep'](realRgb.detach())
        fakeDepPairs = torch.cat([realRgb.detach(),fakeDepth],dim = 1)
        disFakeDep = net.discriminator['Depth'](fakeDepPairs)
        depthLoss = disCriteria(disFakeDep,torch.ones_like(disFakeDep)) + \
                            100 * L1Criteria(realDepth.detach(),fakeDepth)
        depthLoss.backward()
        optimizers['RGB2Dep'].step()
        depth,rgb,labels = prefetcher.next()



def testing(epoch):
    net.train(False)
    history = {
        'RGB': 0.0,
        'Depth': 0.0,
        'Total': 0.0
    }

    total = 0
    prefetcher = DataPrefetcher(testData)
    depth,rgb,labels = prefetcher.next()

    for step in range(len(testData) - 1):
        results = net(rgb,depth,isGAN = False)
        history['RGB'] += torch.sum(torch.argmax(results['rrfdPred'], dim = 1) == labels)
        history['Depth'] += torch.sum(torch.argmax(results['frrdPred'], dim = 1) == labels)
        history['Total'] += torch.sum(torch.argmax(results['pred'], dim = 1) == labels)
        total += labels.shape[0]
        depth,rgb,labels = prefetcher.next()

    for key,value in history.items():
        history[key] = value/total
    return history



if __name__ == "__main__":
    dataset = ['UWA30','UCB','DHA']
    dataName = dataset[args.dataset]
    BATCH = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainDataset = PreprocessDataset(dataName = dataName,device = device,isTrain = True)
    testDataset = PreprocessDataset(dataName = dataName,device = device,isTrain = False)

    trainData = DataLoader(trainDataset,batch_size = BATCH,shuffle = True,num_workers = 0)
    testData = DataLoader(testDataset,batch_size = BATCH * 4,shuffle = True,num_workers = 0)

    

    dFeatureNum = 110
    rgbFeatureNum = 3 * 2048

    rgbDim = 256
    depDim = 256

    # assign action number for different datasets
    if dataName == 'UWA30':
        classNum = 30
    elif dataName == 'UCB':
        classNum = 11
    elif dataName == 'DHA':
        classNum = 23

    net = NetworkStructure(rgbFeatureNum,dFeatureNum,classNum,
                        rgbDim = rgbDim,depDim = depDim).to(device)

    criteria = nn.CrossEntropyLoss().to(device)
    disCriteria = nn.BCELoss().to(device)
    L1Criteria = nn.L1Loss().to(device)

    optimizers = {
        'rgbEncoder': optim.AdamW(net.encoder['RGB'].parameters(),lr = 1e-5),
        'depEncoder': optim.AdamW(net.encoder['Depth'].parameters(),lr = 1e-5),
        
        'rgbDecoder': optim.AdamW(net.decoder['RGB'].parameters(),lr = 1e-5,weight_decay=1e-4),
        'depDecoder': optim.AdamW(net.decoder['Depth'].parameters(),lr = 1e-5,weight_decay=1e-4),
        
        'RGB2Dep': optim.AdamW(net.generator['RGB2Dep'].parameters(),lr = 1e-5,weight_decay=1e-4),
        'Dep2RGB': optim.AdamW(net.generator['Dep2RGB'].parameters(),lr = 1e-5,weight_decay=1e-4),
        
        'rgbDis': optim.AdamW(net.discriminator['RGB'].parameters(),lr = 1e-5),
        'depDis': optim.AdamW(net.discriminator['Depth'].parameters(),lr = 1e-5),
        
        'classifier': optim.AdamW(net.classifier.parameters(),lr = 1e-5,weight_decay=1e-4)
    }

    historys = {
        'RGB': 0.0,
        'Depth': 0.0,
        'Total': 0.0   
    }

    history = testing(0)
    for key,value in history.items():
            historys[key] = value.item()
    

    for epoch in range(1,args.epochs):
        
        start = time.time()
        trainHistory = training(epoch,beta = args.beta)
        history = testing(epoch)
        if epoch % 5 == 0:
            print("Epoch %-6d, Accuracy: RGB = %.4f(%.4f) Depth = %.4f(%.4f) Total = %.4f(%.4f) | Time: %.4f s" %
                    (epoch,history['RGB'],historys['RGB'],
                    history['Depth'],historys['Depth'],
                    history['Total'],historys['Total'],time.time() - start))

        for key,value in history.items():
            value = value.item()
            if value > historys[key]:
                historys[key] = value

    filename = 'result.txt'
    with open(filename,'a') as file:
        file.write("\n********************************************")
        file.write("\nTime: " + time.asctime(time.localtime(time.time())) + "\n")
        file.write("Dataset: %d\n" % (args.dataset))
        file.write("Epoch: %d\n" % (args.epochs))
        for key,value in history.items():
            file.write(key +": %.4f" % (historys[key]) + "\n")
        file.write("********************************************\n")