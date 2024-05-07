import torch
import torch.nn  as nn
import torch.nn.functional as F

# bygger et u-net

class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3,padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding=1)
		self.bn = nn.BatchNorm2d(outChannels)
		
	def forward(self, x):
		return self.relu(self.bn(self.conv2(self.relu(self.bn(self.conv1(x))))))
	

class Encoder(nn.Module):
    def __init__(self,channels=[3,64,128,256,512]):
        super().__init__()
        # Encoder funksjoner
        self.channels = channels
        self.blocks = nn.ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)


    def forward(self, x):
        # forward for encoder
        block_outputs = []
        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)
            block_outputs.append(x)
            x = self.maxpool(x)

        x = self.blocks[-1](x)
        block_outputs.append(x)
        return block_outputs
    

class Decoder(nn.Module):
    def __init__(self,channels=[512,256,128,64]):
        super().__init__()
        # Decoder funksjoner
        self.channels = channels
        self.blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])  
        self.upConvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]) 
    def forward(self,x,block_outputs):
        x = self.upConvs[0](x)
        # forward for decoder
        for i in range(1,len(self.channels)-1):
            x = self.upConvs[i](x)
            x = torch.cat([x,block_outputs[i]],dim=1)
            x = self.blocks[i](x)
        return x


class UNet(nn.Module):
	def __init__(self,
			  encChannels=[3,64,128,256,512],
			  decChannels=[512,256,128,64],
			  nbClasses=2,
			  outSize=(512,512)):
		super().__init__()
		# initialize encoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize decoder
		self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
		self.outSize = outSize

	def forward(self, x):
		encFeatures = self.encoder(x)

		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

		map = self.head(decFeatures)

		map = F.interpolate(map, self.outSize)
		return map

# HJELPEFUNKSJONER

# viser bilde og maske av en tile
import torchvision
import matplotlib.pyplot as plt
import utils

def view_prediction(sensor, net, batch, save_as=None, show=True):
    images = batch["image"]
    labels = batch["mask"]

    def clip_preview_band(band):
        b = np.zeros_like(band)
        for i in range(len(b)):
            output = utils.pct_clip(band[i])
            b[i] = output
        return b

    fig, axs = plt.subplots(5, 1,figsize=(21,10),dpi=150,gridspec_kw = {'hspace':0})

    if sensor == "s1":
        r = clip_preview_band(images[:,[1],:,:])
        g = clip_preview_band(images[:,[0],:,:])
        b = clip_preview_band(images[:,[1],:,:]) / clip_preview_band(images[:,[0],:,:])
        rgb_images = torch.tensor(np.array([r,g,b]).squeeze()).permute(1,0,2,3)
        axs[0].set_ylabel("SAR Image")
    elif sensor == "s2":
        rgb_images = images[:,[3,2,1],:,:]
        axs[0].set_ylabel("MSI Images")
    elif sensor == "s1s2":
        r = images[:,[3],:,:]
        g = images[:,[2],:,:]
        # b = images[:,[1],:,:]
        b = clip_preview_band(images[:,[-1],:,:])

        rgb_images = torch.tensor(np.array([r,g,b]).squeeze()).permute(1,0,2,3)
        axs[0].set_ylabel("MSI/SAR Images")

    labels = labels.unsqueeze(1)
    
    axs = axs.flatten()
    
    axs[0].imshow(torchvision.utils.make_grid(rgb_images).permute(1,2,0))
    # axs[0].set_ylabel("RGB Images")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].matshow(torchvision.utils.make_grid(labels).permute(1,2,0)[:,:,0])
    axs[1].set_ylabel("Ground Truth")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    with torch.no_grad():
        if torch.cuda.is_available():
            out = net(images.to("cuda")).to("cpu")
        else:
            out = net(images)

    score = torch.softmax(out,dim=1)
    _, prediction = torch.max(score,dim=1)
    prediction = prediction.unsqueeze(1)
    
    axs[2].matshow(torchvision.utils.make_grid(score[:,1,:,:].unsqueeze(1)).permute(1,2,0)[:,:,0],vmin=0,vmax=1)
    #axs[2].matshow(torchvision.utils.make_grid(prediction).permute(1,2,0)[:,:,0])
    axs[2].set_ylabel(f"Score - built up")    
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    axs[3].matshow(torchvision.utils.make_grid(score[:,2,:,:].unsqueeze(1)).permute(1,2,0)[:,:,0],vmin=0,vmax=1)
    #axs[3].matshow(torchvision.utils.make_grid(prediction).permute(1,2,0)[:,:,0])
    axs[3].set_ylabel(f"Score - roads")    
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    
    # axs[4].imshow(torchvision.utils.make_grid(score[:,0:3,:,:]).permute(1,2,0))
    axs[4].matshow(torchvision.utils.make_grid(prediction).permute(1,2,0)[:,:,0])
    axs[4].set_ylabel(f"Prediction")    
    axs[4].set_xticks([])
    axs[4].set_yticks([])


    plt.subplots_adjust(hspace=None)
    if save_as is not None:
        axs[0].set_title(save_as)
        plt.savefig(save_as,bbox_inches='tight')  
        return fig
    if show:
        plt.show()
    else:
        plt.close()


# lage en accuracy test, som tester presicion, recall og antall riktige prediksjoner og en confusion matrix
import numpy as np
from sklearn.metrics import average_precision_score

def test_accuracy(net, loader, batch_lim=10):
    mpre = []
    mrec = []
    map = []
    preds = []
    truth = []
    
    for i,batch in enumerate(loader):
        with torch.no_grad():
            images = batch["image"]
            labels = batch["mask"]

            if torch.cuda.is_available():
                out = net(images.to("cuda")).to("cpu")
            else:
                out = net(images)

            score = torch.softmax(out,dim=1)
            _, prediction = torch.max(score,dim=1)

            preds.append(prediction)
            truth.append(labels)
            
        
        ap = []
        rec = []
        pre = []
        b,c,h,w = score.shape
        # regn ut presicion og recall for hver klasse og gi ut gjennomsnittet
        for i in range(c):
            class_label = (labels == i) * 1
            if torch.sum(class_label) == 0:
                continue
            ap.append(average_precision_score(class_label.flatten().numpy(),score[:,i,:,:].flatten().numpy()))

            positives = labels == i
            
            TP = torch.sum(torch.eq(prediction[positives],labels[positives]))
            recall = TP/(labels==i).sum()
            if np.isnan(recall):
                rec.append(0)
            else:
                rec.append(recall)
            precision = TP/(prediction==i).sum()
            if np.isnan(precision):
                pre.append(0)
            else:
                pre.append(precision)
            

        map.append(np.mean(np.array(ap)))
        mrec.append(np.mean(np.array(rec)))
        mpre.append(np.mean(np.array(pre)))

        if batch_lim is None:
            pass
        elif i > batch_lim:
            break

    return np.mean(np.array(mrec)),np.mean(np.array(mpre)),np.mean(np.array(map))