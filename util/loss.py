import torch
import torch.nn as nn
import torch.nn.functional as F
def to_one_hot(tensor,nClasses):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,256,h,w).cuda(0).scatter_(1,tensor.view(n,1,h,w),1)
    one_hot = one_hot[:,0:nClasses,:,:]
    return one_hot


class SegmentationLosses(object):
    def __init__(self, label_nc = None,weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.label_nc = label_nc

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'mIoU':
            return self.mIoU
        elif mode =='dice':
            return self.dice_loss
        elif mode =='nll':
            return self.nllloss
        else:
            raise NotImplementedError


    def mIoU(self,inputs,target):
        target = target.byte()
        if target.size()[1] == self.label_nc:
            target_oneHot = target
        else:
            target_oneHot = to_one_hot(target,label_nc)
        '''
        x = 0
        for i in range(0,256):
            for j in range(0,256):
                if target_oneHot[0,:,i,j].any() == 0:
                    x += 1
        print('in the loss: '+str(x))
        '''
        target = target.float()
        target_oneHot = target_oneHot.float()
        
        N = inputs.size()[0]
        #print(inputs[0,:,1,1].sum())
        inputs = F.softmax(inputs,dim=1)
        '''
        print(inputs[0,:,1,1].max())
        print(target_oneHot[0,:,1,1])
        '''
        inter = (inputs * target_oneHot)
        inter = inter.view(N,self.label_nc,-1).sum(2)
        #print('input: '+ str(inputs.mean()))
        #torch.set_printoptions(threshold=5000000)
        #print('inter: '+ str(inter))
        union = (inputs + target_oneHot - (inputs*target_oneHot))
        union = union.view(N,self.label_nc,-1).sum(2)
        #print('union: '+str(union[0,:]))
        loss = (inter/union)
        return -loss.mean()


    def nllloss(self,output,target,ignore_index=-1):
        crit = nn.NLLLoss(ignore_index=ignore_index)
        seg_label_2d = torch.argmax(target,dim=1)
        return crit(output,seg_label_2d)


    def dice_loss(self,output, target, weights=None, ignore_index=None):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """ 
        target = torch.argmax(target,dim = 1)
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        target = torch.argmax(target,dim = 1).squeeze(1)
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




