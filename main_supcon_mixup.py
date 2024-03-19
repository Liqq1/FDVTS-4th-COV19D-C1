import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset1 import Lung3D_eccv_patient_supcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet

import torch.backends.cudnn as cudnn
import random
import math


print("torch = {}".format(torch.__version__))

IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='try2', help='visname')
parser.add_argument('--batch_size', '-bs', default=8, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')
parser.add_argument('--pretrain', '-pre', default=True, type=bool, help='use pretrained')
parser.add_argument('--supcon', '-con', default=False, type=bool, help='use supcon')
parser.add_argument('--mixup', '-mix', default=False, type=bool, help='use mix')
parser.add_argument('--box_lung', '-box_lung', default=False, type=bool, help='data box lung')
parser.add_argument('--seg_sth', '-seg_something', default=None, type=str, help='lung or lesion, cat to input')

parser.add_argument('--iccv_test', '-iccv_test', default=False, type=bool, help='use iccv test as train')
parser.add_argument('--weighted_loss', '-wl', default=True, type=bool, help='weighted ce loss')
parser.add_argument('--mosmed', '-mm', default=False, type=bool, help='use mosmed in challenge 2')
parser.add_argument('--model', '-model', default='resnest50_3D', type=str, help='use mosmed in challenge 2')
parser.add_argument('--val_certain_epoch', '-val_certain_epoch', default=False, type=str, help='use mosmed in challenge 2')
parser.add_argument('--optimizer', '-optim', default='adam', type=str, help='use mosmed in challenge 2')


best_f1 = 0
val_epoch = 1
save_epoch = 10

my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 

def parse_args():
    global args
    args = parser.parse_args()

def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    device = x.get_device()
    if use_cuda:
        index = torch.randperm(batch_size).to(device).long()
    else:
        index = torch.randperm(batch_size).long()

    mixed_x = (lam * x + (1 - lam) * x[index,:]).clone()
    y_a, y_b = y, y[index]
    return mixed_x, y_a.long(), y_b.long(), lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1048576

def main():
    print(torch.cuda.device_count())
    global best_f1
    global save_dir

    parse_args()
    if args.seg_sth:
        ipt_dim=2
    else:
        ipt_dim=1

    # prepare the model
    target_model = SupConResNet(name=args.model, ipt_dim=ipt_dim, head='mlp', feat_dim=128, 
                                                           n_classes=2, supcon=args.supcon)

    if args.supcon:
        s1 = target_model.sigma1
        s2 = target_model.sigma2

    if args.n_classes == 4:
        if args.model == 'P3DCResNet50' or args.model == 'medicalnet_resnet50':
            target_model.encoder.classifier = nn.Linear(2048,4)
        elif args.model == 'medicalnet_resnet34':
            target_model.encoder.classifier = nn.Linear(512,4)
        else:
            target_model.encoder.fc = nn.Linear(2048,4)

    if args.pretrain:
        ckpt = torch.load(' path to pth')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]

        target_model.load_state_dict(unParalled_state_dict,strict=False)

    print('Params: ', count_parameters(target_model))

    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()
    
    # prepare data
    train_data = Lung3D_eccv_patient_supcon(train=True,val=False,n_classes=args.n_classes, supcon=args.supcon, 
                 box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)
    val_data = Lung3D_eccv_patient_supcon(train=False,val=True,n_classes=args.n_classes, supcon=args.supcon, 
                 box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    criterion = SupConLoss(temperature=0.1)
    criterion = criterion.cuda()
    if args.n_classes==4:
        if args.weighted_loss:
            if args.mosmed:
                weight = torch.tensor([0.0931, 0.0985, 0.1433, 0.6651]).cuda() #add mosmed
                # weight = torch.tensor([1., 1., 1., 2.]).cuda() #add mosmed
            else:
                weight = torch.tensor([0.1506, 0.2065, 0.1506, 0.4923]).cuda()
                # weight = torch.tensor([1., 1., 1., 2.]).cuda()
        else:
            weight=None
    else:
        weight = None
    print(weight)
    criterion_clf = nn.CrossEntropyLoss(weight=weight)
    criterion_clf = criterion_clf.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(target_model.parameters(), args.lr, momentum=0.9, weight_decay=1e-5)

    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/con/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    test_log=open('./logs/'+args.visname+'.txt','w')   

    if args.val_certain_epoch:
        # weight_dir = 'checkpoints/con/*grade_resnest50_con64_mix_catlesion/62.pkl'
        weight_dir = 'checkpoints/con/grade_resnest50_con64_mix_catlung_1/58.pkl'
        epoch = int(weight_dir.split('/')[-1].split('.')[0])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']
        target_model.load_state_dict(state_dict, strict=True) 

        val_log=open('./logs/val.txt','w')   
        
        val1(target_model,val_loader,epoch,val_log,args.optimizer)
        exit()

    # train the model
    initial_epoch = 0
    for epoch in range(initial_epoch, args.epochs):
        target_model.train()
        con_matx.reset()
        total_loss1 = .0
        total_loss2 = .0
        total_loss3 = .0
        
        total = .0
        correct = .0
        count = .0
        total_num = .0

        # lr = args.lr
        lr = get_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True)
   
        for i, (imgs, masks, labels, ID) in enumerate(pbar):
            if args.supcon:
                imgs = torch.cat([imgs[0],imgs[1]],dim=0) #2*bsz,256,256
            #lqq: img.shape - [8, 1, 64, 256, 256]
            imgs = imgs.unsqueeze(1).float().cuda() #2*bsz,1,256,256   ? 
            # imgs = imgs.repeat(1,3,1,1,1)   

            if args.seg_sth:
                if args.supcon:           
                    masks = torch.cat([masks[0], masks[1]],dim=0) #2*bsz,256,256
                masks = masks.unsqueeze(1).float().cuda() #2*bsz,1,256,256    
                imgs = torch.cat([imgs, masks], dim=1)     # 2*bs, 2, 128,256,256           

            # lqq: label.shape - [8]
            labels = labels.float().cuda()
            bsz = labels.shape[0]

            ## mixup
            if args.mixup:
                if args.supcon:
                    lam = 0.4
                    target_labels = torch.cat([labels, labels],dim=0)
                    mix_imgs, targets_a, targets_b, lam = mixup_data(imgs, target_labels, lam)

                    _, _, mix_pred = target_model(mix_imgs)
                    mix_pred1, mix_pred2 = torch.split(mix_pred, [bsz, bsz], dim=0) #bsz,n_classs
                    targets_a1, targets_a2 = torch.split(targets_a, [bsz, bsz], dim=0) #bsz,n_classs
                    targets_b1, targets_b2 = torch.split(targets_b, [bsz, bsz], dim=0) #bsz,n_classs


                    mix_pred1 = F.softmax(mix_pred1)
                    mix_pred2 = F.softmax(mix_pred2)
                    _, mix_predicted1 = mix_pred1.max(1)
                    _, mix_predicted2 = mix_pred2.max(1)

                    loss_func1 = mixup_criterion(targets_a1, targets_b1, lam)
                    loss_mixup1 = loss_func1(criterion_clf, mix_pred1)

                    loss_func2 = mixup_criterion(targets_a2, targets_b2, lam)
                    loss_mixup2 = loss_func1(criterion_clf, mix_pred2)
                    
                    loss_mix = 0.5*loss_mixup1+0.5*loss_mixup2
                else:
                    lam = 0.4
                    target_labels = labels
                    #print(imgs.shape)
                    mix_imgs, targets_a, targets_b, lam = mixup_data(imgs, target_labels, lam)

                    _, mix_pred = target_model(mix_imgs)
                    mix_pred = F.softmax(mix_pred)
                    _, predicted = mix_pred.max(1)

                    loss_func = mixup_criterion(targets_a, targets_b, lam)
                    loss_mixup = loss_func(criterion_clf, mix_pred)

                    loss_mix = loss_mixup      

                    # pred_list.append(predicted.cpu().detach())
                    # label_list.append(labels.cpu().detach())

            # if not args.mixup: # mixup only
            #lqq: pred.shape - [8,2]
            _, features, pred = target_model(imgs) #2*bsz,128 #2*bsz,n_class
            if args.supcon:
                f1, f2 = torch.split(features, [bsz, bsz], dim=0) #bsz,128
                features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1) #bsz,2,128
                loss_con = criterion(features,labels)

                pred1, pred2 = torch.split(pred, [bsz, bsz], dim=0) #bsz,n_classs
                pred1 = F.softmax(pred1)
                pred2 = F.softmax(pred2)
                con_matx.add(pred1.detach(),labels.detach())
                con_matx.add(pred2.detach(),labels.detach())
                _, predicted1 = pred1.max(1)
                _, predicted2 = pred2.max(1)
                loss_clf = 0.5*criterion_clf(pred1,labels.long())+0.5*criterion_clf(pred2,labels.long())
                
                pred_list.append(predicted1.cpu().detach())
                label_list.append(labels.cpu().detach())
                pred_list.append(predicted2.cpu().detach())
                label_list.append(labels.cpu().detach())        

            else:
                pred = F.softmax(pred)
                con_matx.add(pred.detach(),labels.detach())
                # predicted.shape - [8]
                _, predicted = pred.max(1)
                loss_clf = criterion_clf(pred,labels.long())            

                pred_list.append(predicted.cpu().detach())
                label_list.append(labels.cpu().detach())

            if args.mixup and not args.supcon:
                loss = loss_mix #+ loss_clf
                loss_con = torch.zeros_like(loss)
                loss_clf = torch.zeros_like(loss)
            elif args.supcon and not args.mixup:
                loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*loss_clf+s2
                loss_mix = torch.zeros_like(loss)
            elif args.supcon and args.mixup:
                loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*(loss_mix + loss_clf)+s2
            else:
                loss = loss_clf
                loss_con = torch.zeros_like(loss)
                loss_mix = torch.zeros_like(loss)


            total_loss1 += loss_con.item()
            total_loss2 += loss_clf.item()
            total_loss3 += loss_mix.item()

            if args.supcon:
                total += 2 * bsz
                correct += predicted1.eq(labels.long()).sum().item()
                correct += predicted2.eq(labels.long()).sum().item()
            else:
                total += bsz
                correct += predicted.eq(labels.long()).sum().item()                
            count += 1
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            pbar.set_description('loss: %.3f' % (total_loss3 / (i+1))+' acc: %.3f' % (correct/total))

        test_log.write('Epoch:%d  lr:%.5f  Loss_con:%.4f Loss_clf:%.4f Loss_mix:%.4f  acc:%.4f \n'%(epoch, lr, total_loss1 / count, total_loss2 / count, total_loss3 / count, correct/total))
        test_log.flush() 

        if (epoch + 1) % val_epoch == 0:
            val1(target_model,val_loader,epoch,test_log, optimizer)
            if args.supcon:
                print(torch.exp(-s1).item(),torch.exp(-s2).item())
         

@torch.no_grad()
def val1(net, val_loader, epoch,test_log, optimizer):
    global best_f1
    parse_args()
    net = net.eval()

    correct = .0
    total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    pred_list = []
    label_list = []

    # total_ = []
    # label_ = []

    pbar = tqdm(val_loader, ascii=True)

    for i, (data, masks, label,id) in enumerate(pbar):
        data = data.unsqueeze(1)
        # data = data.repeat(1,3,1,1,1)   


        data = data.float().cuda()
        label = label.float().cuda()
        if args.seg_sth:
            masks = masks.unsqueeze(1)
            masks = masks.float().cuda()
            data = torch.cat([data, masks], dim=1)

        _, feat, pred = net(data)

        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        pred_list.append(predicted.cpu().detach())
        label_list.append(label.cpu().detach())

        total += data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach()) 
        pbar.set_description(' acc: %.3f'% (100.* correct / total))


    recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average='macro')
    f1_4 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)

    
    print(correct, total)
    acc = 100.* correct/total

    print('val epoch:', epoch, ' val acc: ', acc, 'recall:', recall, "precision:", precision, "f1_macro:",f1, 'f1:', f1_4)
    print(str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   f1:%.4f  con:%s \n'%(epoch,acc, f1, str(con_matx.value())))
    test_log.flush() 

    if not args.val_certain_epoch:
        if f1 >= best_f1:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'f1': f1,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            save_name = os.path.join(save_dir, str(epoch) + '.pkl')
            torch.save(state, save_name)
            best_f1 = f1


if __name__ == "__main__":
    main()
        

