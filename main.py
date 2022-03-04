import sys
from dataset import VideoDataSet
from loss_function import bmn_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal

sys.dont_write_bytecode = True


# write log
log_path=''
def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f) # 直接打印到log_path


def train_BMN(data_loader, model, optimizer, epoch, bm_mask, batch_size, epochs):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    total_steps = len(data_loader.dataset)//batch_size
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        if n_iter!=0 and (n_iter)%30==0:
            train_info = f'step:[{n_iter:3d}/{total_steps:3d}]============>\n'+"BMN training loss(epoch [%d/%d]): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
                    epoch+1, epochs, epoch_tem_loss / (n_iter + 1),
                    epoch_pemclr_loss / (n_iter + 1),
                    epoch_pemreg_loss / (n_iter + 1),
                    epoch_loss / (n_iter + 1))
            # print(f'step:[{n_iter:3d}/{total_steps:3d}]============>')
            # print(
            #     "BMN training loss(epoch [%d/%d]): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            #         epoch+1, 9, epoch_tem_loss / (n_iter + 1),
            #         epoch_pemclr_loss / (n_iter + 1),
            #         epoch_pemreg_loss / (n_iter + 1),
            #         epoch_loss / (n_iter + 1)))
            print_log(log_path, train_info)

    # print(f"\n==================epcho {epoch+1} end=================\n",
    #     "BMN training loss(epoch [%d/%d]): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
    #         epoch+1, epochs, epoch_tem_loss / (n_iter + 1),
    #         epoch_pemclr_loss / (n_iter + 1),
    #         epoch_pemreg_loss / (n_iter + 1),
    #         epoch_loss / (n_iter + 1)))
    train_info = f"================== epoch [{epoch+1}/{epochs}] training finished!=================\n"+\
        "BMN training loss(epoch [%d/%d]): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch+1, epochs, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))
    print_log(log_path, train_info)

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + f"/BMN_{epoch+1}.pth")

def test_BMN(data_loader, model, epoch, bm_mask, epochs):
    model.eval()
    best_loss = 1e10
    best_epoch = -1
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    # print(f"\n============epoch[{epoch+1}/{epochs}] testing finished!===========\n",
    #     "BMN testing loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
    #         epoch+1, epoch_tem_loss / (n_iter + 1),
    #         epoch_pemclr_loss / (n_iter + 1),
    #         epoch_pemreg_loss / (n_iter + 1),
    #         epoch_loss / (n_iter + 1)))
    
    test_info = f"\n============epoch[{epoch+1}/{epochs}] testing finished!===========\n"+\
    "BMN testing loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch+1, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))
    print_log(log_path, test_info)

    # state = {'epoch': epoch + 1,
    #          'state_dict': model.state_dict()}
    # torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch+1
        # torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")
    print_log(log_path, f'==================current best epoch is {best_epoch}=======================\n')


def BMN_Train(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
    #                                           batch_size=opt["batch_size"], shuffle=False,
    #                                           num_workers=8, pin_memory=True)

    # `StepLR` Decays the learning rate of each parameter group by gamma every step_size epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    best_AUC = 0.0
    best_epoch = -1
    for epoch in range(opt["train_epochs"]):
        
        train_BMN(train_loader, model, optimizer, epoch, bm_mask,opt["batch_size"], opt["train_epochs"])

        # test_BMN(test_loader, model, epoch, bm_mask, opt["train_epochs"])
        print_log(log_path, f'=================epoch {epoch+1} testing starts===================')
        current_AUC = my_test(opt, model, epoch)

        if best_AUC < current_AUC:
            best_AUC = current_AUC
            best_epoch = epoch+1
        print_log(log_path, f'\n=========best AUC {best_AUC} in epoch {best_epoch} for validation subset===========\n')

        scheduler.step()


def BMN_inference(opt):
    model = BMN(opt)# 
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()#checkpoint/BMN_checkpoint.pth.tar
    # checkpoint = torch.load("/root/BMN-Boundary-Matching-Network/checkpoint/BMN_best.pth.tar")
    checkpoint_path='checkpoint/BMN_15.pth'
    checkpoint = torch.load(checkpoint_path) #checkpoint/BMN_AUC67.7.pth.tar
    print('checkpoint[epoch]: ', checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,#batch_size_old=1
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)



def my_test(opt, model, epoch):
    model.eval()
    opt['mode'] = 'inference'
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,#batch_size_old=1
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)

    # for Post processing
    print_log(log_path, "Post processing start")
    BMN_post_processing(opt)
    print_log(log_path, "Post processing finished")
    result_AUC=evaluation_proposal(opt)
    epochs = opt["train_epochs"]
    print_log(log_path, f"===============epoch [{epoch+1}/{epochs}] testing finished!================")
    opt['mode'] = 'train'
    
    return result_AUC


def show_config(opt):
    print_log(log_path, '=====================Config=====================')
    for k,v in opt.items():
        print_log(log_path, k,': ',v)
    print_log(log_path, '======================End=======================')


def main(opt):
    
    if opt["mode"] == "train":
        BMN_Train(opt)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference(opt)
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    checkpoint_path = opt["checkpoint_path"]
    if not os.path.exists(checkpoint_path) and opt["mode"] == "train":
        os.makedirs(checkpoint_path)
    # opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    # opt['mode'] = 'inference'
    # print(opt['mode'])
    # json.dump(opt, opt_file)
    # opt_file.close()

    log_path = opt["log_path"]

    opt["exp_info"] = opt["exp_info"]+f'the expriment for BMN {opt["mode"]}. FFN linear layers implemented by conv!'

    show_config(opt)
   
    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)
