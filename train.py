# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#---

# This code is based on the work of OPPO and has been modified by jinboyan.

import os
import torch
import traceback
from random import randint
import random 
import sys 
import uuid
import time 
import json
import wandb
import torchvision
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.system_utils import mkdir_p
from utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss,msssim
from utils.image_utils import psnr,easy_cmap
from helper_train import getloss, controlgaussians, trbfunction
from scene import Scene
from argparse import Namespace
from helper3dg import getparser, getrenderparts
from renderer import train_render as render
from scene.saro_gaussian import GaussianModel

def train(dataset, opt, pipe, saving_iterations,testing_iterations, debug_from,start_iteration,checkpoint = None ,densify=0, duration=50,wandb_run = None, rgbfunction="rgbv1", rdpip="v2",no_report=False):
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    first_iter = start_iteration

    
    gaussians = GaussianModel(dataset)
    gaussians.preprocesspoints = opt.preprocesspoints 
    gaussians.duration = duration


    scene = Scene(dataset, gaussians, duration=duration, loader=dataset.loader,shuffle=False)
    print("checkpoint:",checkpoint)
    if checkpoint:
        gaussians.load_ply(checkpoint)
        # (model_params, first_iter) = torch.load(checkpoint)

        # if first_iter > opt.static_iteration:#已经进行过转换了
        #     gaussians.is_dynamatic = True
        # gaussians.restore(model_params, opt)

    gaussians.training_setup(opt)


    currentxyz = gaussians._xyz 
    maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
    minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
     
   
    maxbounds = [maxx, maxy, maxz]
    minbounds = [minx, miny, minz]



    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(3)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    best_psnr = 0.0
    history_data=None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    flag = 0


    if opt.batch > 1 and opt.multiview:
    # if 1:
        #针对多目数据集，记录同一时刻的相机列表
        traincameralist = scene.getTrainCamInfos().copy() if dataset.use_loader else scene.getTrainCameras().copy()
        
        train_camname_dict = {}
        for idx,cam in enumerate(traincameralist):
            if cam.image_name not in train_camname_dict:
                train_camname_dict[cam.image_name] = []
            train_camname_dict[cam.image_name].append(idx)

        traincam_dataset = scene.getTrainCameras()
        loader = DataLoader(traincam_dataset, batch_size=opt.batch,shuffle=True,num_workers=8,collate_fn=list)
        test_loader = DataLoader(scene.getTestCameras(), batch_size=1,shuffle=False,num_workers=8,collate_fn=lambda x: x)


    elif opt.batch ==1 and not opt.multiview:
        traincameralist = scene.getTrainCameras().copy()

    scene.recordpoints(0, "start training")

                                                            

    
    if (densify == 1 or  densify == 2 or densify == 4) and not dataset.random_init: 
        #这个过滤对减少悬浮物非常的重要
        zmask = gaussians._xyz[:,2] < 4.5  

        gaussians.prune_points(zmask) 
        print("After pure z<4.5",gaussians._xyz.shape[0])
        torch.cuda.empty_cache()



    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') ] #记录所有的loss
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    


    iteration = first_iter


    testing_iterations += [i for i in range(opt.densify_until_iter,opt.iterations) if i%500==0]
    print(testing_iterations)
    while iteration < opt.iterations+1:
        # print(iteration)
        for camindex in loader: #统一使用dataloder
            iteration +=1
            if iteration > opt.iterations:
                break
            # if opt.coarse_iteration >=0 and iteration > opt.coarse_iteration:
            #     if not gaussians.is_fine:
            #         gaussians.coarse2fine()
            #         opt.lambda_dscale_entropy = 0
            if iteration > opt.static_iteration:
                stage = "dynamatic"
                if  not gaussians.is_dynamatic:
                    gaussians.static2dynamatic()
            else:
                stage = "static" 
                # camindex = [scene.getTestCameras()[0]]


            iter_start.record()
            if opt.all_no_intergral:
                use_intergral = False
                scale_intergral = False
            else:
                if opt.use_intergral_afterdensify:
                    use_intergral =True
                else:
                    if iteration > opt.densify_until_iter:
                        use_intergral=False
                    else:
                        use_intergral =True
                if iteration > opt.densify_until_iter:
                    scale_intergral= False
                else:
                    scale_intergral = True
            # scale_intergral = False
            gaussians.update_learning_rate(iteration,stage=stage,use_intergral=use_intergral,scale_intergral=scale_intergral)
            
            if (iteration - 1) == debug_from:
                pipe.debug = True

            if opt.batch > 1:
                gaussians.zero_gradient_cache()

                batch_point_grad = []
                batch_visibility_filter = []
                batch_radii = []
                batch_select_mask = []
                # gaussians.get_batch_feature()
                for idx,viewpoint_cam in enumerate(camindex):
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background,stage=stage)
                    image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg) 
                    
                    select_mask = None #当是bi-flow动态阶段时，valid_mask不是None
                    if "select_mask" in render_pkg:
                        select_mask = render_pkg["select_mask"]
                        batch_select_mask.append(select_mask)
                    gt_image = viewpoint_cam.original_image.float().cuda()

                    Ll1 = l1_loss(image, gt_image)
                    loss,loss_dict = getloss(opt, Ll1, ssim, image, gt_image, gaussians,lambda_all)

                    loss.backward() #这里loss一定要在这里backward，否则累计的计算图会爆显存
                    batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))

                    # # batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                    batch_radii.append(radii)
                    batch_visibility_filter.append(visibility_filter)
                    # print(viewspace_point_tensor.grad)


                    gaussians.cache_gradient(stage)#把梯度保存下来。针对batch feature,前batch-1次将梯度暂存，在最后一次再将梯度传给batch feature
                    gaussians.optimizer.zero_grad(set_to_none = True)# 清空梯度



                iter_end.record()
                gaussians.set_batch_gradient(opt.batch,stage)

            else:
                raise NotImplementedError("Batch size 1 is not supported")

            if dataset.use_shs :
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                psnr_for_log = psnr(image, gt_image).mean().double()
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * loss_dict[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema

                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",}
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                
                if iteration == opt.iterations:
                    progress_bar.close()


                if not no_report:
                    test_psnr,history_data = training_report(wandb_run,test_loader,iteration, scene.model_path,train_camname_dict, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                    ( pipe, background) , loss_dict=loss_dict,history_data=history_data,stage=stage )
                    if (iteration in testing_iterations ):
                        if test_psnr >= best_psnr:
                            best_psnr = test_psnr
                            
                            print("\n[ITER {}] Saving best checkpoint".format(iteration))
                            scene.save(iteration,best_ckpt=True)
                            # save_path = os.path.join(scene.model_path + "/point_cloud/chkpnt_best.pth")
                            # mkdir_p(os.path.dirname(save_path))
                            # torch.save((gaussians.capture(), iteration), save_path)
            
                #save
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)




                # Densification and pruning here
                if iteration < opt.densify_until_iter :
                    if opt.batch>1:
                        if len(batch_select_mask) >0:
                            visibility_count = torch.zeros(gaussians.get_xyz.shape[0]).to("cuda")
                            radii_list = []
                            batch_viewspace_point_grad = torch.zeros(gaussians.get_xyz.shape[0]).to("cuda")
                            for idx, select_mask in enumerate(batch_select_mask):
                                # print(select_mask.shape,visibility_count.shape,batch_visibility_filter[idx].shape)
                                visibility_count[select_mask] += batch_visibility_filter[idx]
                                batch_viewspace_point_grad[select_mask] += batch_point_grad[idx]

                                radii = torch.zeros(gaussians.get_xyz.shape[0]).to(batch_radii[idx])
                                # print(batch_radii[idx].dtype)
                                radii[select_mask] = batch_radii[idx]
                                radii_list.append(radii)
                            radii = torch.stack(radii_list,1).max(1)[0]
                            visibility_filter = visibility_count > 0
                        else:
                            visibility_count = torch.stack(batch_visibility_filter,1).sum(1) #计算batch中每个点的可见总数
                            visibility_filter = visibility_count > 0
                            radii = torch.stack(batch_radii,1).max(1)[0]
                            
                            batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)#将grad加起来
                        batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter]  / visibility_count[visibility_filter] #grad除以可见次数，得到batch平均grad
                        batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                        batch_t_grad = None

                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) #更新gs在2d情况下的最大半径,这个要写成逐K的
                    if opt.batch>1:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter,batch_t_grad) #增加累计梯度
                    else:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #增加累计梯度
                
                controlgaussians(opt, gaussians, densify, iteration, scene)
              
                # Optimizer step
                if iteration < opt.iterations:

                    gaussians.optimizer.step() #根据梯度更新参数
                    
                    gaussians.optimizer.zero_grad(set_to_none = True)

def training_report(wd_writer, test_loader,iteration, model_path,train_camname_dict, loss, l1_loss, elapsed, testing_iterations, scene : Scene,renderFunc, renderArgs,history_data=None,loss_dict=None,**renderKwargs):
    if  wd_writer:
        wandb.log({
            # 'train_loss_patches/l1_loss': Ll1.item(),
            # 'train_loss_patches/ssim_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed,
            'total_points': scene.gaussians.get_xyz.shape[0],
            'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu().numpy()),
        }, step=iteration)
        if hasattr(scene.gaussians,"_trbf_center"):
            wandb.log({
                'scene/trbf_scale_histogram':wandb.Histogram(scene.gaussians.get_trbfscale.cpu().numpy()),
                'scene/trbf_center_histogram':wandb.Histogram(scene.gaussians.get_trbfcenter.cpu().numpy()),
                'scene/trf_center_mean':scene.gaussians.get_trbfcenter.mean().cpu().item(),
                'scene/trf_center_std':scene.gaussians.get_trbfcenter.std().cpu().item()
            },step=iteration)
            #观测前5帧的opacity分布
            select_mask = (scene.gaussians._trbf_center < (5/300)).squeeze()
            wandb.log({
                # 'scene/first_5_opacity_histogram':wandb.Histogram(scene.gaussians.get_opacity[select_mask].cpu().numpy()),
                'scene/first_5_points_num':select_mask.sum().item(),
            },step=iteration)

            #对比几个时间段的tgrad
            # t_grad = scene.gaussians.t_gradient_accum / scene.gaussians.denom
            # t_grad[t_grad.isnan()] = 0.0
            # t_grad_1_5 = t_grad[select_mask].mean()
            # select_mask = (torch.logical_and(scene.gaussians._trbf_center >= (100/300),scene.gaussians._trbf_center < (105/300))).squeeze()
            # t_grad_100_105= t_grad[select_mask].mean()
            # select_mask = (torch.logical_and(scene.gaussians._trbf_center >= (200/300),scene.gaussians._trbf_center < (205/300))).squeeze()
            # t_grad_200_205= t_grad[select_mask].mean()

            # wandb.log({
            #     'scene/t_grad_1_5':t_grad_1_5.item(),
            #     'scene/t_grad_100_105':t_grad_100_105.item(),
            #     'scene/t_grad_200_205':t_grad_200_205.item(),
            #     'scene/t_grad':wandb.Histogram(t_grad.cpu().numpy()),
            #     'scene/t_grad_mean':t_grad.mean().item(),
            # },step=iteration)
            # if iteration%20 ==0:
            # #观测scale和xyz_grad的关系
            #     scale = scene.gaussians.get_trbfscale.cpu().numpy()
            #     xyz_grad = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
            #     xyz_grad[xyz_grad.isnan()] = 0.0
            #     xyz_grad = xyz_grad.cpu().numpy()
            #     scale_bins = np.linspace(0.1,1,10)
            #     xyz_grad_means = [np.mean(xyz_grad[scale >= i -0.1 & scale < i]) for i in scale_bins]
            #     print(xyz_grad_means)
            #     if history_data is None:
            #         history_data = {"scale_keys":[],"xyz_grad_means":[]}
            #     elif "scale_keys" not in history_data or "xyz_grad_means" not in history_data:
            #         history_data["scale_keys"] = []
            #         history_data["xyz_grad_means"] = []
            #     history_data["scale_keys"].append(iteration)
            #     history_data["xyz_grad_means"].append(xyz_grad_means)
            #     wandb.log(
            #     {
            #         validation_configs['name'] + '/psnr_perframe': wandb.plot.line_series(
            #             xs=scale_bins,
            #             ys=history_data['xyz_grad_means'],
            #             keys=history_data['scale_keys'],
            #             title="viewgrad_perscale",
            #             xname="trbf_scale",
            #         )
            #     }
            # )


        if loss_dict is not None:
            loss_dict_wandb={}
            for loss_name in loss_dict.keys():
                loss_dict_wandb[f'train_loss_patches/'+loss_name[1:]+'_loss'] = loss_dict[loss_name].item() 
            # print(loss_dict_wandb)
            wandb.log(loss_dict_wandb,step=iteration)


    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        if history_data is None:
            history_data = {'psnr_perframe':[],"keys":[]}
        elif "psnr_perframe" not in history_data or "keys" not in history_data:
            history_data['psnr_perframe'] = []
            history_data["keys"] = []

        validation_configs = [{'name': 'test', 'cameras' :scene.getTestCameras()},
                            #   {"name":'train', 'cameras':[scene.getTrainCameras()[idx] for idx in train_camname_dict['cam10'][:4]]}
                              ]

        for config in validation_configs:
            render_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "renders")
            os.makedirs(render_path, exist_ok=True)
            # print(config)
            if config['cameras'] and len(config['cameras']) > 0:
                if config["name"]=="test":
                    l1_test_list = []
                    ssim_test_list = []
                    msssim_test_list = []
                    psnr_test_list=[]
                for idx,viewpoint in enumerate(tqdm(config['cameras'])):
                    # viewpoint = batch_data
                    # for viewpoint in batch_data:
                        # print(viewpoint.timestamp)
                        # if iteration not in testing_iterations:
                        #     viewpoint = validation_configs['cameras'][280]
                        gt_image = viewpoint.original_image.float().cuda()
                        viewpoint = viewpoint.cuda()
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,**renderKwargs )
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)                
                        # depth = easy_cmap(render_pkg['depth'][0])
                        # alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)

                        #这里还是不能代替test，并不能展示全部的图像,且上传非常慢。要么就不显示图像了
                        # if wd_writer and (idx %1==0):
                        #     grid = [gt_image, image, depth]
                        #     grid = make_grid(grid, nrow=2)
                        #     wandb.log({config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name): [wandb.Image(grid, caption="Ground Truth vs. Rendered")]}, step=iteration)

                            # tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                        # if config['name'] == 'test':
                        # print(psnr(image, gt_image).mean())
                        if idx%5==0:
                            #每5张保存一次
                            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

                        if config["name"]=="test":
                            psnr_test_list.append(psnr(image, gt_image).mean().double().item())
                            l1_test_list.append(l1_loss(image, gt_image).mean().double().item())
                            ssim_test_list.append(ssim(image, gt_image).mean().double().item())
                            msssim_test_list.append(msssim(image[None].cpu(), gt_image[None].cpu()))

                        if iteration not in testing_iterations:
                            break #这种情况只测试第一张图
                if config["name"]=="test":
                    psnr_test =np.mean(psnr_test_list)
                    l1_test = np.mean(l1_test_list)
                    ssim_test = np.mean(ssim_test_list)
                    msssim_test = np.mean(msssim_test_list)
                    frame_idx_list =[i for i in range(len(psnr_test_list))]
                    history_data['psnr_perframe'].append(psnr_test_list)
                    history_data['keys'].append(iteration)
                    # print(history_data['psnr_perframe'])
                    # print(history_data['keys'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if wd_writer and iteration in testing_iterations:
                        wandb.log({
                            config['name'] + '/ l1_loss': l1_test,
                            config['name'] + '/psnr': psnr_test,
                            config['name'] + '/ssim': ssim_test,
                            config['name'] + '/ msssim': msssim_test
                        }, step=iteration)
                        wandb.log(
                            {
                                config['name'] + '/psnr_perframe': wandb.plot.line_series(
                                    xs=frame_idx_list,
                                    ys=history_data['psnr_perframe'],
                                    keys=history_data['keys'],
                                    title="psnr_perframe",
                                    xname="frames",
                                )
                            }
                        )
                    ##write to json
                    full_dict ={}
                    per_view_dict = {}
                    full_dict.update({"SSIM": ssim_test.item(),
                                        "PSNR": psnr_test.item(),
                                        # "LPIPS": torch.tensor(lpipss).mean().item(),
                                        # "ssimsv2": torch.tensor(ssimsv2).mean().item(),
                                        # "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
                                        # "times": torch.tensor(times).mean().item()
                                        })
                
                    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(ssim_test_list, frame_idx_list)},
                                                                            "PSNR": {name: psnr for psnr, name in zip(psnr_test_list, frame_idx_list)},
                                                                            # "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                            # "ssimsv2": {name: v for v, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
                                                                            # "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},
                                                                            })
                
                    
                    
                    with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
                            json.dump(full_dict, fp, indent=True)

                    with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
                        json.dump(per_view_dict, fp, indent=True)
                    if config['name'] == 'test':
                        psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter,history_data
if __name__ == "__main__":
    print("current pid:",os.getpid())
    args, lp_extract, op_extract, pp_extract = getparser()
    print("start_train")

    torch.manual_seed(666)
    np.random.seed(666)

    if args.model_path == "":
        args.model_path = os.path.join("log",os.path.join(args.dataset, args.exp_name ))
    print("model_path:", args.model_path)

    wandb_run = None
    if not args.no_wandb:
        tags = ['test']
        wandb_run = wandb.init(project=args.dataset, name=args.exp_name,config=args,save_code=True,resume=False,tags=tags) #resume为true并没有什么好处
    try:
        train(lp_extract, op_extract, pp_extract, args.save_iterations,args.testing_iterations, args.debug_from, args.start_iteration,checkpoint=args.checkpoint,densify=args.densify, duration=args.duration, wandb_run=wandb_run,rgbfunction=args.rgbfunction, rdpip=args.rdpip,no_report=args.no_report)
    except Exception as e:
        print("Error during training: ", e)
        traceback.print_exc()
        wandb.finish()
        raise e
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        wandb.finish()
    # All done
    finally:
        # print("\nTraining complete.")
        if wandb_run:
            wandb.finish()
