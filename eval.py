
import os
import time

import torch
from pytorch_msssim import ssim
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader, test_data
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f

def _eval(model, args):
    # state_dict = torch.load(args.test_model)
    # model.load_state_dict(state_dict['model'])

    checkpoint = torch.load(args.test_model)
    # delete unmatched total_ops total_params
    state_dict = []
    for n, p in checkpoint['model'].items():
        if "total_ops" not in n and "total_params" not in n:
            state_dict.append((n, p))
    state_dict = dict(state_dict)
    
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    factor = 8
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            h_gt, w_gt = label_img.shape[2], label_img.shape[3]
            H_gt, W_gt = ((h_gt + factor) // factor) * factor, ((w_gt + factor) // factor * factor)
            padh_gt = H_gt - h_gt if h_gt % factor != 0 else 0
            padw_gt = W_gt - w_gt if w_gt % factor != 0 else 0
            label_img_ = f.pad(label_img, (0, padw_gt, 0, padh_gt), 'reflect')

            pred = model(input_img,label_img_)[0][2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)


            label_img = (label_img).cuda()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            ssim_adder(ssim_val)
           
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            print('%d iter PSNR: %.2f SSIM: %f' % (iter_idx + 1, psnr, ssim_val))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f' % (ssim_adder.average()))

# 去噪test
def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_data(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    factor = 8
    runtime=Adder()
    with torch.no_grad():

        for iter_idx, data in enumerate(dataloader):
            input_img, name = data
            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            start_time = time.time()
            pred = model.test(input_img)[2]
            end_time = time.time()
            runtime_per_bbox = end_time - start_time
            runtime(runtime_per_bbox)
            pred = pred[:,:,:h,:w]
            pred_clip = torch.clamp(pred, 0, 1)
            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

        print('==========================================================')
        print(runtime.average())