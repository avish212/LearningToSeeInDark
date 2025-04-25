# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io, scipy.misc
import torch
import numpy as np
import rawpy
import glob

# ──────────────────────────────────────────────────────────────────────────────
# ⟵ changed: import the Tiny-U-Net wrapper
from tiny_unet import TinyUNetSID
# ──────────────────────────────────────────────────────────────────────────────

input_dir       = './dataset/Sony/short/'
gt_dir          = './dataset/Sony/long/'
checkpoint_dir  = './checkpoint/Sony/'
result_dir      = './result_Sony/'
ckpt            = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(fn)[0:5]) for fn in test_fns]

DEBUG = 0
if DEBUG:
    test_ids = test_ids[:5]

def pack_raw(raw):
    """Pack Bayer RAW → 4-channel tensor."""
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)        # subtract black level
    H, W = im.shape
    im = im.reshape(H, W, 1)

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# ⟵ changed: instantiate TinyUNetSID instead of UNetSony
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
unet   = TinyUNetSID().to(device)
unet.load_state_dict(torch.load(ckpt, map_location=device))
# ──────────────────────────────────────────────────────────────────────────────

if not os.path.isdir(os.path.join(result_dir, 'final')):
    os.makedirs(os.path.join(result_dir, 'final'))

with torch.no_grad():
    unet.eval()
    for test_id in test_ids:
        # use only the first short-exposure frame in each sequence
        for in_path in glob.glob(input_dir + f'{test_id:05d}_00*.ARW'):
            in_fn = os.path.basename(in_path)
            print(in_fn)

            gt_path = glob.glob(gt_dir + f'{test_id:05d}_00*.ARW')[0]
            in_exp  = float(in_fn[9:-5])
            gt_exp  = float(os.path.basename(gt_path)[9:-5])
            ratio   = min(gt_exp / in_exp, 300)

            raw       = rawpy.imread(in_path)
            input_raw = np.expand_dims(pack_raw(raw), 0) * ratio

            gt_raw = rawpy.imread(gt_path)
            gt_img = gt_raw.postprocess(use_camera_wb=True,
                                        half_size=False,
                                        no_auto_bright=True,
                                        output_bps=16)
            gt_img = np.expand_dims(np.float32(gt_img / 65535.0), 0)

            input_raw = np.minimum(input_raw, 1.0)
            in_tensor = torch.from_numpy(input_raw).permute(0, 3, 1, 2).to(device)
            out_tensor = unet(in_tensor)
            out_img = out_tensor.permute(0, 2, 3, 1).cpu().numpy()
            out_img = np.clip(out_img, 0, 1)[0]

            gt_img  = gt_img[0]
            scale   = raw.postprocess(use_camera_wb=True,
                                      half_size=False,
                                      no_auto_bright=True,
                                      output_bps=16)
            scale   = np.float32(scale / 65535.0)
            scale   = scale * np.mean(gt_img) / np.mean(scale)

            # save results
            scipy.misc.toimage(out_img  * 255).save(
                result_dir + f'final/{test_id:05d}_00_{int(ratio)}_out.png')
            scipy.misc.toimage(scale     * 255).save(
                result_dir + f'final/{test_id:05d}_00_{int(ratio)}_scale.png')
            scipy.misc.toimage(gt_img    * 255).save(
                result_dir + f'final/{test_id:05d}_00_{int(ratio)}_gt.png')
