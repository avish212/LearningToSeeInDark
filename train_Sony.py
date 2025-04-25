# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io, scipy.misc
import torch
from torch import optim
import numpy as np
import rawpy
import glob

# ──────────────────────────────────────────────────────────────────────────────
# ⟵ changed: use Tiny U-Net wrapper instead of UNetSony
from tiny_unet import TinyUNetSID
# ──────────────────────────────────────────────────────────────────────────────

input_dir       = './dataset/Sony/short/'
gt_dir          = './dataset/Sony/long/'
checkpoint_dir  = './result_Sony/'
result_dir      = './result_Sony/'

# get train IDs
train_fns  = glob.glob(gt_dir + '0*.ARW')
train_ids  = [int(os.path.basename(fn)[0:5]) for fn in train_fns]

ps         = 512      # patch size
save_freq  = 500

DEBUG = 0
if DEBUG:
    save_freq = 2
    train_ids = train_ids[:5]

# ────────────────────────── utility funcs ─────────────────────────────────────
def pack_raw(raw):
    """Pack Bayer RAW → 4-channel tensor."""
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)          # subtract black level
    H, W = im.shape
    im = im.reshape(H, W, 1)

    return np.concatenate((im[0:H:2, 0:W:2, :],
                           im[0:H:2, 1:W:2, :],
                           im[1:H:2, 1:W:2, :],
                           im[1:H:2, 0:W:2, :]), axis=2)

def criterion(out_img, gt_img):
    """L1 loss (same as original script)."""
    return torch.mean(torch.abs(out_img - gt_img))

# ────────────────────────── data caches ───────────────────────────────────────
gt_images     = [None] * 6000
input_images  = {k: [None] * len(train_ids) for k in ['300', '250', '100']}
g_loss        = np.zeros((5000, 1))

# resume logic
allfolders = glob.glob(result_dir + '*0')
lastepoch  = max([int(f[-4:]) for f in allfolders] + [0])

# ────────────────────────── model & optimiser ─────────────────────────────────
device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
unet     = TinyUNetSID().to(device)           # ⟵ changed
unet.train()

learning_rate = 1e-4
G_opt         = optim.Adam(unet.parameters(), lr=learning_rate)

# ────────────────────────── training loop ─────────────────────────────────────
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + f'{epoch:04d}'):
        continue                                     # already done

    if epoch > 2000:
        learning_rate = 1e-5
        for g in G_opt.param_groups:
            g['lr'] = learning_rate

    cnt = 0
    for ind in np.random.permutation(len(train_ids)):
        train_id = train_ids[ind]

        in_files = glob.glob(input_dir + f'{train_id:05d}_00*.ARW')
        in_path  = in_files[np.random.randint(0, len(in_files))]  # ⟵ fixed
        in_fn    = os.path.basename(in_path)

        gt_path  = glob.glob(gt_dir + f'{train_id:05d}_00*.ARW')[0]
        gt_fn    = os.path.basename(gt_path)

        in_exp   = float(in_fn[9:-5])
        gt_exp   = float(gt_fn[9:-5])
        ratio    = min(gt_exp / in_exp, 300)

        cnt += 1
        st  = time.time()

        # cache load
        ratio_key = str(ratio)[:3]
        if input_images[ratio_key][ind] is None:
            raw   = rawpy.imread(in_path)
            input_images[ratio_key][ind] = np.expand_dims(pack_raw(raw), 0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im     = gt_raw.postprocess(use_camera_wb=True, half_size=False,
                                        no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), 0)

        # random crop / aug
        H, W = input_images[ratio_key][ind].shape[1:3]
        xx, yy = np.random.randint(0, W-ps), np.random.randint(0, H-ps)

        inp_patch = input_images[ratio_key][ind][:, yy:yy+ps, xx:xx+ps, :]
        gt_patch  = gt_images[ind][:, yy*2:yy*2+ps*2, xx*2:xx*2+ps*2, :]

        if np.random.rand() < 0.5:
            inp_patch = np.flip(inp_patch, 1); gt_patch = np.flip(gt_patch, 1)
        if np.random.rand() < 0.5:
            inp_patch = np.flip(inp_patch, 2); gt_patch = np.flip(gt_patch, 2)
        if np.random.rand() < 0.5:
            inp_patch = np.transpose(inp_patch, (0,2,1,3))
            gt_patch  = np.transpose(gt_patch , (0,2,1,3))

        inp_patch = np.clip(inp_patch, 0, 1)
        gt_patch  = np.clip(gt_patch , 0, 1)

        in_img = torch.from_numpy(inp_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch ).permute(0,3,1,2).to(device)

        G_opt.zero_grad()
        out_img = unet(in_img)
        loss    = criterion(out_img, gt_img)
        loss.backward()
        G_opt.step()

        g_loss[ind] = loss.item()
        print(f"{epoch} {cnt}  Loss={np.mean(g_loss[g_loss>0]):.3f}  "
              f"Time={time.time()-st:.2f}s")

        # save interim results
        if epoch % save_freq == 0:
            out_np = out_img.permute(0,2,3,1).cpu().numpy()
            out_np = np.clip(out_np, 0, 1)

            out_img_path = os.path.join(result_dir, f'{epoch:04d}')
            if not os.path.isdir(out_img_path):
                os.makedirs(out_img_path)

            temp = np.concatenate((gt_patch[0], out_np[0]), axis=1)
            scipy.misc.toimage(temp * 255).save(
                os.path.join(out_img_path, f'{train_id:05d}_00_train_{int(ratio)}.jpg'))

    # end of epoch ─ save checkpoint
    torch.save(unet.state_dict(), checkpoint_dir + 'model.pth')
