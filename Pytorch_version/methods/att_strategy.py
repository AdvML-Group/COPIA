import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

class ATTUtils:
    @staticmethod
    def tr_01_pc(num, length):
        rate_l = num
        tensor = torch.cat((torch.ones(rate_l), torch.zeros(length - rate_l)))
        return tensor

class ATTStrategy:
    def __init__(self, model, model_name, gamma=0.5, lam=10.0):
        self.model = model
        self.model_name = model_name
        self.gamma = gamma
        self.lam = lam

        self.hooks = []
        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        self.back_attn_idx = 0
        self.truncate_layers = None
        self.weaken_factor = []

        self._init_config()

    def _init_config(self):
        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            self.back_attn_idx = 11
            self.truncate_layers = ATTUtils.tr_01_pc(10, 12)
            self.weaken_factor = [0.45, 0.7, 0.65]
        elif self.model_name == 'pit_b_224':
            self.back_attn_idx = 12
            self.truncate_layers = ATTUtils.tr_01_pc(9, 13)
            self.weaken_factor = [0.25, 0.6, 0.65]
        elif self.model_name == 'cait_s24_224':
            self.back_attn_idx = 24
            self.truncate_layers = ATTUtils.tr_01_pc(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
        elif self.model_name == 'visformer_small':
            self.back_attn_idx = 7
            self.truncate_layers = ATTUtils.tr_01_pc(8, 8)
            self.weaken_factor = [0.4, 0.8, 0.3]
        else:
            print(f"Warning: Model {self.model_name} not explicitly supported. Using defaults.")
            self.back_attn_idx = 11
            self.truncate_layers = ATTUtils.tr_01_pc(10, 12)
            self.weaken_factor = [0.45, 0.7, 0.65]

    def _calculate_gpf(self, out_grad, running_var):
        if running_var != 0:
            var_curr = torch.var(out_grad)
            if var_curr == 0: var_curr = 1e-6
            gpf = (self.gamma + self.lam * (1 - torch.sqrt(var_curr / (running_var + 1e-8)))).clamp(0, 1)
        else:
            gpf = self.gamma
        return gpf

    def _apply_spatial_mask(self, out_grad, gpf):
        grad_clone = out_grad.detach()
        B = grad_clone.shape[0]

        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224', 'visformer_small', 'pit_b_224']:
            C, H, W = grad_clone.shape[1], grad_clone.shape[2], grad_clone.shape[3]
            out_grad_cpu = grad_clone.cpu().numpy().reshape(B, C, H * W)

            max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
            min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)

            max_H, max_W = max_all // H, max_all % H
            min_H, min_W = min_all // H, min_all % H

            out_grad[:, range(C), max_H, :] *= gpf
            out_grad[:, range(C), :, max_W] *= gpf
            out_grad[:, range(C), min_H, :] *= gpf
            out_grad[:, range(C), :, min_W] *= gpf

        elif self.model_name in ['cait_s24_224']:
            H, W, C = grad_clone.shape[1], grad_clone.shape[2], grad_clone.shape[3]
            out_grad_cpu = grad_clone.cpu().numpy().reshape(B, H * W, C)

            max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

            max_H, max_W = max_all // H, max_all % H
            min_H, min_W = min_all // H, min_all % H

            out_grad[:, max_H, :, range(C)] *= gpf
            out_grad[:, :, max_W, range(C)] *= gpf
            out_grad[:, min_H, :, range(C)] *= gpf
            out_grad[:, :, min_W, range(C)] *= gpf

        return out_grad

    def attn_ATT(self, module, grad_in, grad_out):
        mask = torch.ones_like(grad_in[0]) * self.truncate_layers[self.back_attn_idx] * self.weaken_factor[0]
        out_grad = mask * grad_in[0][:]
        gpf = self._calculate_gpf(out_grad, self.var_A)
        out_grad = self._apply_spatial_mask(out_grad, gpf)

        self.var_A = torch.var(out_grad).detach()
        self.back_attn_idx -= 1
        if self.back_attn_idx < 0: self.back_attn_idx = 0
        return (out_grad,)

    def attn_cait_ATT(self, module, grad_in, grad_out):
        mask = torch.ones_like(grad_in[0]) * self.truncate_layers[self.back_attn_idx] * self.weaken_factor[0]
        out_grad = mask * grad_in[0][:]

        gpf = self._calculate_gpf(out_grad, self.var_A)

        B, H, W, C = grad_in[0].shape
        out_grad_cpu = out_grad.detach().cpu().numpy()
        max_all = np.argmax(out_grad_cpu[0, :, 0, :], axis=0)
        min_all = np.argmin(out_grad_cpu[0, :, 0, :], axis=0)

        out_grad[:, max_all, :, range(C)] *= gpf
        out_grad[:, min_all, :, range(C)] *= gpf

        self.var_A = torch.var(out_grad).detach()
        self.back_attn_idx -= 1
        if self.back_attn_idx < 0: self.back_attn_idx = 0
        return (out_grad,)

    def q_ATT(self, module, grad_in, grad_out):
        mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
        out_grad = mask * grad_in[0][:]

        gpf = self._calculate_gpf(out_grad, self.var_qkv)
        out_grad[:] *= gpf

        self.var_qkv = torch.var(out_grad).detach()
        return (out_grad, grad_in[1], grad_in[2])

    def v_ATT(self, module, grad_in, grad_out):
        mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
        out_grad = mask * grad_in[0][:]
        gpf = self._calculate_gpf(out_grad, self.var_qkv)

        if self.model_name in ['visformer_small']:
             B, C, H, W = grad_in[0].shape
             out_grad_cpu = out_grad.detach().cpu().numpy().reshape(B, C, H * W)
             max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
             min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
             max_H, max_W = max_all // H, max_all % H
             min_H, min_W = min_all // H, min_all % H

             out_grad[:, range(C), max_H, max_W] *= gpf
             out_grad[:, range(C), min_H, min_W] *= gpf

        elif self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224', 'pit_b_224', 'cait_s24_224']:
            c = grad_in[0].shape[2]
            out_grad_cpu = out_grad.detach().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

            out_grad[:, max_all, range(c)] *= gpf
            out_grad[:, min_all, range(c)] *= gpf

        self.var_qkv = torch.var(out_grad).detach()
        return (out_grad, grad_in[1])

    def mlp_ATT(self, module, grad_in, grad_out):
        mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
        out_grad = mask * grad_in[0][:]
        gpf = self._calculate_gpf(out_grad, self.var_mlp)

        if self.model_name in ['visformer_small']:
             B, C, H, W = grad_in[0].shape
             out_grad_cpu = out_grad.detach().cpu().numpy().reshape(B, C, H * W)
             max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
             min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
             max_H, max_W = max_all // H, max_all % H
             min_H, min_W = min_all // H, min_all % H
             out_grad[:, range(C), max_H, max_W] *= gpf
             out_grad[:, range(C), min_H, min_W] *= gpf

        elif self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
            c = grad_in[0].shape[2]
            out_grad_cpu = out_grad.detach().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
            out_grad[:, max_all, range(c)] *= gpf
            out_grad[:, min_all, range(c)] *= gpf

        self.var_mlp = torch.var(out_grad).detach()
        if len(grad_in) > 1:
            return (out_grad,) + grad_in[1:]
        else:
            return (out_grad,)

    def __enter__(self):
        self.hooks = []

        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        self._init_config()

        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.hooks.append(self.model.blocks[i].attn.attn_drop.register_backward_hook(self.attn_ATT))
                self.hooks.append(self.model.blocks[i].attn.qkv.register_backward_hook(self.v_ATT))
                self.hooks.append(self.model.blocks[i].mlp.register_backward_hook(self.mlp_ATT))

        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind, used_block_ind = 0, block_ind
                elif block_ind < 9:
                    transformer_ind, used_block_ind = 1, block_ind - 3
                else:
                    transformer_ind, used_block_ind = 2, block_ind - 9

                blk = self.model.transformers[transformer_ind].blocks[used_block_ind]
                self.hooks.append(blk.attn.attn_drop.register_backward_hook(self.attn_ATT))
                self.hooks.append(blk.attn.qkv.register_backward_hook(self.v_ATT))
                self.hooks.append(blk.mlp.register_backward_hook(self.mlp_ATT))

        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.hooks.append(self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(self.attn_ATT))
                    self.hooks.append(self.model.blocks[block_ind].attn.qkv.register_backward_hook(self.v_ATT))
                    self.hooks.append(self.model.blocks[block_ind].mlp.register_backward_hook(self.mlp_ATT))
                else:
                    blk = self.model.blocks_token_only[block_ind - 24]

                    self.hooks.append(blk.attn.attn_drop.register_backward_hook(self.attn_cait_ATT))
                    self.hooks.append(blk.attn.q.register_backward_hook(self.q_ATT))
                    self.hooks.append(blk.attn.k.register_backward_hook(self.v_ATT))
                    self.hooks.append(blk.attn.v.register_backward_hook(self.v_ATT))
                    self.hooks.append(blk.mlp.register_backward_hook(self.mlp_ATT))

        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    blk = self.model.stage2[block_ind]
                else:
                    blk = self.model.stage3[block_ind - 4]
                self.hooks.append(blk.attn.attn_drop.register_backward_hook(self.attn_ATT))
                self.hooks.append(blk.attn.qkv.register_backward_hook(self.v_ATT))
                self.hooks.append(blk.mlp.register_backward_hook(self.mlp_ATT))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []
