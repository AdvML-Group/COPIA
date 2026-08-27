import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

from dataset import params
from model import get_model

class BaseAttack(object):
    def __init__(self, attack_name, model_name, target, epsilon=16/255, steps=10, decay=1.0):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        self.loss_flag = -1 if self.target else 1

        self.epsilon = epsilon
        self.steps = steps
        self.decay = decay
        self.step_size = self.epsilon / self.steps

        self.used_params = params(self.model_name)
        self.model = get_model(self.model_name)

        self.model.cuda()
        self.model.eval()
        self.loss_fn = nn.CrossEntropyLoss()

        self.hook_handles = []

        self.seed = 3407
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def _register_hooks(self):
        pass

    def _remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles = []

    def forward(self, inps, labels):
        self._register_hooks()

        try:
            inps = inps.cuda()
            labels = labels.cuda()

            momentum = torch.zeros_like(inps).cuda()
            unnorm_inps = self._mul_std_add_mean(inps)
            perts = torch.zeros_like(unnorm_inps).cuda()
            perts.requires_grad_()

            for i in range(self.steps):
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))
                cost = self.loss_flag * self.loss_fn(outputs, labels).cuda()
                cost.backward()
                grad = perts.grad.data
                grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
                grad += momentum * self.decay
                momentum = grad
                perts.data = self._update_perts(perts.data, grad, self.step_size)
                perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
                perts.grad.data.zero_()

            return (self._sub_mean_div_std(unnorm_inps + perts.data)).detach(), None

        finally:
            self._remove_hooks()

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype, device=inps.device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.used_params['std'], dtype=dtype, device=inps.device).view(1, -1, 1, 1)
        return inps * std + mean

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype, device=inps.device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.used_params['std'], dtype=dtype, device=inps.device).view(1, -1, 1, 1)
        return (inps - mean) / std

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)

        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1, 2, 0])
            image[image < 0] = 0
            image[image > 1] = 1
            image = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))

            image.save(save_path)

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images
