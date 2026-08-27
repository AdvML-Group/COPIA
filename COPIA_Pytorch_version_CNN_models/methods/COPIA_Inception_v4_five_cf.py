import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from .method_base import BaseAttack

seed = 2013
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

class COPIA_Inception_v4_five_cf(BaseAttack):
    def __init__(self, model_name, target=False, epsilon=32 / 255,
                 steps=10, decay=1.0, ens=30,
                 layer_idx=[0, 1, 2, 3, 4],
                 scale=0.2, lr=0.025):
        super().__init__('DANAA_CF_INCEPTION_V4', model_name, target, epsilon, steps, decay)
        self.step_size = (epsilon / steps) * 1.2
        self.ens = ens
        self.layer_idx = layer_idx
        self.scale = scale
        self.lr = lr
        self.lam = 10.0

        if hasattr(self.model, "aux_logits"):
            self.model.aux_logits = False

    def _get_target_layer(self):
        m = self.model.model if hasattr(self.model, "model") else self.model

        all_layers = [
            m.features[2],
            m.features[4],
            m.features[5],
            m.features[9],
            m.features[10]
        ]

        return [all_layers[i] for i in self.layer_idx]

    def _get_layer_activation(self, x):
        layers = self._get_target_layer()
        feats = {}
        hooks = []

        def hook_fn(i):
            def hook(module, inp, out):
                feats[i] = out[0] if isinstance(out, tuple) else out

            return hook

        for i, l in enumerate(layers):
            hooks.append(l.register_forward_hook(hook_fn(i)))

        try:
            with torch.no_grad():
                _ = self.model(x)
            return [feats[i].detach() for i in range(len(layers))]
        finally:
            for h in hooks: h.remove()

    def _compute_danaa_attribution(self, inps, cf_inps, labels):
        self.model.eval()
        layers = self._get_target_layer()
        num_layers = len(layers)

        cf_logits = []
        with torch.no_grad():
            for j in range(5):
                o = self.model(cf_inps[:, j])
                if isinstance(o, tuple): o = o[0]
                cf_logits.append(o.detach())

        agg = [0 for _ in range(num_layers)]
        x_up = [inps.clone().detach().requires_grad_(True) for _ in range(5)]
        x_down = [inps.clone().detach().requires_grad_(True) for _ in range(5)]

        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        for _ in range(self.ens):
            for j in range(5):
                feats_up = {}
                hooks_up = []

                def get_hook_up(i):
                    def hook(m, inp, out):
                        res = out[0] if isinstance(out, tuple) else out
                        feats_up[i] = res
                        if res.requires_grad: res.retain_grad()

                    return hook

                for i, l in enumerate(layers):
                    hooks_up.append(l.register_forward_hook(get_hook_up(i)))

                noisy_up = (x_up[j] + torch.randn_like(x_up[j]) * self.scale).clamp(-1, 1).requires_grad_(True)
                out_up = self.model(noisy_up)
                if isinstance(out_up, tuple): out_up = out_up[0]

                l_mse_up = mse(out_up, cf_logits[j])
                l_ce_up = ce(out_up, labels)

                g_in_up = torch.autograd.grad(l_mse_up, noisy_up, retain_graph=True)[0]
                x_up[j] = (x_up[j] + self.lr * g_in_up.sign()).detach().requires_grad_(True)
                g_feat_up = torch.autograd.grad(l_ce_up, [feats_up[i] for i in range(num_layers)], allow_unused=True)

                for h in hooks_up: h.remove()

                feats_down = {}
                hooks_down = []

                def get_hook_down(i):
                    def hook(m, inp, out):
                        res = out[0] if isinstance(out, tuple) else out
                        feats_down[i] = res
                        if res.requires_grad: res.retain_grad()

                    return hook

                for i, l in enumerate(layers):
                    hooks_down.append(l.register_forward_hook(get_hook_down(i)))

                noisy_down = (x_down[j] + torch.randn_like(x_down[j]) * self.scale).clamp(-1, 1).requires_grad_(True)
                out_down = self.model(noisy_down)
                if isinstance(out_down, tuple): out_down = out_down[0]

                l_mse_down = mse(out_down, cf_logits[j])
                l_ce_down = ce(out_down, labels)

                g_in_down = torch.autograd.grad(l_mse_down, noisy_down, retain_graph=True)[0]
                x_down[j] = (x_down[j] - self.lr * g_in_down.sign()).detach().requires_grad_(True)
                g_feat_down = torch.autograd.grad(l_ce_down, [feats_down[i] for i in range(num_layers)],
                                                  allow_unused=True)

                for h in hooks_down: h.remove()

                for i in range(num_layers):
                    gu = g_feat_up[i] if g_feat_up[i] is not None else torch.zeros_like(agg[i])
                    gd = g_feat_down[i] if g_feat_down[i] is not None else torch.zeros_like(agg[i])
                    if isinstance(agg[i], int):
                        agg[i] = (gu + gd).detach()
                    else:
                        agg[i] += (gu + gd).detach()

        for i in range(num_layers):
            grad = agg[i] / (self.ens * 5.0)
            norm = torch.norm(grad, p=2, dim=list(range(1, grad.ndim)), keepdim=True) + 1e-8
            agg[i] = -grad / norm

        base = [0 for _ in range(num_layers)]
        for j in range(5):
            fu = self._get_layer_activation(x_up[j])
            fd = self._get_layer_activation(x_down[j])
            for i in range(num_layers):
                if isinstance(base[i], int):
                    base[i] = (fu[i] + fd[i])
                else:
                    base[i] += (fu[i] + fd[i])
        for i in range(num_layers): base[i] /= 10.0

        return agg, base

    def forward(self, inps, labels, cf_inps=None):
        inps, labels, cf_inps = inps.cuda(), labels.cuda(), cf_inps.cuda()
        agg, base = self._compute_danaa_attribution(inps, cf_inps, labels)

        momentum = torch.zeros_like(inps).cuda()
        perts = torch.zeros_like(inps).cuda().requires_grad_(True)
        layers = self._get_target_layer()
        num_layers = len(layers)

        for _ in range(self.steps):
            feats_holder = {}
            hooks = []

            def hook_fn(i):
                def hook(m, inp, out):
                    feats_holder[i] = out[0] if isinstance(out, tuple) else out
                    if feats_holder[i].requires_grad: feats_holder[i].retain_grad()

                return hook

            for i, l in enumerate(layers):
                hooks.append(l.register_forward_hook(hook_fn(i)))

            adv = inps + perts
            out = self.model(adv)
            if isinstance(out, tuple): out = out[0]

            step_agg_grad = [g.clone() for g in agg]
            total_loss = 0

            for i in range(num_layers - 1, 0, -1):
                delta = feats_holder[i] - base[i]

                if step_agg_grad[i].shape != delta.shape:
                    step_agg_grad[i] = F.interpolate(step_agg_grad[i], size=delta.shape[-2:], mode='bilinear',
                                                     align_corners=False)

                loss_i = torch.sum(delta * step_agg_grad[i])

                loss_i = -loss_i
                total_loss += loss_i

                mid_grad = torch.autograd.grad(loss_i, feats_holder[i - 1], retain_graph=True, allow_unused=True)[0]
                if mid_grad is not None:
                    if step_agg_grad[i - 1].shape != mid_grad.shape:
                        mid_grad = F.interpolate(mid_grad, size=step_agg_grad[i - 1].shape[-2:], mode='bilinear',
                                                 align_corners=False)

                    norm = torch.norm(mid_grad, p=2, dim=list(range(1, mid_grad.ndim)), keepdim=True) + 1e-8
                    mid_grad_normalized = mid_grad / norm

                    step_agg_grad[i - 1] = step_agg_grad[i - 1] + 1.0 * mid_grad_normalized

            delta_0 = feats_holder[0] - base[0]
            if step_agg_grad[0].shape != delta_0.shape:
                step_agg_grad[0] = F.interpolate(step_agg_grad[0], size=delta_0.shape[-2:], mode='bilinear',
                                                 align_corners=False)
            loss_0 = torch.sum(delta_0 * step_agg_grad[0])

            loss_0 = -loss_0
            total_loss += loss_0

            grad = torch.autograd.grad(total_loss, perts)[0]
            for h in hooks: h.remove()

            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            grad = grad + momentum * self.decay
            momentum = grad

            perts.data = perts.data + self.step_size * grad.sign()
            perts.data = torch.clamp(perts.data, -self.epsilon, self.epsilon)
            perts.data = torch.clamp(inps + perts.data, -1, 1) - inps

        return (inps + perts).detach(), None
