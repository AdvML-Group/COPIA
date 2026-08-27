import torch
import torch.nn as nn

class LibraGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        gate = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        return x * gate.detach()

class LibraLayerNorm(nn.Module):
    def __init__(self, original_ln):
        super().__init__()
        self.weight = original_ln.weight
        self.bias = original_ln.bias
        self.eps = original_ln.eps
        self.normalized_shape = original_ln.normalized_shape

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        std = (var + self.eps).sqrt().detach()
        x_norm = (x - mean) / std
        return x_norm * self.weight + self.bias

class LibraAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LibraCaiTClassAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.q = original_attn.q
        self.k = original_attn.k
        self.v = original_attn.v
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.detach()

        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls

class LibraTalkingHeadAttn(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_l = original_attn.proj_l
        self.proj_w = original_attn.proj_w
        self.proj_drop = original_attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LibraVisformerAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LibraStrategy:
    def __init__(self, model, model_name='vit_base'):
        self.model = model
        self.model_name = model_name.lower()
        self.original_modules = {}
        self.patched_modules = []
        self.is_active = False

    def __enter__(self):
        if not self.is_active:
            self._replace_modules(self.model, prefix='')
            self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_active:
            self._restore_modules()
            self.is_active = False

    def _replace_modules(self, module, prefix=''):
        for name, child in module.named_children():
            fullname = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (nn.GELU, nn.SiLU)):
                self.original_modules[fullname] = child
                setattr(module, name, LibraGELU())
                self.patched_modules.append(fullname)

            elif isinstance(child, nn.LayerNorm):
                self.original_modules[fullname] = child
                setattr(module, name, LibraLayerNorm(child))
                self.patched_modules.append(fullname)

            elif child.__class__.__name__ in ['Attention', 'TalkingHeadAttn']:
                self.original_modules[fullname] = child
                if child.__class__.__name__ == 'TalkingHeadAttn':
                    setattr(module, name, LibraTalkingHeadAttn(child))
                elif 'visformer' in self.model_name:
                    setattr(module, name, LibraVisformerAttention(child))
                else:
                    setattr(module, name, LibraAttention(child))
                self.patched_modules.append(fullname)

            elif child.__class__.__name__ in ['ClassAttention', 'ClassAttn']:
                self.original_modules[fullname] = child
                setattr(module, name, LibraCaiTClassAttention(child))
                self.patched_modules.append(fullname)

            else:
                self._replace_modules(child, prefix=fullname)

    def _restore_modules(self):
        for fullname in reversed(self.patched_modules):
            parent = self.model
            if '.' in fullname:
                parent_path, child_name = fullname.rsplit('.', 1)
                for part in parent_path.split('.'):
                    parent = getattr(parent, part)
            else:
                child_name = fullname

            setattr(parent, child_name, self.original_modules[fullname])

        self.original_modules.clear()
        self.patched_modules.clear()
