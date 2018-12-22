import torch
import torch.nn as nn
import torch.nn.functional as f






































class OldGeneratorLoss(nn.Module):
    """Combined loss over image and features"""
    
    def __init__(self, 
                 discriminator, 
                 valid=1, 
                 hole=6, 
                 perceptual=0.05, 
                 style=120, 
                 tv=0.1, 
                 confidence=1):

        super(GeneratorLoss, self).__init__()
        self.__dict__.update(locals())
        self.elem_loss = ElementwiseLoss()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.variation_loss = TotalVariationLoss()


    def forward(self,
                generated,
                original,
                mask):

        self.discriminator.eval()

        composite = mask*original + (1-mask)*generated

        org_conf, org_feats = self.discriminator(original)
        gen_conf, gen_feats = self.discriminator(generated)
        cmp_conf, cmp_feats = self.discriminator(composite)

        valid = self.elem_loss((1 - mask) * generated, (1 - mask) * original)
        hole = self.elem_loss(mask * generated, mask * original)
        
        p_gen = self.perceptual_loss(gen_feats, org_feats)
        p_cmp = self.perceptual_loss(cmp_feats, org_feats)
        
        s_gen = self.style_loss(gen_feats, org_feats)
        s_cmp = self.style_loss(cmp_feats, org_feats)

        tv = self.tv*self.variation_loss(composite)

        ones = torch.ones_like(org_conf)
        conf = self.elem_loss(org_conf, ones) + self.elem_loss(cmp_conf, ones)

        print(f'valid: {valid}')
        print(f'hole: {hole}')
        print(f'p_gen: {p_gen}')
        print(f'p_cmp: {p_cmp}')
        print(f's_gen: {s_gen}')
        print(f's_cmp: {s_cmp}')
        print(f'tv: {tv}')

        loss = self.valid*valid + \
                self.hole*hole + \
                self.perceptual*(p_gen+p_cmp) + \
                self.style*(s_gen + s_cmp) + \
                self.tv*tv + \
                self.confidence * conf

        self.discriminator.train()

        return loss

class OldDiscriminatorLoss(nn.Module):
    """Loss designed to measure discriminator performance."""

    def __init__(self, confidence=1, feats=1e-4):
        super(DiscriminatorLoss, self).__init__()
        self.__dict__.update(locals())
        self.elem_loss = ElementwiseLoss()
        self.perceptual = PerceptualLoss()

    def forward(self, c_gen, c_org, f_gen, f_org):
        ones = torch.ones_like(c_gen)
        zeros = torch.zeros_like(c_gen)
        conf_loss = self.elem_loss(c_gen, zeros) + self.elem_loss(c_org, ones)
        feats_loss = self.perceptual(f_gen, f_org)

        return self.confidence * conf_loss + self.feats * feats_loss

###############################################################################
# Loss Components                                                             #
###############################################################################

class ElementwiseLoss(nn.Module):
    """Perform an elementwise loss over a pair of images."""

    def __init__(self):
        super(ElementwiseLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return f.l1_loss(y_pred, y_true)


class PerceptualLoss(nn.Module):
    """Elementwise loss over a pair of activation layers."""

    def __init__(self,):
        super(PerceptualLoss, self).__init__()

    def forward(self, pred_feats, orig_feats):
        lst = [
            f.l1_loss(pred_feats[i], orig_feats[i]).unsqueeze(0) for i in range(len(pred_feats))
        ]

        return torch.sum(torch.stack(lst))


class StyleLoss(nn.Module):
    """L1 loss over a set of autocorrelated features."""

    def __init__(self,):
        super(StyleLoss, self).__init__()

    def forward(self, pred_feats, orig_feats):
        pred_grams = self.__gram(pred_feats)
        true_grams = self.__gram(orig_feats)

        loss = 0
        for i in range(len(pred_grams)):
            h, w = pred_grams[i].size()
            print(h, w)
            loss += f.l1_loss(pred_grams[i], true_grams[i]).div(h * w)

        return loss

    def __gram(self, feats):
        if feats[0].dim() == 2:
            return feats

        for i in range(len(feats)):
            b, c, h, w = feats[i].size()
            feats[i] = feats[i].reshape(b * c, h * w)
            feats[i] = torch.mm(feats[i].t(), feats[i])

        return feats


class TotalVariationLoss(nn.Module):
    """Loss to encourage smoother images"""

    def __init__(self,):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        batches, channels, height, width = image.size()
        left_shift = image.clone()
        left_shift[..., :width - 1] = left_shift[..., 1:width]
        left_shift[..., width - 1] = 0

        up_shift = image.clone()
        up_shift[:, :, :height - 1, :] = up_shift[:, :, 1:height, :]
        up_shift[:, :, height - 1, :] = 0

        return f.l1_loss(left_shift, image) + f.l1_loss(up_shift, image)
