import torch
import torch.nn as nn
import torch.nn.functional as f


class TotalLoss(nn.Module):
    """ Combined loss over image and features """
    def __init__(self, valid=1, hole=6, perceptual=0.05, style=120, tv=0.1):
        super(TotalLoss, self).__init__()
        self.__dict__.update(locals())
        self.elem_loss = ElementwiseLoss()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.variation_loss = TotalVariationLoss()

    def forward(self,
                y_pred,
                y_true,
                y_comp,
                feats_pred,
                feats_true,
                feats_comp,
                mask):

        loss = 0
        loss += self.valid * self.elem_loss((1 - mask) * y_pred, (1 - mask) * y_true)
        loss += self.hole * self.elem_loss(mask * y_pred, mask * y_true)
        loss += self.perceptual * self.perceptual_loss(feats_pred, feats_true)
        loss += self.perceptual * self.perceptual_loss(feats_comp, feats_true)
        loss += self.style * (self.style_loss(feats_pred, feats_true) \
                              + self.style_loss(feats_comp, feats_true))
        loss += self.tv*self.variation_loss(y_comp)

        return loss


class ElementwiseLoss(nn.Module):
    """Perform an elementwise loss over a pair of images"""

    def __init__(self,):
        super(ElementwiseLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return f.l1_loss(y_pred, y_true)


class PerceptualLoss(nn.Module):
    """Elementwise loss over a pair of activation layers"""

    def __init__(self,):
        super(PerceptualLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sum([f.l1_loss(y_pred[key], y_true[key]) for key in y_pred])


class StyleLoss(nn.Module):
    """L1 loss over a set of autocorrelated features"""

    def __init__(self,):
        super(StyleLoss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_grams = self.__gram(y_pred)
        true_grams = self.__gram(y_true)

        loss = 0
        for key in pred_grams:
            h, w = pred_grams[key].size()
            loss += f.l1_loss(pred_grams[key], true_grams[key]).div(h * w)

        return loss

    def __gram(self, feats):

        for key in feats:
            b, c, h, w = feats[key].size()
            feats[key] = feats[key].reshape(b * c, h * w)
            feats[key] = torch.mm(feats[key], feats[key].t())

        return feats


class TotalVariationLoss(nn.Module):
    """Loss to encourage smoother images"""

    def __init__(self,):
        super(TotalVariationLoss, self).__init__()

    def forward(self, y_pred):
        batches, channels, height, width = y_pred.size()
        left_shift = y_pred.clone()
        left_shift[..., :width - 1] = left_shift[..., 1:width]
        left_shift[..., width - 1] = 0

        up_shift = y_pred.clone()
        up_shift[:, :, :height - 1, :] = up_shift[:, :, 1:height, :]
        up_shift[:, :, height - 1, :] = 0

        return f.l1_loss(left_shift - y_pred) + f.l1_loss(up_shift - y_pred)
