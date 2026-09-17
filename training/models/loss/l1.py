import torch


class BinaryTruncatedL1:
    """

    Args:
        force_negative_t (bool): force negative values to go below -t instead of just t
        trunc_term (float): truncation term to prevent overfit
    """

    def __init__(self, force_negative_t, trunc_term=0.5):
        self.fnt = 1 if force_negative_t else -1
        self.t = trunc_term

    def __call__(self, preds, targets):
        neg_scores = preds[targets == 0]
        pos_scores = preds[targets > 0]
        # neg tries to be below -self.t if "force_negative_t" is True
        # otherwise just try to go below self.
        neg_loss = torch.clip(neg_scores + self.fnt * self.t, min=0)
        # pos tries to be above self.t
        pos_loss = torch.clip(-pos_scores + self.t, min=0)

        if len(neg_loss) > 0:
            neg_loss = torch.mean(neg_loss)
        else:
            neg_loss = 0
        if len(pos_loss) > 0:
            pos_loss = torch.mean(pos_loss)
        else:
            pos_loss = 0

        return neg_loss + pos_loss
