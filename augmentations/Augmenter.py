import torch
import numpy as np
import augmentations.rad as rad
import augmentations.new_augs as new_augs

aug_to_func = {
    'grayscale': rad.random_grayscale,
    'cutout': rad.random_cutout,
    'cutout_color': rad.random_cutout_color,
    'flip': rad.random_flip,
    'rotate': rad.random_rotation,
    'rand_conv': rad.random_convolution,
    # 'color_jitter': rad.random_color_jitter,
    'no_aug': rad.no_aug,
    'blackout': new_augs.blackout
}


def create_aug_func_dict(augs_list: list):
    augs_func_dict = dict()
    for aug_name in augs_list:
        if aug_name == 'crop' or aug_name == 'translate':
            continue

        assert aug_name in aug_to_func.keys()
        augs_func_dict[aug_name] = aug_to_func[aug_name]

    return augs_func_dict


def create_aug_func_list(augs_list: list):
    augs_func_list = list()
    for i in range(len(augs_list)):
        aug_name = augs_list[i]
        if aug_name == 'crop' or aug_name == 'translate':
            continue

        assert aug_name in aug_to_func.keys()
        augs_func_list.append(aug_to_func[aug_name])

    return augs_func_list


class Augmenter:
    def __init__(self, cfg: dict, device: str) -> None:
        # TODO: Add post aug sizing to CFG
        augs = cfg['train']['augmentation']['augs']
        self.probs_list = cfg['train']['augmentation']['distribution']

        if len(augs) != len(self.probs_list):
            raise ValueError(
                'Len of list of augs does not equal number of bins in Categorical distribution')

        self.batch_sz = cfg['train']['augmentation']['batch_sz']
        self.is_full = cfg['train']['augmentation']['is_full']
        self.device = device

        self.augs = list()

        if 'crop' in augs:
            self.is_crop = True

        if 'translate' in augs:
            self.is_translate = True

        if augs == 'rad':
            self.augs = [value for key, value in aug_to_func.items()]
            self.is_crop = True
            self.is_translate = True
        elif not augs or isinstance(augs, str):
            raise ValueError(
                f'Augs should string: "rad" or non-empty list not: {augs}')
        elif cfg['algorithm'] == 'Aug_PPO':
            raise NotImplementedError
        else:
            self.augs = create_aug_func_list(augs_list=augs)

        self.num_augs = len(self.augs)

    def augment_tensors_in_batches(self, input):
        if self.is_full:
            self.batch_sz = input.shape[0]
        else:
            sampled_batch_idxes = np.random.choice(
                input.shape[0], self.batch_sz)
            input = input[sampled_batch_idxes]

        sampled_idxes = np.random.choice(self.num_augs, self.batch_sz)
        unique_values = np.unique(sampled_idxes)
        input_aug = torch.clone(input).to(device=self.device)

        for value in unique_values:
            idxes_matching = np.where(sampled_idxes == value)[0]
            input_aug[idxes_matching] = self.augs[value](
                input[idxes_matching]).to(self.device)

        return input_aug
