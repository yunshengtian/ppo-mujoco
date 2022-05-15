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

        self.augs = dict()

        if 'crop' in augs:
            self.is_crop = True

        if 'translate' in augs:
            self.is_translate = True

        if augs == 'rad':
            self.augs = aug_to_func
            self.is_crop = True
            self.is_translate = True
        elif not augs or isinstance(augs, str):
            raise ValueError(
                f'Augs should string: "rad" or non-empty list not: {augs}')
        elif cfg['algorithm'] == 'Aug_PPO':
            raise NotImplementedError
        else:
            self.augs = create_aug_func_dict(augs_list=augs)

        self.aug_keys = list(self.augs.keys())

    def augment_tensors_in_batches(self, input):
        if self.is_full:
            self.batch_sz = input.shape[0]
            idxes_arr = np.arange(input.shape[0])
        else:
            idxes_arr = np.random.choice(
                input.shape[0], self.batch_sz, replace=False)

        inputs_augmented = torch.zeros(
            tuple([self.batch_sz] + list(input.shape[1:])))
        
        # TODO: Optimize this

        for i in range(len(idxes_arr)):
            idx = idxes_arr[i]
            sampled_aug = np.random.choice(
                self.aug_keys, 1, p=self.probs_list)[0]
            print(f"Applying aug: {sampled_aug}")
            inputs_augmented[i] = self.augs[sampled_aug](
                input[idx].unsqueeze(dim=0)).squeeze(0)

        return inputs_augmented.to(device=self.device)

    def augment_tensors(self, input):
        sampled_idxes = self.sampling_distribution.sample(
            sample_shape=torch.zeros(input.shape[0]).shape)
        unique_idxes = torch.unique(input=sampled_idxes).tolist()
        obs_augmented = torch.clone(input)
        augs_list = list(self.augs.keys())

        for idx in unique_idxes:
            aug = self.augs[augs_list[idx]]
            selected_idxes = (sampled_idxes == idx).nonzero().flatten()

            input_selected = input[selected_idxes]

            if self.is_crop:
                input_selected = rad.random_crop(input_selected)
            elif self.is_translate:
                input_selected = rad.random_translate(input_selected)

            tnsr_augmented = aug(input_selected)

            assert tnsr_augmented.dtype is torch.IntTensor

            obs_augmented[selected_idxes] = tnsr_augmented

        return obs_augmented.to(device=self.device)
