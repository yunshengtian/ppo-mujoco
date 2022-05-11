import torch
import numpy as np
import augmentations.rad as rad
import augmentations.new_augs as new_augs

aug_to_func = {
    # 'crop': rad.random_crop,
    'grayscale': rad.random_grayscale,
    'cutout': rad.random_cutout,
    'cutout_color': rad.random_cutout_color,
    # 'flip': rad.random_flip,
    'rotate': rad.random_rotation,
    'rand_conv': rad.random_convolution,
    # 'color_jitter': rad.random_color_jitter,
    # 'translate': rad.random_translate,
    'no_aug': rad.no_aug,
    'blackout': new_augs.blackout
}


def create_aug_func_dict(augs_list):
    augs_func_dict = dict()
    for aug_name in augs_list:
        assert aug_name in aug_to_func.keys()
        augs_func_dict[aug_name] = aug_to_func[aug_name]

    return augs_func_dict


class Augmenter:
    def __init__(self, cfg: dict) -> None:
        augs = cfg['train']['augmentation']['augs']
        self.probs_list = cfg['train']['augmentation']['distribution']

        if len(augs) != len(self.probs_list):
            raise ValueError(
                'Len of list of augs does not equal number of bins in Categorical distribution')

        self.batch_sz = cfg['train']['augmentation']['batch_sz']
        self.is_full = cfg['train']['augmentation']['is_full']

        self.augs = dict()

        if augs == 'rad':
            self.augs = aug_to_func
        elif not augs or isinstance(augs, str):
            raise ValueError(
                f'Augs should string: "rad" or non-empty list not: {augs}')
        else:
            self.augs = create_aug_func_dict(augs_list=augs)

        self.aug_keys = list(self.augs.keys())

    def augment_tensors_in_batches(self, input):
        if self.is_full:
            self.batch_sz = input.shape[0]

        idxes_arr = np.random.choice(
            input.shape[0], self.batch_sz, replace=False)
        inputs_augmented = torch.clone(input)

        for idx in idxes_arr:
            sampled_aug = np.random.choice(
                self.aug_keys, 1, p=self.probs_list)[0]
            print(f"Applying aug: {sampled_aug}")
            inputs_augmented[idx] = self.augs[sampled_aug](input[idx].unsqueeze(dim=0)).squeeze(0)

        return inputs_augmented

    def augment_tensors(self, input):
        sampled_idxes = self.sampling_distribution.sample(
            sample_shape=torch.zeros(input.shape[0]).shape)
        unique_idxes = torch.unique(input=sampled_idxes).tolist()
        obs_augmented = torch.clone(input)
        augs_list = list(self.augs.keys())

        for idx in unique_idxes:
            aug = self.augs[augs_list[idx]]
            selected_idxes = (sampled_idxes == idx).nonzero().flatten()
            tnsr_augmented = aug(input[selected_idxes])

            assert tnsr_augmented.dtype is torch.IntTensor

            obs_augmented[selected_idxes] = tnsr_augmented

        return obs_augmented
