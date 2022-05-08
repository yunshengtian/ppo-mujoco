import torch
import augmentations.rad as rad


class Augmenter:
    def __init__(self, cfg: dict) -> None:
        augs = cfg['train']['augmentation']['augs']
        probs_list = cfg['train']['augmentation']['distribution']
        self.sampling_distribution = torch.distributions.Categorical(
            probs=torch.tensor(probs_list))

        if len(augs) != len(probs_list):
            raise ValueError(
                'Len of list of augs does not equal number of bins in Categorical distribution')

        self.augs = dict()

        aug_to_func = {
            'crop': rad.random_crop,
            'grayscale': rad.random_grayscale,
            'cutout': rad.random_cutout,
            'cutout_color': rad.random_cutout_color,
            'flip': rad.random_flip,
            'rotate': rad.random_rotation,
            'rand_conv': rad.random_convolution,
            'color_jitter': rad.random_color_jitter,
            'translate': rad.random_translate,
            'no_aug': rad.no_aug,
        }

        if augs == 'rad':
            self.augs = aug_to_func
        elif not augs or isinstance(augs, str):
            raise ValueError(
                f'Augs should string: "rad" or non-empty list not: {augs}')
        else:
            for aug_name in augs:
                assert aug_name in aug_to_func.keys()
                self.augs[aug_name] = aug_to_func[aug_name]

    def augment_tensors(self, input):
        sampled_idxes = self.sampling_distribution.sample(
            sample_shape=torch.zeros(input.shape[0]).shape)
        unique_idxes = torch.unique(input=sampled_idxes).tolist()
        obs_augmented = torch.clone(input)
        augs_list = list(self.augs.keys())

        for idx in unique_idxes:
            aug = self.augs[augs_list[idx]]
            selected_idxes = (sampled_idxes == idx).nonzero().flatten()
            obs_augmented[selected_idxes] = aug(input[selected_idxes])

        return obs_augmented
