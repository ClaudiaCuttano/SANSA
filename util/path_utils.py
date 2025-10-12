SAM2_WEIGHTS_URL = {
    'tiny':  'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',
    'base':  'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
    'large': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
}
SAM2_PATHS_CONFIG = {
    'tiny':  ('pretrain/sam2_hiera_tiny.pt', '../sam2/sam2_configs/sam2_hiera_t.yaml'),
    'base':  ('pretrain/sam2_hiera_base_plus.pt', '../sam2/sam2_configs/sam2_hiera_b+.yaml'),
    'large': ('pretrain/sam2_hiera_large.pt', '../sam2/sam2_configs/sam2_hiera_l.yaml')
}


