import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        
        index = []
        assert part in ['train', 'eval', 'dev']
        
        part2 = 'trn' if part=='train' else 'trl'
        
        with open(f'./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.{part2}.txt', 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            s = line[:-1].split(' ')
            flac_path = f'./LA/ASVspoof2019_LA_{part}/flac/' + s[1] + '.flac'
            t_info = torchaudio.info(str(flac_path))
            length = t_info.num_frames / t_info.sample_rate
            index.append({
                'speaker_id': s[0],
                'path': flac_path,
                'class': s[-1],
                'audio_len': length
            })
            
        super().__init__(index, *args, **kwargs)