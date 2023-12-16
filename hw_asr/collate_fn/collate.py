import logging
from typing import List
import torch
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
#     print(dataset_items[0])

#     print(dataset_items[0]['audio'].shape)
#     print(dataset_items[1]['audio'].shape)
#     print(dataset_items[2]['audio'].shape)
#     print()
#     print(dataset_items[0]['spectrogram'].shape)
#     print(dataset_items[1]['spectrogram'].shape)
#     print(dataset_items[2]['spectrogram'].shape)
    
    audios_cropped = []
    labels = []
    
    for item in dataset_items:
        audio = item["audio"][0].tolist()
        
        if len(audio) > 64000:
            start = np.random.randint(low = 0, high = len(audio) - 64000)
            audios_cropped.append(audio[start:start+64000])
        else:
            audio = np.tile(audio, 64000 // len(audio) + 10)[:64000]
            audios_cropped.append(audio)
            
        labels.append(1 if item['label'] == 'bonafide' else 0)
    
    return {
        "audios" : torch.tensor(audios_cropped),
        "labels" : torch.tensor(labels)
            }