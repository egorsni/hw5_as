import logging
from typing import List
import torch

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
    result_batch = {}
    audio_length = []
    spectrogram_length = []
    texts_length = []
    
    for item in dataset_items:
        audio_length.append(item['audio'].shape[1])
        spectrogram_length.append(item['spectrogram'].shape[2])
        texts_length.append(item['text_encoded'].shape[1])
        
    audios = torch.zeros((len(audio_length), max(audio_length)))
    spectrograms = torch.zeros((len(audio_length), 128, max(spectrogram_length)))
    texts_encoded = torch.zeros((len(audio_length), max(texts_length)))
    texts = []
    paths = []
    
    for i, item in enumerate(dataset_items):
        audios[i, :audio_length[i]] = item['audio']
        spectrograms[i,:, :spectrogram_length[i]] = item['spectrogram']
        texts_encoded[i, :texts_length[i]] = item['text_encoded']
        texts.append(item['text'])
        paths.append(item['audio_path'])
    
    return {
        "audio" : audios,
        "spectrogram" : spectrograms,
        "text" : texts,
        "text_encoded" : texts_encoded,
        "text_encoded_length" : torch.tensor(texts_length),
        "spectrogram_length" : torch.tensor(spectrogram_length),
        "audio_path" : paths
            }