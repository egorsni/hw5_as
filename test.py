import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
import torchaudio
import numpy as np

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

def load_audio(path):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    target_sr = 16000
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def main(config, audios_dir, out_file, model_path):
#     logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_arch)
#     logger.info(model)

#     logger.info("Loading checkpoint: {} ...".format(config.resume))
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    print(model_path)
    model.load_state_dict(torch.load(model_path)["state_dict"])

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    index = []
    files = []
    
    results = []
    with torch.no_grad():
        for file in os.listdir(audios_dir):
            audio_wave = load_audio(os.path.join(audios_dir, file))
            index.append(audio_wave)
            files.append(file)
            audio = audio_wave[0].tolist()
            if len(audio) > 64000:
                start = np.random.randint(low = 0, high = len(audio) - 64000)
                audio = audio[start:start+64000]
            else:
                audio = np.tile(audio, 64000 // len(audio) + 10)[:64000]
            print(audio_wave.shape)
            res = model(torch.tensor([audio]).to(device))
            
            print(res.shape)
            print(res)
            results.append(res[0].tolist())
    print(results)
    results = torch.nn.functional.softmax(torch.tensor(results), dim=-1).tolist()
    with open(out_file, 'w') as f:
        f.write('FILENAME' + '|' + "SPOOF PROB" + '|' + "BONAFIDE PROB" + '\n')
        for file, file_res in zip(files, results):
            f.write(str(file) + '|' + str(file_res[0]) + '|' + str(file_res[1]) + '\n')
    
    


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--audios",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
#     model_config = Path(args.resume).parent / "config.json"
    with open(args.config) as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
#     if args.test_data_folder is not None:
#         test_data_folder = Path(args.test_data_folder).absolute().resolve()
#         assert test_data_folder.exists()
#         config.config["data"] = {
#             "test": {
#                 "batch_size": args.batch_size,
#                 "num_workers": args.jobs,
#                 "datasets": [
#                     {
#                         "type": "CustomDirAudioDataset",
#                         "args": {
#                             "audio_dir": str(test_data_folder / "audio"),
#                             "transcription_dir": str(
#                                 test_data_folder / "transcriptions"
#                             ),
#                         },
#                     }
#                 ],
#             }
#         }

#     assert config.config.get("data", {}).get("test", None) is not None
#     config["data"]["test"]["batch_size"] = args.batch_size
#     config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.audios, args.output, args.resume)
