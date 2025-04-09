import os.path as osp
import random
import numpy as np


def train_test_split_t2vqa_db(dataset_path, ann_file, split="8-1-1", seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, label = line_split
            label = np.array([float(label)], dtype=np.float32)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    ratio = [int(s) for s in split.split("-")]
    ratio = [r / sum(ratio) for r in ratio]
    assert len(ratio) == 3, "split must be a string like '8-1-1'"
    return (
        video_infos[: int(ratio[0] * len(video_infos))],
        video_infos[
            int(ratio[0] * len(video_infos)) : int(
                (ratio[0] + ratio[1]) * len(video_infos)
            )
        ]+
        video_infos[int((ratio[0] + ratio[1]) * len(video_infos)) :],
        []
    )
