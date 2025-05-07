import os.path as osp
import random
import numpy as np
import pandas as pd


def train_test_split_t2vqa_db(
    dataset_path, ann_file, split="8-1-1", seed=42, debias=False, hard_train=False
):
    random.seed(seed)
    ratio = [int(s) for s in split.split("-")]
    ratio = [r / sum(ratio) for r in ratio]
    assert len(ratio) == 3, "split must be a string like '8-1-1'"
    video_infos = []
    if hard_train:
        val_files = []
        hard_samples = np.load(
            "/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/hard_samples.npy"
        )
        with open(
            "/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/val.lst",
            "r",
        ) as fval:
            for line in fval.readlines():
                filename = line.strip()
                val_files.append(filename)
        train_split = []
        val_split = []
        test_split = []
        with open(ann_file, "r") as fin:
            for line in fin.readlines():
                line_split = line.strip().split("|")
                filename, prompt, label = line_split
                label = np.array([float(label)], dtype=np.float32)
                filename = osp.join(dataset_path, filename)
                if filename in hard_samples:
                    train_split.append(
                        dict(filename=filename, prompt=prompt, label=label)
                    )
                if filename in val_files:
                    val_split.append(
                        dict(filename=filename, prompt=prompt, label=label)
                    )
                    test_split.append(
                        dict(filename=filename, prompt=prompt, label=label)
                    )

    else:
        # with open(ann_file, "r") as fin:
        #     for line in fin.readlines():
        #         line_split = line.strip().split("|")
        #         filename, prompt, label = line_split
        #         label = np.array([float(label)], dtype=np.float32)
        #         filename = osp.join(dataset_path, filename)
        #         video_infos.append(dict(filename=filename, prompt=prompt, label=label))
        # random.shuffle(video_infos)
        # train_split = video_infos[: int(ratio[0] * len(video_infos))]
        # val_split = video_infos[int(ratio[0] * len(video_infos)) : int((ratio[0] + ratio[1]) * len(video_infos))]+video_infos[int((ratio[0] + ratio[1]) * len(video_infos)) :]
        # test_split = video_infos[int(ratio[0] * len(video_infos)) : int((ratio[0] + ratio[1]) * len(video_infos))]+video_infos[int((ratio[0] + ratio[1]) * len(video_infos)) :]
        train_data = pd.read_csv(
            "/data0/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/2_train.csv"
        )
        val_data = pd.read_csv(
            "/data0/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/2_val.csv"
        )
        train_split = []
        val_split = []
        for i in range(len(train_data)):
            filename = train_data["vid_name"][i]
            filename = osp.join(dataset_path, filename)
            prompt = train_data["prompt"][i]
            label = train_data["label"][i]
            label = np.array([float(label)], dtype=np.float32)
            train_split.append(dict(filename=filename, prompt=prompt, label=label))
        for i in range(len(val_data)):
            filename = val_data["vid_name"][i]
            filename = osp.join(dataset_path, filename)
            prompt = val_data["prompt"][i]
            label = val_data["label"][i]
            label = np.array([float(label)], dtype=np.float32)
            val_split.append(dict(filename=filename, prompt=prompt, label=label))
        test_split = val_split.copy()
        # debiasing: accordning to the proportion, increasing the number of low and high quality videos
        if debias:
            import matplotlib.pyplot as plt

            # s1: split data into 5 groups, with quality range [0-19, 20-39, 40-59, 60-79, 80-100]
            levels = {
                "l1": [0, 20],
                "l2": [20, 40],
                "l3": [40, 60],
                "l4": [60, 80],
                "l5": [80, 101],
            }
            video_new = {"l1": [], "l2": [], "l3": [], "l4": [], "l5": []}
            for vi in train_split:
                label = vi["label"]
                for k, v in levels.items():
                    if label >= v[0] and label < v[1]:
                        video_new[k].append(vi)
            # s2: calculate the number of each group
            num_samples = [len(video_new[k]) for k in video_new.keys()]
            # s3: increase the number of other groups to the same number as the group with the most samples
            max_num = max(num_samples)
            for k, v in video_new.items():
                if len(v) < max_num:
                    video_new[k] = v * (max_num // len(v))
            # s4: combine the new groups
            video_debias = []
            for k, v in video_new.items():
                video_debias += v
            # s5: shuffle the new data
            random.shuffle(video_debias)
            # s6: calculate the number of each group
            plt.figure()
            plt.bar(video_new.keys(), num_samples)
            plt.xlabel("Quality Levels")
            plt.ylabel("Number of Samples")
            plt.savefig("quality_levels.png")
            plt.close()
            return (
                video_debias,
                val_split,
                test_split,
            )

    return (
        train_split,
        val_split,
        test_split,
    )


def train_test_split_lgvq_db(
    dataset_path, ann_file, split="8-1-1", seed=42, debias=False, hard_train=False
):
    random.seed(seed)
    ratio = [int(s) for s in split.split("-")]
    ratio = [r / sum(ratio) for r in ratio]
    assert len(ratio) == 3, "split must be a string like '8-1-1'"
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(";")
            filename,qs,qt,qa = line_split
            prompt = filename.split("/")[-1].rstrip('.mp4')
            [qs,qt,qa] = [np.array([float(q)], dtype=np.float32) for q in [qs,qt,qa]]
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=[qs,qt,qa]))
    random.shuffle(video_infos)
    train_split = video_infos[: int(ratio[0] * len(video_infos))]
    val_split = video_infos[int(ratio[0] * len(video_infos)) : int((ratio[0] + ratio[1]) * len(video_infos))]+video_infos[int((ratio[0] + ratio[1]) * len(video_infos)) :]
    test_split = video_infos[int(ratio[0] * len(video_infos)) : int((ratio[0] + ratio[1]) * len(video_infos))]+video_infos[int((ratio[0] + ratio[1]) * len(video_infos)) :]
    return (
        train_split,
        val_split,
        test_split,
    )

if __name__ == "__main__":
    dataset_path = "/data0/xxy/data/LGVQ/videos"
    ann_file = "/data0/xxy/data/LGVQ/MOS.txt"
    train_split, val_split, test_split = train_test_split_lgvq_db(
        dataset_path, ann_file, split="8-1-1", seed=42, debias=False, hard_train=False
    )
    print(len(train_split), len(val_split), len(test_split))
    # check if the video files exist
    for vi in train_split:
        filename = vi["filename"]
        if not osp.exists(filename):
            print(f"File {filename} does not exist")
    for vi in val_split:
        filename = vi["filename"]
        if not osp.exists(filename):
            print(f"File {filename} does not exist")