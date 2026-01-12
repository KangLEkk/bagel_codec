import io
import json
import random
from PIL import Image

import pyarrow.parquet as pq

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list,
        num_used_data,
        vit_transform=None,
        fixed_prompt=None,

        # ✅ new: json map {image_path: [prompt,...]}
        json_map_path=None,

        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.fixed_prompt = fixed_prompt
        self.data_status = data_status

        self.json_map_path = json_map_path
        self._json_map = None

        if self.json_map_path:
            # json-map mode: data_paths are image paths for sharding
            with open(self.json_map_path, "r", encoding="utf-8") as f:
                self._json_map = json.load(f)

            # keys are absolute paths; shard by worker/rank using base class
            self.data_paths = list(self._json_map.keys())
        else:
            # parquet mode (original)
            self.data_paths = get_parquet_data_paths(data_dir_list, num_used_data)

        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()

        # resume cursor
        if self.data_status is not None:
            start_a = self.data_status[worker_id][0]
            start_b = self.data_status[worker_id][1]
            start_c = self.data_status[worker_id][2] + 1
        else:
            start_a = 0
            start_b = 0
            start_c = 0

        vae_stride = self.transform.stride
        vit_stride = self.vit_transform.stride if self.vit_transform is not None else None

        # -----------------------------
        # Mode A: json_map_path
        # -----------------------------
        if self.json_map_path:
            print(
                f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name} [json_map]: "
                f"resuming at idx#{start_a}"
            )
            img_paths = data_paths_per_worker[start_a:]

            while True:
                for local_idx, img_path in enumerate(img_paths, start=start_a):
                    try:
                        image = pil_img2rgb(Image.open(img_path))
                    except Exception as e:
                        print(f"[json_map] open failed: {img_path} err={e}")
                        continue

                    # prompts
                    if self.fixed_prompt:
                        caption_token = self.tokenizer.encode(str(self.fixed_prompt))
                    else:
                        prompts = self._json_map.get(img_path, None)
                        if not prompts:
                            caption_token = self.tokenizer.encode(" ")
                        else:
                            caption = random.choice(prompts)
                            caption_token = self.tokenizer.encode(caption)

                    # two-branch transforms
                    vae_image_tensor = self.transform(image)
                    H, W = vae_image_tensor.shape[1:]
                    n_vae = (H * W) // (vae_stride ** 2)

                    vit_image_tensor = None
                    n_vit = 0
                    if self.vit_transform is not None:
                        vit_image_tensor = self.vit_transform(image)
                        Hv, Wv = vit_image_tensor.shape[1:]
                        n_vit = (Hv * Wv) // (vit_stride ** 2)

                    # Bagel-style plan: text(cond, no loss) -> vit_image(cond, no loss) -> vae_image(target, loss=1)
                    sequence_plan = [
                        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                    ]
                    if vit_image_tensor is not None:
                        sequence_plan.append(
                            {"type": "vit_image", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None}
                        )
                    sequence_plan.append(
                        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None}
                    )

                    image_tensor_list = []
                    if vit_image_tensor is not None:
                        image_tensor_list.append(vit_image_tensor)
                    image_tensor_list.append(vae_image_tensor)

                    text_ids_list = [caption_token]
                    num_tokens = len(caption_token) + n_vit + n_vae

                    yield dict(
                        image_tensor_list=image_tensor_list,
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            # 对齐 parquet 的 3-tuple 结构：这里用 (img_idx, 0, 0)
                            "data_indexes": [local_idx, 0, 0],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )

                # repeat
                start_a = 0
                img_paths = data_paths_per_worker
                print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id} [json_map]")

        # -----------------------------
        # Mode B: parquet (original)
        # -----------------------------
        parquet_start_id = start_a
        row_group_start_id = start_b
        row_start_id = start_c

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name} [parquet]: "
            f"resuming at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            try:
                                image_byte = row["image"]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                print(f"[parquet] image decode err={e} rg#{row_group_id} file={parquet_file_path}")
                                continue

                            # caption
                            if self.fixed_prompt:
                                caption_token = self.tokenizer.encode(str(self.fixed_prompt))
                            else:
                                try:
                                    caption_dict = json.loads(row["captions"])
                                    caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                                    caption_token = random.choice(caps_token) if len(caps_token) else self.tokenizer.encode(" ")
                                except Exception as e:
                                    print(f"[parquet] caption err={e} rg#{row_group_id} file={parquet_file_path}")
                                    continue

                            # transforms
                            vae_image_tensor = self.transform(image)
                            H, W = vae_image_tensor.shape[1:]
                            n_vae = (H * W) // (vae_stride ** 2)

                            vit_image_tensor = None
                            n_vit = 0
                            if self.vit_transform is not None:
                                vit_image_tensor = self.vit_transform(image)
                                Hv, Wv = vit_image_tensor.shape[1:]
                                n_vit = (Hv * Wv) // (vit_stride ** 2)

                            sequence_plan = [
                                {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                            ]
                            if vit_image_tensor is not None:
                                sequence_plan.append(
                                    {"type": "vit_image", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None}
                                )
                            sequence_plan.append(
                                {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None}
                            )

                            image_tensor_list = []
                            if vit_image_tensor is not None:
                                image_tensor_list.append(vit_image_tensor)
                            image_tensor_list.append(vae_image_tensor)

                            text_ids_list = [caption_token]
                            num_tokens = len(caption_token) + n_vit + n_vae

                            yield dict(
                                image_tensor_list=image_tensor_list,
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                },
                            )

                        row_start_id = 0
                    row_group_start_id = 0

            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id} [parquet]")
