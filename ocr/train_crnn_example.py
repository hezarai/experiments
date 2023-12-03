import os

from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import ImageProcessor, ImageProcessorConfig
from hezar.data import OCRDataset, OCRDatasetConfig
from hezar.trainer import Trainer, TrainerConfig
import pandas as pd


class CRNNOCRDataset(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, split=None, **kwargs):
        super().__init__(config, split, **kwargs)

    def _load(self, split=None):
        def _resolve_path(path):
            return f"{os.path.abspath(self.config.path)}/{path}"

        annotations = self.config.annotation_file
        dataset = pd.read_csv(annotations)
        dataset["path"] = dataset["path"].apply(_resolve_path)
        dataset = dataset.dropna()
        return dataset

    def __getitem__(self, index):
        path, text = self.data.iloc[index]
        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"][0]
        labels = self._text_to_tensor(text)
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs


preprocessor = ImageProcessor(
    ImageProcessorConfig(
        mean=[0.6595],
        std=[0.1501],
        rescale=0.00392156862745098,
        size=(384, 32),
        mirror=True,
        gray_scale=True,
    )
)

train_dataset = CRNNOCRDataset(
    OCRDatasetConfig(
        path="./data/fa-mixed",
        text_split_type="char_split",
        annotation_file="./annotations_train.csv",
        image_processor_config=preprocessor.config,
        test_split_size=0.1,
        max_length=48,
        reverse_digits=True,
    ),
    split="train",
)

eval_dataset = CRNNOCRDataset(
    OCRDatasetConfig(
        path="./ocr/data/fa-mixed",
        text_split_type="char_split",
        annotation_file="./annotations_test.csv",
        image_processor_config=preprocessor.config,
        test_split_size=0.1,
        max_length=48,
        reverse_digits=True,
    ),
    split="test",
)

model = CRNNImage2Text(
    CRNNImage2TextConfig(
        id2label=train_dataset.config.id2label,
        map2seq_in_dim=1024,
        map2seq_out_dim=96,
    ),
)

train_config = TrainerConfig(
    output_dir="crnn-fa-printed-parsynth-4m",
    task="image2text",
    device="cuda",
    batch_size=128,
    num_dataloader_workers=16,
    num_epochs=5,
    metrics=["cer"]
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)

trainer.train()
