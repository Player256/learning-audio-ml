import torch
import torchaudio
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoTokenizer

# Don't import MimiModel at module level - only when needed
# This avoids import-time PyTorch detection issues in newer transformers


class MimiSTTDataset(IterableDataset):

    def __init__(self, dataset_stream, mimi_model, mimi_feature_extractor, tokenizer,
                 max_audio_len=20.0, target_sample_rate=24000):
        super().__init__()
        self.dataset = dataset_stream
        self.mimi = mimi_model
        self.feature_extractor = mimi_feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len
        self.target_sample_rate = target_sample_rate

        self.SOT = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.EOT = tokenizer.eos_token_id
        self.TRANSCRIBE = tokenizer.convert_tokens_to_ids("<|transcribe|>")

        print(f"Special tokens: SOT={self.SOT}, EOT={self.EOT}, TRANSCRIBE={self.TRANSCRIBE}")

    def __iter__(self):
        for example in self.dataset:
            try:
                result = self.process_example(example)
                if result is not None:
                    yield result
            except Exception as e:
                print(f"Skipping example due to error: {e}")
                continue

    def process_example(self, sample):
        """Process one example: audio -> Mimi codes, text -> tokens."""

        audio = torch.from_numpy(sample['audio_filepath']['array']).float()
        sample_rate = sample['audio_filepath']['sampling_rate']

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio = resampler(audio)

        max_samples = int(self.max_audio_len * self.target_sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        
        if len(audio) < self.target_sample_rate * 0.5:  
            return None

        inputs = self.feature_extractor(
            raw_audio=audio.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt"
        )

        with torch.no_grad():
            encoder_outputs = self.mimi.encode(inputs["input_values"])
            audio_codes = encoder_outputs.audio_codes.squeeze(0) 


        text_tokens = self.tokenizer.encode(sample['transcription'], add_special_tokens=False)

        if len(text_tokens) > 256:  
            return None

        input_ids = [self.SOT, self.TRANSCRIBE] + text_tokens
        labels = [self.TRANSCRIBE] + text_tokens + [self.EOT]

        return {
            'audio_codes': audio_codes,  
            'input_ids': torch.tensor(input_ids, dtype=torch.long),  
            'labels': torch.tensor(labels, dtype=torch.long),  
        }


def collate_fn(batch):
    max_audio_len = max(item['audio_codes'].shape[1] for item in batch)
    max_input_len = max(item['input_ids'].shape[0] for item in batch)
    max_label_len = max(item['labels'].shape[0] for item in batch)

    batch_size = len(batch)
    num_codebooks = batch[0]['audio_codes'].shape[0]

    audio_codes = torch.zeros(batch_size, num_codebooks, max_audio_len, dtype=torch.long)
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    labels = torch.full((batch_size, max_label_len), -100, dtype=torch.long)  

    audio_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)
    text_mask = torch.zeros(batch_size, max_input_len, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        audio_len = item['audio_codes'].shape[1]
        input_len = item['input_ids'].shape[0]
        label_len = item['labels'].shape[0]

        audio_codes[i, :, :audio_len] = item['audio_codes']
        audio_mask[i, :audio_len] = 1.0

        input_ids[i, :input_len] = item['input_ids']
        text_mask[i, :input_len] = 1.0

        labels[i, :label_len] = item['labels']

    return {
        'audio_codes': audio_codes,
        'input_ids': input_ids,
        'labels': labels,
        'audio_attention_mask': audio_mask,
        'attention_mask': text_mask,
    }


def create_dataloaders(s3_path, batch_size, num_workers=0):
    from transformers import MimiModel

    print("Loading Mimi model...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    special_tokens = {
        'additional_special_tokens': ['<|transcribe|>', '<|startoftranscript|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        print(f"Added {num_added} special tokens (new vocab size: {len(tokenizer)})")

    print(f"Loading dataset from {s3_path}...")

    train_stream = load_dataset(
        "arrow",
        data_files=f"{s3_path}/train/*.arrow",
        split="train",
        streaming=True
    ).cast_column("audio_filepath", Audio(sampling_rate=24000)).shuffle(buffer_size=5000)

    val_stream = load_dataset(
        "arrow",
        data_files=f"{s3_path}/validation/*.arrow",
        split="train",
        streaming=True
    ).cast_column("audio_filepath", Audio(sampling_rate=24000))

    train_dataset = MimiSTTDataset(train_stream, mimi, feature_extractor, tokenizer)
    val_dataset = MimiSTTDataset(val_stream, mimi, feature_extractor, tokenizer)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer, mimi
