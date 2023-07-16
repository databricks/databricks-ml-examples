# Databricks notebook source
# install Hugging Face Libraries
%pip install  git+https://github.com/huggingface/peft.git
# install Hugging Face Libraries
%pip install  bitsandbytes

# COMMAND ----------

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from peft import LoraConfig, get_peft_model


# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q", "v", "q_proj", "v_proj"]
)

# COMMAND ----------

model_id = "Salesforce/blip2-flan-t5-xl"
# We load our model and processor using `transformers`
model = AutoModelForVision2Seq.from_pretrained(model_id, load_in_8bit=True,device_map='auto')
processor = AutoProcessor.from_pretrained(model_id)

# COMMAND ----------

from datasets import load_dataset
fashion_dir = '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_images/'
dataset = load_dataset("imagefolder", data_dir=fashion_dir, split="train")

# COMMAND ----------

dataset

# COMMAND ----------

dataset = dataset.shuffle(seed=42).shuffle(seed=24)

# COMMAND ----------

dataset[100]

# COMMAND ----------

from PIL import Image
display(dataset[100]['image'])

# COMMAND ----------

# Get our peft model and print the number of trainable parameters
model = get_peft_model(model, config)
model.print_trainable_parameters()

# COMMAND ----------

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding


def collator(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


# COMMAND ----------

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#Do not bother doing this without a GPU. It will be excrutiatingly slow and resource intensive
device = "cuda"# if torch.cuda.is_available() else "cpu"

# COMMAND ----------

#model_save_dir = '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_10_epoch/'
model_save_dir = '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_30_epoch/'

# COMMAND ----------

model.train()

for epoch in range(30):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    model.save_pretrained(model_save_dir+str(epoch))



# COMMAND ----------

processor.save_pretrained('/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_30_epoch/processor/')

# COMMAND ----------

#epoch2_loc = '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model/1'

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# COMMAND ----------

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from peft import LoraConfig, get_peft_model
model_id = "Salesforce/blip2-flan-t5-xl"

# We load our model and processor using `transformers`
model = AutoModelForVision2Seq.from_pretrained(model_id, load_in_8bit=True,device_map='auto')
processor = AutoProcessor.from_pretrained('/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_10_epoch/processor/')

# COMMAND ----------

model_finetuned = PeftModel.from_pretrained(model, '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_30_epoch/29')


# COMMAND ----------

from PIL import Image
import requests
#img_url = "https://www.jcrew.com/s7-img-facade/AS211_RD5697_m?hei=850&crop=0,0,680,0"
img_url = "https://images.unsplash.com/photo-1618517351616-38fb9c5210c6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bWFuJTIwdCUyMHNoaXJ0fGVufDB8fDB8fHww&w=1000&q=80"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
display(raw_image)

# COMMAND ----------

model_finetuned = PeftModel.from_pretrained(model, '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_model_30_epoch/10')


# COMMAND ----------

inputs = processor(raw_image,return_tensors="pt").to('cuda', torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=150)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
