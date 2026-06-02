<!-- # Image Captioning
Image Captioning by deep learning model with Encoder as Efficientnet and Decoder as Decoder of Transformer -->
#

#  Objective

The objective of this project is to build a model that can generate captions for images.

The directory structure of this project is shown below:
```bash
root/
├── data/
│   
│
└── image_captioning/ # this repository
    ├── images/
    ├── pretrained/
    ├── results/
    ├── caption.py
    ├── datasets.py
    ├── evaluation.py
    ├── models.py
    ├── README.md
    ├── train.py
    └── utils.py
```
#  Model

I use Encoder as Efficientnet to extract features from image and Decoder as Transformer to generate caption.I use the Attention mechanism each step to attend image features.


#  Dataset

- ERI-AIC (Proposed Dataset)
- Arabic Flickr8k
- Arabic MS-COCO


## Experimental Results

### Arabic Flickr8k

| BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
| 49.8   | 12.2   | 0.362  | 0.330 | 0.37  |

### Arabic MS-COCO
| BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
| 43.2   | 8.2    | 0.310  | 0.280 | 0.33 |

### ERI-AIC (Proposed Dataset)
| BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE |
| 65.1   | 21.4   |0.430   | 0.380 | 0.39  |

#  Training and Validation
##  Pre-processing
###  Images
I preprocessed the images with the following steps:
- Resize the images to 256x256 pixels.
- Convert the images to RGB.
- Normalize the images with mean and standard deviation.
I normalized the image by the mean and standard deviation of the ImageNet images' RGB channels.

```




##  Training
###  Model configs

- embedding_dim: 512
- vocab_size: 30522
- max_seq_len: 128
- encoder_layers: 6
- decoder_layers: 12
- num_heads: 8
- dropout: 0.1

###  Hyperparameters

- n_epochs: 25
- batch_size: 24
- learning_rate: 1e-4
- optimizer: Adam
- adam parameters: betas=(0.9, 0.999), eps=1e-9
- loss: CrossEntropyLoss
- metric: bleu-4
- early_stopping: 5

## 5.3. Validation
I evaluate the model on the validation set after each epoch. For each image, I generate a caption and evaluate the BLEU-4 score with list of reference captions by sentence_bleu. And for all the images, I calculate the BLEU-4 score with the corpus_bleu function from NLTK.

You can see the detaile in the `train.py` file. Run `train.py` to train the model.
```bash
python train.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --batch_size 24 \
    --n_epochs 25 \
    --learning_rate 1e-4 \
    --early_stopping 5 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --log_path ./images/log_training.json \
    --log_visualize_dir ./images/
```

# 6. Evaluation
See the `evaluation.py` file. Run `evaluation.py` to evaluate the model.

```bash
python evaluation.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --output_dir ./results/
