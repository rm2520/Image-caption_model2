import torch
import matplotlib.pyplot as plt
import os
import json
from torchvision import transforms



transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def visualize_log(log, log_visualize_dir):
    # Plot loss per epoch
    plt.figure()
    plt.plot(log["train_loss"], label="train")
    plt.plot(log["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per epoch")
    filename = os.path.join(log_visualize_dir, "loss_epoch.png")
    plt.savefig(filename)

    # Plot bleu4 per epoch
    plt.figure()
    plt.plot(log["train_bleu4"], label="train")
    plt.plot(log["val_bleu4"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Bleu4")
    plt.legend()
    plt.title("BLEU-4 per epoch")
    filename = os.path.join(log_visualize_dir, "bleu4_epoch.png")
    plt.savefig(filename)

    # Plot loss per batch
    plt.figure()
    train_loss_batch = []
    for loss in log["train_loss_batch"]:
        train_loss_batch += loss
    plt.plot(train_loss_batch, label="train")
    
    val_loss_batch = []
    for loss in log["val_loss_batch"]:
        val_loss_batch += loss
    plt.plot(val_loss_batch, label="val")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per batch")
    filename = os.path.join(log_visualize_dir, "loss_batch.png")
    plt.savefig(filename)


#from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap

def metric_scores(annotation_path, prediction_path):
    # annotation_file = "captions_val2014.json"
    # results_file = "captions_val2014_fakecap_results.json"
    # format results_file
    # {"image_id": 1, "caption": "a caption"}

    results = {}
    coco = COCO(annotation_path)
    coco_result = coco.loadRes(prediction_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
        results[metric] = score

    return results


def convert_karpathy_to_coco_format(karpathy_path, annotation_path, phase="test"):
    # phase in {"train", "val", "test"}
    phase = {"train", "restval"} if phase == "train" else {phase}
    tst_split="test"
    flicker = json.load(open(annotation_path))
    flicker_tst = json.load(open(annotation_path))
    karpathy = json.load(open(karpathy_path))

    karpathy_ids = set([x["imgid"] for x in karpathy["images"] if x["split"] in phase])
    karpathy_ids_tst = set([x["imgid"]  for x in karpathy["images"] if x["split"] in tst_split])

    flicker_tst["images"] = [x for x in flicker["images"] if x["imgid"] in karpathy_ids_tst]
    flicker["images"] = [x for x in flicker["images"] if x["imgid"] in karpathy_ids]

    df = []
    tst=[]
    for image in flicker["images"]:
       for sent in image["sentences"]:
           annotations= {
               "image_id":sent["imgid"],
               "id":sent["sentid"],
               "caption":sent["raw"]
           }
           df.append(annotations)
    for image in flicker_tst["images"]:
        for sent in image["sentences"]:
            raw={
                    "image_id":sent["imgid"] ,
                    "caption":sent["raw"]

            }
            tst.append(raw)



    flicker["annotations"] = [x for x in df]
    flicker["tst"]=[x for x in tst]
    return flicker