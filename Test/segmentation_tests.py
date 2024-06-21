import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def segmentation_tests(test_dataset, model_output):
    def jaccard_index(target, pred, smooth=1e-10):
        intersection = (pred.int() & target.int()).sum((1))
        union = (pred.int() | target.int()).sum((1))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def correct_predictions(target, pred):
        intersection = (target == 1) & (pred == 1)
        matches = (intersection.sum(dim=(1, 2)) > 0).sum().item()
        percentage_matches = (matches / target.size(0)) * 100
        return percentage_matches
    
    # Get n_classes
    n_classes = model_output.size(1)-5

    # Get prediction mask
    mask = model_output[:, 4:n_classes+4]

    # Get original images
    if n_classes > 1:
        sim = []
        for i in range(2):
            s = [item[i+1] for item in test_dataset] # By using item[0] we could get the ground truth directly from any image
            sim.append(torch.stack(s))
        sim = torch.stack(sim, dim=1)
        sim = torch.squeeze(sim, dim=2)
    
    else:
        sim = [item[1] for item in test_dataset]
        sim = torch.stack(sim)

    # Downsize the image to 48x48
    reduced_images = F.interpolate(sim, size=(48, 48), mode='bilinear', align_corners=False)

    # Binarize tensors
    reduced_images = (reduced_images[:] > 0.1).float()

    # Plot the overlapping images
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.subplot(2, 4, i + 1)
        plt.imshow(reduced_images[0,i].squeeze(), cmap="gray", origin="lower")
        plt.imshow(mask[0,i].squeeze(), cmap="jet", alpha=0.5, origin="lower")
    plt.tight_layout()
    plt.show()

    for i in range(n_classes):
        print(f"Jaccard Index for class {i+1}: {jaccard_index(reduced_images[:, i], mask[:, i]):.4f}")
        print(f"Percentage of matches in class {i+1}: {correct_predictions(reduced_images[:, i], mask[:, i]):.0f}%")