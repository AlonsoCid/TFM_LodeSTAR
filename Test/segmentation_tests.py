import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage

def segmentation_tests(test_dataset, model_output):
    def jaccard_index(target, pred, smooth=1e-10):
        intersection = (pred.int() & target.int()).sum((1))
        union = (pred.int() | target.int()).sum((1))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    # def recall(target, pred): # This one check for ground truth and mask overlaping, the new one uses positions which is more accurate and we need it for multiple object detection
    #     intersection = (target == 1) & (pred == 1)
    #     matches = (intersection.sum(dim=(1, 2)) > 0).sum().item()
    #     percentage_matches = (matches / target.size(0)) * 100
    #     return percentage_matches

    def recall_and_false_positives(mask, position):
        true_positives = 0
        artifacts = 0
    
        for i, single_mask in enumerate(mask):
            x, y = position[i]

            # Check for true positive
            if single_mask[int(x), int(y)] == 1:
                true_positives += 1
        
            # Count false positives
            labeled_array, num_regions = ndimage.label(single_mask)
            artifacts += num_regions
    
        recall = (true_positives / len(mask)) * 100
        false_positives = (artifacts - true_positives)/len(mask)
    
        return recall, false_positives
    
    # Get n_classes
    n_classes = model_output.size(1)-5

    # Get prediction mask
    mask = model_output[:, 4:n_classes+4]

    # Get original images
    sim =[]
    for i in range(n_classes):
        s = [item[i+1] for item in test_dataset] # By using item[0] we could get the ground truth directly from any image
        sim.append(torch.stack(s))
    sim = torch.stack(sim, dim=1)
    sim = torch.squeeze(sim, dim=2)

    # Get the positions
    positions = [item[-n_classes:] for item in test_dataset]
    positions = [torch.stack(item) for item in positions]
    positions = torch.stack(positions)
    positions = positions / 2

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
        # plt.scatter(positions[0][i][1], positions[0][i][0], c="r")
    plt.tight_layout()
    plt.show()

    for i in range(n_classes):
        print(f"Class {i+1}:")
        recall_score, fp_count = recall_and_false_positives(mask[:,i], positions[:,i])
        print(f"Jaccard Index: {jaccard_index(reduced_images[:, i], mask[:, i]):.4f}")
        print(f"Recall %: {recall_score:.0f}%")
        print(f"Average number of artifacts: {fp_count}")