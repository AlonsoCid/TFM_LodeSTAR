import torch

def get_euclidean_distance(test_dataset, detect_results, model_output):

    # Get segmentation masks
    n_classes = model_output.size(1)-5
    mask = model_output[:, 4:n_classes+4]

    # Get gt positions
    positions = [item[-n_classes:] for item in test_dataset]
    positions = [torch.stack(item) for item in positions]
    positions = torch.stack(positions)
    positions = positions / 2

    # x, y = positions[0][0]
    # print(positions[0][0])
    print(mask.shape)
    
    # Create a tensor to store the distances
    n_samples = len(positions)
    n_classes = n_classes
    distances = torch.zeros((n_samples, n_classes, 1))

    # Make calculations
    for sample in range(n_samples):
        for class_ in range(n_classes):
            x = positions[sample][class_]

            # Check if all positions in the image have been predicted, if the aren't skip the sample
            if class_ >= len(detect_results[sample]):
                continue
            
            y = detect_results[sample][class_]

            # # Check that tensor y is not empty
            # if len(y) == 0:
            #     continue
            y = torch.from_numpy(y)
            
            # Check that the class has been detected 
            a, b = x
            if mask[sample][class_][int(a), int(b)] != 1:
                continue

            distance = torch.sqrt(torch.sum((x - y)**2))
            distances[sample][class_] = distance
    return distances