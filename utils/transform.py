from torchvision import transforms

def get_transforms(dataset_name, pretrained=False):
    if pretrained:
        # Normalizzazione per backbones pre-addestrate (ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == "cifar10":
        # Normalizzazione per CIFAR-10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
    elif dataset_name == "cifar100":
        # Normalizzazione per CIFAR-100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    # Trasformazioni per il training
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Trasformazioni per il test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform_train, transform_test