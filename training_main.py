### IMPORT LIBRARIES ###
from libs import *
import torch
# from vit_pytorch.cct import CCT
# from vit_pytorch import SimpleViT
# from vit_pytorch.nest import NesT
import timm

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#for reproducibility 
set_seeds()

def get_base_model_and_transforms(model_name):
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "vit_l_32":
        weights = ViT_L_32_Weights.DEFAULT
        model = vit_l_32(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "vit_l_16":
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "maxvit_t":
        weights = MaxVit_T_Weights.DEFAULT
        model = maxvit_t(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # AdaptiveAvgPool2d(output_size=1)
            model.classifier[1]  # Flatten(start_dim=1, end_dim=-1)
            )
    elif model_name == "swin_b":
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights)
        num_features = model.head.in_features
        model.head = nn.Identity()
        
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Make all the layer of the base model trainable 
    for param in model.parameters():
        param.requires_grad = True
    
    preprocess = weights.transforms()
    
    # model name; number of output features, and model specific preprocess
    return model, num_features, preprocess


def collate_fn(batch):
    images, patient_id = zip(*batch)
    clipped_images = []
    for img in images:
        if len(img) > 1000:
            img = img[:1000]  # If more than 1000 patches, clip it to 1000
        clipped_images.append(img)
    return clipped_images, patient_id
    
    

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    


class FourPneumonia(nn.Module):
    def __init__(self, base_model, input_features, batch_size):
        super(FourPneumonia, self).__init__()
        
        # Use the provided base model
        self.base_model = base_model       
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=0.2)
        # 1D Convolution blocks
        self.c1 = nn.Sequential(
            torch.nn.Linear(in_features = input_features, out_features = 128),  
            nn.ReLU()
        )

      
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(in_features=128, out_features=4)
        )
        
    def forward(self, x):
        
        # print("__________________________")
        # print(x.size())
        # print("__________________________")


        patient_features = self.base_model(x)
        # print("__________________________")
        # print(patient_features.size())
        # print("__________________________")
        x = self.dropout(patient_features)

        x = self.c1(x)

        x = self.dropout(x)

        x = self.output(x)
        return x
        


def save_to_csv(filename, headers, data1, data2, data3, details):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing training details
        for key, value in details.items():
            writer.writerow([key, value])
        
        # Add an empty line
        writer.writerow([])
        
        # Headers
        writer.writerow(headers)
        
        # Get the maximum number of rows to iterate over
        max_rows = max(len(data1), len(data2), len(data3))

        for i in range(max_rows):
            row = [
                data1[i] if i < len(data1) else '',
                data2[i] if i < len(data2) else '',
                data3[i] if i < len(data3) else '',
            ]
            writer.writerow(row)


# class_names = train_dataset.class_names
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        


def show_images(images, labels, preds):
    plt.figure(figsize=(16, 9))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks = [], yticks = [])
        image = image.numpy().transpose((1, 2, 0))
        mean  = np.array([0.485, 0.456, 0.406])
        std   = np.array([0.229, 0.224, 0.225])
        image = image*std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col   = 'green'
        if preds[i] != labels[i]:
            col = 'red'
            
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color = col)
    plt.tight_layout()
    plt.show()


def show_preds():
    model.eval()
    images, labels = next(iter(dl_test))
    outputs  = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)        

        
def main(args):
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    BASE_MODEL_NAME = args.base_model_name
    CHECKPOINT_PATH = args.checkpoint_file
    RESULTS = args.results_dir
    TRAIN_IMAGES_DIR = args.train_images_dir
    VALID_IMAGES_DIR = args.valid_images_dir


    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)
  

    train_transform = torchvision.transforms.Compose([
                      torchvision.transforms.Resize(size = (128, 128)),
                      torchvision.transforms.RandomHorizontalFlip(p=0.5),
                      torchvision.transforms.ColorJitter(contrast=0.5),
                      torchvision.transforms.RandomEqualize(p=0.5),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    # train_transform2 = torchvision.transforms.Compose([
    #                   torchvision.transforms.Resize(size = (224, 224)),
    #                   torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #                   torchvision.transforms.AutoAugment(),
    #                   torchvision.transforms.ToTensor(),
    #                   torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    valid_transform = torchvision.transforms.Compose([
                      torchvision.transforms.Resize(size = (128, 128)),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    dataset_train = PatientDataset(images_dir=TRAIN_IMAGES_DIR, 
                            classes = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"],
                            transform = train_transform, 
                            target_transform=None)

    dataset_valid = PatientDataset(images_dir=VALID_IMAGES_DIR, 
                            classes = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"],
                            transform = valid_transform, 
                            target_transform=None)

    train_length = len(dataset_train)
    print("Train length", train_length)

    valid_length = len(dataset_valid)
    print("valid length", valid_length)

    
    # train_dataset, _ , _ = random_split(dataset_train,[train_length, valid_length, test_length], generator=torch.Generator().manual_seed(42))

    # _ , valid_dataset , _ = random_split(dataset_valid,[train_length, valid_length, test_length], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)



    
    
    
    # Initialize model, criterion and optimizer
    # model = FourPneumonia(base_model, input_features = num_features, batch_size=BATCH_SIZE).to(device)
  
    # model = CCT(
    # img_size = (224, 224),
    # embedding_dim = 384,
    # n_conv_layers = 2,
    # kernel_size = 7,
    # stride = 2,
    # padding = 3,
    # pooling_kernel_size = 3,
    # pooling_stride = 2,
    # pooling_padding = 1,
    # num_layers = 14,
    # num_heads = 6,
    # mlp_ratio = 3.,
    # num_classes = 4,
    # positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
    #               ).to(device)
    
    # model = SimpleViT(
    # image_size = 224,
    # patch_size = 32,
    # num_classes = 4,
    # dim = 512,
    # depth = 4, #4
    # heads = 16,
    # mlp_dim = 1024).to(device)



    # model = NesT(
    #     image_size = 224,
    #     patch_size = 4,
    #     dim = 96,
    #     heads = 3,
    #     num_hierarchies = 3,        # number of hierarchies
    #     block_repeats = (2, 2, 8),  # the number of transformer blocks at each hierarchy, starting from the bottom
    #     num_classes = 4
    # ).to(device)

    model = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=4, img_size=128, drop_rate=0.4)
    model.to(device)

#    if torch.cuda.device_count() > 1:
#        print("Multiple GPU Detected")
#        print(f"Using {torch.cuda.device_count()} GPUs")
#        model = nn.DataParallel(model)
    # Initialize lists to store the training and validation losses
    train_losses = []
    val_losses = []
    weights = torch.tensor([7.24, 2.815, 3.068, 5.560]).to(device) 

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Define the early stopping mechanism
    early_stopping = EarlyStopping(patience=10, verbose=True, path=CHECKPOINT_PATH)
    
    

    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
    
        # Training phase with progress bar
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False):
            images = images.to(device) 
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            # print("--------------")
            # print(outputs)
            # print("--------------")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * len(images)
    
            # Delete tensors to free up memory
            del images, labels, outputs
    
        torch.cuda.empty_cache()
    
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}")
    
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        accuracy = 0.0 
    
        # Validation phase with progress bar
        for images, labels in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}", leave=False):
            images = images.to(device) 
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * len(images)

            _, preds = torch.max(outputs, 1)
            accuracy += sum((preds == labels).cpu().numpy())
    
            # Delete tensors to free up memory
            del images, labels, outputs
    
        torch.cuda.empty_cache()
    
        val_loss = running_val_loss / len(valid_loader.dataset)
        accuracy = accuracy / len(valid_loader.dataset)
        val_losses.append(val_loss)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}, Accuracy : {accuracy:.4f}")
        # show_preds()

        if accuracy >= 0.95:
            print('Performance condition satisfied, stopping..')
            return
    
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Training complete.")
    
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Training and Validation Loss Curves{BASE_MODEL_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS + f'training_validation_loss_curve{BASE_MODEL_NAME}.png')
    #plt.show()

    


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Training and Testing Script')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size')
    parser.add_argument('--base_model_name', type=str, default="resnet18", help='Base Model Name')
    parser.add_argument('--checkpoint_file', default='./saved_model/best_model.pth', help='Path to save the model checkpoint in .pth')
    parser.add_argument('--results_dir', default="./saved_result/", help='Location for saving details and results')
    parser.add_argument('--train_images_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--valid_images_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")

    
    args = parser.parse_args()
    
    main(args)
