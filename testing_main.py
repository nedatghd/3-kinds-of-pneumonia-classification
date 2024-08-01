### IMPORT LIBRARIES ###
from libs import *
### IMPORT LIBRARIES ###
# from libs import *
import torch
from vit_pytorch.cct import CCT
from vit_pytorch import SimpleViT


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
        param.requires_grad = False
    
    preprocess = weights.transforms()
    
    # model name; number of output features, and model specific preprocess
    return model, num_features, preprocess
    
class FourPneumonia(nn.Module):
    def __init__(self, base_model, input_features, batch_size):
        super(FourPneumonia, self).__init__()
        
        # Use the provided base model
        self.base_model = base_model       
        self.batch_size = batch_size
        
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

        x = self.c1(patient_features)
        x = self.output(x)
        return x
        

# class_names = train_dataset.class_names

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
    BATCH_SIZE = args.batch_size
    BASE_MODEL_NAME = args.base_model_name
    CHECKPOINT_PATH = args.checkpoint_file
    RESULTS = args.results_dir
    IMAGES_DIR = args.images_dir

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Load the saved model weights
    # base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)
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

    model = SimpleViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 4,
    dim = 512,
    depth = 4, #4
    heads = 16,
    mlp_dim = 1024).to(device)



    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint, strict=True)
    
    print("Model Loaded Successfully")
#    if torch.cuda.device_count() > 1:
#        print("Multiple GPU Detected")
#        print(f"Using {torch.cuda.device_count()} GPUs")
#        model = nn.DataParallel(model)
    test_transform = test_transform = torchvision.transforms.Compose([
                                      torchvision.transforms.Resize(size = (224, 224)),
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dataset = PatientDataset(images_dir=IMAGES_DIR, 
                            classes = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"],
                            transform=test_transform, 
                            target_transform=None)

    
    dataset_length = len(dataset)
    print("Test length", dataset_length)

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Ensure model is in evaluation mode and on the correct device       
    model.eval()
    
    y_pred = []
    y_true = []
    accuracy = 0.0 
    # Make predictions on the test dataset
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predicting"):
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    classes = dataset.classes
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    image_file_name_CM = RESULTS+"/CM{BASE_MODEL_NAME}"
    model_title = BASE_MODEL_NAME

    accuracy += sum((y_pred == y_true))
    accuracy = accuracy / len(test_loader.dataset)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("-"*90)
    print("Derived Report")
    print("-"*90)
    print("%s%.2f%s"% ("Accuracy      : ", accuracy*100, "%"))
    print("%s%.2f%s"% ("Precision     : ", precision*100, "%"))
    print("%s%.2f%s"% ("Recall        : ", recall*100,    "%"))
    print("%s%.2f%s"% ("F1-Score      : ", f1*100,        "%"))
    print("-"*90)
    print("\n\n")

    CM = confusion_matrix(y_true, y_pred)

    fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10,8), hide_ticks=True,cmap=plt.cm.Blues)
    plt.xticks(range(len(classes)), classes, fontsize=12)
    plt.yticks(range(len(classes)), classes, fontsize=12)
    plt.title("Confusion Matrix for Model File (Test Dataset): \n"+model_title, fontsize=11)
    fig.savefig(image_file_name_CM, dpi=100)
    plt.show()


    cls_report_print = classification_report(y_true, y_pred, target_names=classes)

    cls_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    print("\n\n")
    print("-"*90)
    print("Report for Model File: ", model_title)
    print("-"*90)
    print(cls_report_print)
    print("-"*90)
    print("Testing complete.")


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Testing Script')
    
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size')
    parser.add_argument('--base_model_name', type=str, default="resnet18", help='Base Model Name')
    parser.add_argument('--checkpoint_file', default='./saved_model/best_model.pth', help='Path to save the model checkpoint in .pth')
    parser.add_argument('--results_dir', default="./saved_result/", help='Location for saving details and results')
    parser.add_argument('--images_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")

    
    args = parser.parse_args()
    
    main(args)
