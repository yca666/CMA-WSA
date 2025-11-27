from sacred import Experiment
import os
os.environ["WANDB_MODE"] = "offline"
ex = Experiment("CM-MMoE", save_git_info=False)

@ex.config
def config():
    # save
    subDir = "512_128cam32d4_iiaw9_alpha0.4"
    use_wandb = True
    wandbName = subDir
    wandbKey = "ca271f6739d12f8984404246e8583b3cddd0b465"
    project = "CM-MMoE-cam-iia"
    job_type = "train"
    # Loss function weighting parameters
    alpha = 0.4  # RMSE weight in the loss; acc_loss weight is (1 - alpha)
    fusion_mode = "cam"  # Fusion mode options: "moe", "crossattention", "fusion", "cam"
    CAM_C =32 # Number of channels C in CAM (for feature reshaping)
    CAM_IMAGE_DIM = 512  # Image feature dimension for CAM
    CAM_TEXT_DIM = 128  # Text feature dimension for CAM
    use_wavelet_iia = True  # Enable wavelet-based IIA module
    wavelet_type = 'db4'  # Wavelet type options: 'db4', 'haar', 'db8', 'bior2.2'
    iia_kernel_size = 9  # WIIA kernel_size
    # Ablation switch and types for wavelet_iia
    wavelet_iia_ablation_enable = False
    wavelet_iia_ablation_type = 'mask_LL'  
    # Options: 'none', 'mask_LL', 'mask_LH', 'mask_HL', 'mask_HH',
    # 'disable_attention_LH', 'disable_attention_HL'

    # t-SNE visualization parameters
    save_features = False  # Whether to save features for t-SNE visualization
    

    MoE = False
    answer_number = 50
    question_classes = 14
    # if MoE:
    EXPERTS = 4
    TOP = 2

    normalize = False
    opts = True
    one_step = True
    num_epochs = 30
    thread_epoch = 20

    learning_rate = 5e-5  # 5e-5

    saveDir = "./outputs/"
    saveDir = os.path.join(saveDir, subDir + '/')
    new_data_path = './datasets/CM_dataset/'
    source_image_size = 512

    image_resize = 224
    imageSize = 224  # Add imageSize parameter for compatibility
    
    FUSION_IN = 768
    FUSION_HIDDEN = 512
    DROPOUT = 0.3

    add_mask = True
    pin_memory = True
    persistent_workers = True

    num_workers = 4

    real_batch_size = 32
    batch_size = 32  # batch_size * steps == real_batch_size
    steps = int(real_batch_size / batch_size)
    weight_decay = 0
    opt = "Adam"
    scheduler = True
    CosineAnnealingLR = True
    warmUp = False
    L1Reg = False
    resample = False
    trainText = True
    trainImg = True
    finetuneMask = True

    if scheduler:
        end_learning_rate = 1e-6

    json_path = os.path.join(new_data_path, 'JsonFiles')
    DataConfig = {
        "images_path": os.path.join(new_data_path, "image"),
        "sourceMask_path": os.path.join(new_data_path, "source"),
        "targetMask_path": os.path.join(new_data_path, "target"),
        "backgroundMask_path": os.path.join(new_data_path, "background"),
        "seg_path": os.path.join(new_data_path, "segmentation"),
        "answersJson": os.path.join(json_path, "Answers.json"),
        "allQuestionsJSON": os.path.join(json_path, "All_Questions.json"),
        "train": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Train_Questions.json"),
        },
        "val": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Val_Questions.json"),
        },
        "test": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Test_Questions.json"),
        },
    }
    LEN_QUESTION = 40
    clipList = [
        "clip",
        "rsicd",
        "clip_b_32_224",
        "clip_b_16_224",
        "clip_l_14_224",
        "clip_l_14_336",
    ]
    vitList = ["vit-b", "vit-s", "vit-t"]
    maskHead = "unet"
    if maskHead == "unet":
        maskModelPath = (
            "models/imageModels/milesial_UNet/unet_carvana_scale1.0_epoch2.pth"
        )

    imageHead = "clip_b_32_224"
    if imageHead == "clip_b_32_224":
        imageModelPath = "models/clipModels/openai_clip_b_32"
        VISUAL_OUT = 768
        image_resize = 224
    elif imageHead == "siglip_512":
        imageModelPath = "models/clipModels/siglip_512"
        image_resize = 512
    else:
        image_resize = 256
    textHead = "clip_b_32_224"
    if textHead == "clip_b_32_224":
        textModelPath = "models/clipModels/openai_clip_b_32"
        QUESTION_OUT = 512
    elif textHead == "siglip_512":
        textModelPath = "models/clipModels/siglip_512"
        QUESTION_OUT = 768
    elif textHead == "skipthoughts":
        textModelPath = "models/textModels/skip-thoughts"
        QUESTION_OUT = 2400
    attnConfig = {
        "embed_size": FUSION_IN,
        "heads": 6,
        "mlp_input": 768,
        "mlp_ratio": 4,
        "mlp_output": 768,
        "attn_dropout": 0.1,
    }