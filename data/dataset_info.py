# ------------------------------------------------------
# OpenI dataset, in total 15 classes
# ------------------------------------------------------
OPENI_LABEL_LIST = [
    '',  # other
    'Enlarged Cardiomediastinum',
    'Atelectasis',
    'Pleural Other',
    'Pleural Effusion',
    'No Finding',
    'Cardiomegaly',
    'Lung Opacity',
    'Pneumothorax', # does not have test data
    'Edema',
    'Lung Lesion',
    'Consolidation',
    'Support Devices',
    'Fracture',
    'Pneumonia'
]

MNIST_LABEL_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

CIFAR10_LABEL_LIST = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

VOC2012_LABEL_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
                         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

ISIC2018_LABEL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

XRAY14_LABEL_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",  "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
                        "Nodule", "Pneumonia", "Pneumothorax", "Pleural Thickening", "No Finding"]

# chestmnist, multi-label, 14 classes
CHESTMNIST_LABEL_LIST = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
                         'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

CHESTMNIST_LABEL_LIST_W_BACKGROUND = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
                         'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'Healthy']

# dermamnist, multi-class, 7 classes, Skin Disease Classification
SKIN_LABEL_LIST = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma',
                    'melanocytic nevi', 'melanoma', 'vascular lesions']

# pathmnist, multi-class, 9 classes, Colon Pathology Tissue Classification
CANCER_LABEL_LIST = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle', 'normal colon mucosa',
                      'cancer-associated stroma', 'colorectal adenocarcinoma epithelium']


# ------------------------------------------------------
# ------------------------------------------------------
OPENI_LABEL_INDEX_LIST = [i for i in range(len(OPENI_LABEL_LIST))]

OPENI_index_label_map = {index: label for index, label in enumerate(OPENI_LABEL_LIST)}

OPENI_label_to_index_map = {label: index for index, label in OPENI_index_label_map.items()}

# Step 0: No Finding, Enlarged Cardiomediastinum, Cardiomegaly
# Step 1: No Finding, Pleural Other, Pleural Effusion, Pneumothorax
# Step 2: No Finding, Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Lung Lesion
# Step 3: No Finding, Fracture
# Step 4: No Finding, Support Devices
# Step 5: No Finding, other

INC_TASK_OPENI_LABEL_LIST = [
    ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly'],
    ['No Finding', 'Pleural Other', 'Pleural Effusion', 'Pneumothorax'],
    ['No Finding', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Lung Lesion'],
    ['No Finding', 'Fracture'],
    ['No Finding', 'Support Devices'],
    ['No Finding', '']
    ]

INC_TASK_OPENI_INDEX_LIST = [
    [OPENI_label_to_index_map.get(label, -1) for label in labels] for labels in INC_TASK_OPENI_LABEL_LIST
]

# Accumulated class label since CL learning to evaluate overall performance (old & new classes)
INC_TASK_OPENI_INDEX_TEST_LIST = INC_TASK_OPENI_INDEX_LIST.copy()
for i in range(len(INC_TASK_OPENI_INDEX_TEST_LIST)):
    # Accumulate previous lists
    if i > 0:
        INC_TASK_OPENI_INDEX_TEST_LIST[i] = INC_TASK_OPENI_INDEX_TEST_LIST[i - 1] + INC_TASK_OPENI_INDEX_TEST_LIST[i]
    # Remove duplicates while preserving order
    seen = set()
    INC_TASK_OPENI_INDEX_TEST_LIST[i] = [x for x in INC_TASK_OPENI_INDEX_TEST_LIST[i] if not (x in seen or seen.add(x))]

# print("OPENI_LABEL_LIST: {0}".format(OPENI_LABEL_LIST))
# print("OPENI_LABEL_INDEX_LIST: {0}".format(OPENI_LABEL_INDEX_LIST))
# print("OPENI_index_label_map: {0}".format(OPENI_index_label_map))
# print("OPENI_label_to_index_map: {0}".format(OPENI_label_to_index_map))
# print("INC_TASK_OPENI_LABEL_LIST: {0}".format(INC_TASK_OPENI_LABEL_LIST))
# print("INC_TASK_OPENI_INDEX_LIST: {0}".format(INC_TASK_OPENI_INDEX_LIST))
# print("INC_TASK_OPENI_INDEX_TEST_LIST: {0}".format(INC_TASK_OPENI_INDEX_TEST_LIST))

# ------------------------------------------------------
# MIMIC dataset, in total 15 classes, same as OPENI
# ------------------------------------------------------