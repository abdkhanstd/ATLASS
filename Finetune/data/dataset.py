# data/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image

class FineTuneDataset(Dataset):
    """
    Custom dataset for fine-tuning models on various datasets mapped to JSIEC classes.
    """
    def __init__(self, dataset_path, split='train', transform=None):
        self.root_dir = os.path.join(dataset_path, split)
        self.transform = transform
        self.dataset_name = os.path.basename(dataset_path)
        
        # Define class mappings for different datasets
        self.class_mappings = {
            'JSIEC': {cls: cls for cls in sorted(os.listdir(os.path.join('datasets', 'JSIEC', 'train')))},
            'APTOS2019': {
                'anodr': '0.0.Normal',
                'bmilddr': '0.3.DR1',
                'cmoderatedr': '1.0.DR2',
                'dseveredr': '1.1.DR3',
                'eproliferativedr': '29.1.Blur fundus with suspected PDR'
            },
            'MESSIDOR2': {
                'anodr': '0.0.Normal',
                'bmilddr': '0.3.DR1',
                'cmoderatedr': '1.0.DR2',
                'dseveredr': '1.1.DR3',
                'eproliferativedr': '29.1.Blur fundus with suspected PDR'
            },
            'IDRiD': {
                'anoDR': '0.0.Normal',
                'bmildDR': '0.3.DR1',
                'cmoderateDR': '1.0.DR2',
                'dsevereDR': '1.1.DR3',
                'eproDR': '29.1.Blur fundus with suspected PDR'
            },
            'PAPILA': {
                'anormal': '0.0.Normal',
                'bsuspectglaucoma': '10.0.Possible glaucoma',
                'cglaucoma': '10.1.Optic atrophy'
            },
            'Glaucoma_fundus': {
                'anormal_control': '0.0.Normal',
                'bearly_glaucoma': '10.0.Possible glaucoma',
                'cadvanced_glaucoma': '10.1.Optic atrophy'
            },
            'OCTID': {
                'ANormal': '0.0.Normal',
                'CSR': '5.0.CSCR',
                'Diabetic_retinopathy': '1.0.DR2',
                'Macular_Hole': '8.MH',
                'ARMD': '6.Maculopathy'
            },
            'Retina': {
                'anormal': '0.0.Normal',
                'cglaucoma': '10.1.Optic atrophy',
                'bcataract': '29.0.Blur fundus without PDR',
                'ddretina_disease': '6.Maculopathy'
            },
            'RFMiD': {
                'WNL': '0.0.Normal',
                'DR': '1.0.DR2',
                'DN': '1.0.DR2',
                'HTN': '11.Severe hypertensive retinopathy',
                'BRVO': '2.0.BRVO',
                'CRVO': '2.1.CRVO',
                'CRAO': '3.RAO',
                'ARMD': '6.Maculopathy',
                'CSR': '5.0.CSCR',
                'CSC': '5.0.CSCR',
                'CME': '6.Maculopathy',
                'ME': '6.Maculopathy',
                'ERM': '7.ERM',
                'MH': '8.MH',
                'AION': '10.1.Optic atrophy',
                'ODE': '12.Disc swelling and elevation',
                'ODC': '14.Congenital disc abnormality',
                'ODP': '14.Congenital disc abnormality',
                'ON': '10.1.Optic atrophy',
                'IIH': '12.Disc swelling and elevation',
                'RD': '4.Rhegmatogenous RD',
                'RP': '15.0.Retinitis pigmentosa',
                'CNV': '19.Fundus neoplasm',
                'CWS': '22.Cotton-wool spots',
                'AH': '28.Silicon oil in eye',
                'CF': '26.Fibrosis',
                'CL': '24.Chorioretinal atrophy-coloboma',
                'GRT': '16.Peripheral retinal degeneration and break',
                'HPED': '20.Massive hard exudates',
                'HR': '25.Preretinal hemorrhage',
                'LS': '27.Laser Spots',
                'MCA': '24.Chorioretinal atrophy-coloboma',
                'MHL': '25.Preretinal hemorrhage',
                'MS': '21.Yellow-white spots-flecks',
                'MYA': '9.Pathological myopia',
                'OPDM': '10.0.Possible glaucoma',
                'PRH': '25.Preretinal hemorrhage',
                'RHL': '25.Preretinal hemorrhage',
                'RTR': '16.Peripheral retinal degeneration and break',
                'RPEC': '24.Chorioretinal atrophy-coloboma',
                'RS': '26.Fibrosis',
                'RT': '16.Peripheral retinal degeneration and break',
                'SOFE': '20.Massive hard exudates',
                'ST': '23.Vessel tortuosity',
                'TD': '13.Dragged Disc',
                'TSLN': '21.Yellow-white spots-flecks',
                'TV': '23.Vessel tortuosity',
                'VS': '23.Vessel tortuosity',
            }
        }
        
        # Get the original dataset classes in sorted order
        jsiec_path = os.path.join('datasets', 'JSIEC', 'train')
        self.classes = sorted(os.listdir(jsiec_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        if self.dataset_name in self.class_mappings:
            mapping = self.class_mappings[self.dataset_name]
            for class_name in os.listdir(self.root_dir):
                if class_name in mapping:
                    mapped_class = mapping[class_name]
                    class_dir = os.path.join(self.root_dir, class_name)
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(valid_extensions):
                            img_path = os.path.join(class_dir, img_name)
                            if os.path.isfile(img_path):
                                self.samples.append((img_path, self.class_to_idx[mapped_class]))
        else:
            print(f"Warning: Dataset {self.dataset_name} not in mapping, skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding label at the specified index.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
