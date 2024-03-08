import io
import sqlite3
from torch.utils import data as data
from PIL import Image
from torchvision.transforms import functional as TF

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SQLDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        
        db_file = opt['db_file']
        db_table = 'images'
        lr_col = 'lr_img'
        hr_col = 'hr_img'
        
        hflip = True
        rotate = True

        self.total_images = self.get_num_rows()

    def get_num_rows(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT MAX(ROWID) FROM {self.db_table}')
            db_rows = cursor.fetchone()[0]

        return db_rows
    
    def __len__(self):
        return self.total_images
    
    def __getitem__(self, item):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT {self.lr_col}, {self.hr_col} FROM {self.db_table} WHERE ROWID={item+1}')
            lr, hr = cursor.fetchone()

        gt_path = f"{item}.png"
        lq_path = f"{item}_x2.png"
        img_gt = Image.open(io.BytesIO(lr)).convert('RGB')
        img_lq = Image.open(io.BytesIO(hr)).convert('RGB')

        img_gt, img_lq = TF.to_tensor(lr), TF.to_tensor(hr)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}