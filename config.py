import os

class UNetConfig:

    def __init__(self,
                 epochs = 10,  
                 batch_size = 8,    
                 validation = 10.0,   
                 out_threshold = 1,

                 optimizer='Adam',
                 lr = 0.0003,     
                 lr_decay_milestones = [],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels = 3, 
                 n_classes = 3,  
                 scale = 1,    

                 load = False,   
                 save_cp = True,

                
                 model='mynet',
                 bilinear =True,
                 deepsupervision =False ,  
                 ):
        super(UNetConfig, self).__init__()

        self.images_dir = 'train image path'
        self.masks_dir = 'train label path'
        self.val_img_dir = 'test image path'
        self.val_lab_dir = 'test label path'
        self.checkpoints_dir = 'checkpoint path'


        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)

