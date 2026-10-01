import uvicorn
import io
from fastapi import FastAPI,HTTPException,File,UploadFile
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image

classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy',
 'bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud',
 'cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox',
 'girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',
 'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear',
 'pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
 'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower',
 'sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe',
 'whale','willow_tree','wolf','woman','worm']



class PredFoto(nn.Module):
  def __init__(self):
    super().__init__()
    self.first =nn. Sequential(
      nn.Conv2d(3, 32,kernel_size=3, padding=1),
      nn. ReLU(),
      nn. MaxPool2d (2),

      nn.Conv2d(32, 64,kernel_size=3, padding=1),
      nn. ReLU(),
      nn. MaxPool2d (2),


      nn.Conv2d(64, 128,kernel_size=3, padding=1),
      nn. ReLU(),
      nn. MaxPool2d (2),
    )

    self. second =nn. Sequential (
     nn. Flatten(),
     nn. Linear (128 * 4 * 4, 256),
     nn. ReLU(),
     nn. Linear (256, 100)
   )
  def forward (self, image):
    image = self.first (image)
    image = self.second (image)
    return image








foto_app = FastAPI()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PredFoto()
model.load_state_dict(torch.load('model7_path',map_location= device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


@foto_app.post ('/predict/')
async def check_image (file: UploadFile=File(...)):
     try:
         image_data = await file.read()
         if not image_data:
             raise HTTPException (status_code=400, detail= 'Файл кошулган жок')
         img = Image.open (io.BytesIO (image_data))
         img_tenser = transform(img).unsqueeze(0).to(device)

         with torch.no_grad():
             y_pred = model(img_tenser)
             pred = y_pred.argmax(dim= 1).item()
         return {"Answer": classes[pred]}

     except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







if __name__=="__main__":
    uvicorn.run(foto_app,host='127.0.0.1',port=8000)