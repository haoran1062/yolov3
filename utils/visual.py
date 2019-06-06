# encoding:utf-8
import visdom, time, numpy as np, torch, random, cv2, json
from PIL import Image

from torchvision import transforms


class Visual(object):

   def __init__(self, env='default', log_to_file=None, **kwargs):
       self.vis = visdom.Visdom(env=env, log_to_filename=log_to_file, **kwargs)
    #    if log_to_file:
    #        self.create_log_at(log_to_file, env)
    #    else:
       self.index = {} 
       self.log_text = ''

   def reinit(self, env='default', **kwargs):
       self.vis = visdom.Visdom(env=env, **kwargs)
       return self
    
   def create_log_at(self, file_path, current_env, new_env=None):
      new_env = current_env if new_env is None else new_env
      vis = visdom.Visdom(env=current_env)

      data = json.loads(vis.get_window_data())
      self.index = data
      if len(data) == 0:
         print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
         return

   def plot_many(self, d):
       
       for k, v in d.iteritems():
           self.plot(k, v)

   def img_many(self, d):
       for k, v in d.iteritems():
           self.img(k, v)

   def plot(self, name, y, **kwargs):
       
       x = self.index.get(name, 0)
    #    print(x)
    #    print(x.keys())
       self.vis.line(Y=np.array([y]), X=np.array([x]),
                     win=name,
                     opts=dict(title=name),
                     update='append' if x > 0 else None,
                     **kwargs)
       self.index[name] = x + 1

   def img(self, name, img_, **kwargs):
       def cv2PIL(img):
           cv2.imwrite('temp.jpg', img)
           return Image.open('temp.jpg')
        #    print('convert')
        #    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #    return Image.fromarray(img)

       if isinstance(img_, np.ndarray):
           img_ = cv2PIL(img_)
           img_ = transforms.ToTensor()(img_)
       self.vis.images(img_,
                      win=name,
                      opts=dict(title=name),
                      **kwargs)

   def log(self, info, win='log_text'):
       
       self.log_text += ('[{time}] {info} <br>'.format(
                           time=time.strftime('%m%d_%H%M%S'),
                           info=info))
       self.vis.text(self.log_text, win)

   def __getattr__(self, name):
       
       return getattr(self.vis, name)