
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.mask import *
import pickle
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# In[28]:


dataDir='../coco/'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


# In[ ]:


# initialize COCO api for instance annotations
coco=COCO(annFile)




# catIds = coco.getCatIds(catNms=['person']);
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))


# imgIds = coco.getImgIds(imgIds = [21685])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread(img['coco_url'])
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns1 = coco.loadAnns(annIds)
# mask1=coco.annToMask(anns1[1])

#mask1=coco.annToMask(anns1[0])
#rle1= encode(mask1)
#bb1 = toBbox(rle1)


catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds.sort()
# imgIds=imgIds[:100]
#bestmatch = [[[] for i in range(len(imgIds))] for j in range(len(imgIds))]
f= open('bestmatch.txt','w+')
for i in range(len(imgIds)):
    ma=-1
    imgIds1 = coco.getImgIds(imgIds = [imgIds[i]])
    img1 = coco.loadImgs(imgIds1[np.random.randint(0,len(imgIds1))])[0]
    annIds1 = coco.getAnnIds(imgIds=img1['id'], catIds=catIds, iscrowd=None)
    anns1 = coco.loadAnns(annIds1)
    bboxmany1=[]
    for k in range(len(anns1)):
        bboxmany1.append(toBbox(encode(coco.annToMask(anns1[k]))))

    #bb1 = toBbox(encode(coco.annToMask(anns1[0])))
    for j in range(len(imgIds)):
        if i==j:
            continue
        imgIds2 = coco.getImgIds(imgIds = [imgIds[j]])
        img2 = coco.loadImgs(imgIds2[np.random.randint(0,len(imgIds2))])[0]
        annIds2 = coco.getAnnIds(imgIds=img2['id'], catIds=catIds, iscrowd=None)
        anns2 = coco.loadAnns(annIds2)
        bboxmany2=[]
        for k in range(len(anns2)):
            bboxmany2.append(toBbox(encode(coco.annToMask(anns2[k]))))

        tmp1=[]
        for k in range(len(anns1)):
            tmp=[]
            for l in range(len(anns2)):
                tmp.append(iou([bboxmany1[k]],[bboxmany2[l]],[1])[0][0])
            tmp1.append(tmp)
            # bestmatch[i][j].append(tmp)
        f.write(str(tmp1))
        f.write('\n')
        #print bestmatch[i][j]
        #bb2 = toBbox(encode(coco.annToMask(anns2[0])))
        #bestmatch[i][j]= iou([bb1],[bb2],[1])
        #print iou([bb1],[bb2],[1])[0][0]
f.close()
# pickle.dump(bestmatch,open('bestmatch.txt','w+'))

