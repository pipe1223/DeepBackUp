import os
import glob

MAIN_PATH = "D:/Work/data/2dslice/"
PATH1 = MAIN_PATH + "train_224/"

def to_txt(IMGPATH,file_name,val_name):
    f = open(file_name,"w")
    f_val = open(val_name,"w")
    ls = [os.path.basename(x) for x in glob.glob(IMGPATH + "*")]
    for i in range(len(ls)):
        imgname = ls[i]
        if i < len(ls)*0.8:
	    f.write(imgname+"\n")
	else:
	    f_val.write(imgname+"\n")
    f_val.close()
    f.close()

if __name__=="__main__":
    to_txt(PATH1,"med_train.txt","med_val.txt")
