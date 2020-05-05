from PIL import Image, ImageDraw
import sys,glob,os

def main():
    dir_in = sys.argv[1] 
    dir_out = "../gif/"
    print("Saving from folder: ", dir_in)
    list_im = sorted(glob.glob(dir_in + "/*"))
    print("Number of images found = ", len(list_im))
    fol_name = os.path.basename(dir_in)
    path_out = "{}/{}.gif".format(dir_out, fol_name)
    images = [Image.open(f) for f in list_im]
    images[0].save(path_out, save_all=True, append_images=images[1:], duration=500,loop=0)
    print("Saved at ", path_out) 
    

if __name__ == "__main__":
    main()    
