import os
if __name__ == "__main__":
    directory = "./datasets/remove"
    removeBefore = 10
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            timestamp = file
            foldername = os.path.splitext(timestamp)[0]
            with open(directory+"/"+timestamp) as fp:
                for line in fp:
                    gameoverFrame = line.rstrip()
                    nr = int(gameoverFrame)
                    for framenr in range(max(0,nr-10),nr+1):                
                        framenr = str(framenr) + ".png"
                        jumpfullpath = "./datasets/"+foldername+"/jump/"+foldername +"_" +framenr
                        nojumpfullpath = "./datasets/"+foldername+"/no_jump/"+foldername + "_" + framenr
                        if os.path.isfile(jumpfullpath):
                            os.remove(jumpfullpath)
                            print(jumpfullpath+" deleted")
                        elif os.path.isfile(nojumpfullpath):
                            os.remove(nojumpfullpath)
                            print(nojumpfullpath+" deleted")
                        else:
                            print("file does not exist")
