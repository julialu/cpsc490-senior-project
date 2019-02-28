

# script for preprocessing images to greyscale and 48x48
def setupData(foldername,numRows=72,numCols=128):
	#Given a 'foldername' with .jpg frames, set turn into grayscaled numpy arrays.
	#numRows and numCols gives number of rows/columns we rescale each frame to.
	#If downloaded at 144p, default is numRows=144, numCols=256
	
	filenames = [] #Hold the names of the files so we can pull them back up.
	temp = [] #temp array to hold converted frames.
	size = numCols,numRows #Need to define this variable for resizing later.
						   #default 144p is 256x144 (numCols = 256, numRows = 144, convention is width x height).
	
	for filename in os.listdir(foldername):
		path = foldername+filename
		im = Image.open(path).convert('L') #Load image and convert to grayscale.
		#im.show()
		
		im_resized = im.resize(size, Image.ANTIALIAS) #Re-size image

		arr = np.array(im_resized.getdata()) #Convert image into numpy array.
		arr = arr/255.0 #Scale all pixel values to between 0 and 1.
		
		temp.append(arr)
		filenames.append(filename)
	
	#Convert to numpy array and return.
	return np.array(temp),filenames

if __name__ == "__main__":


    #Path where the images are
    path = "/Users/julialu/cpsc490-senior-project/data/Training/Faces"

    #Path where the faces will be saved
   # savePath = "/Users/julialu/cpsc490-senior-project/data/Training/Faces"

    setupData(path)