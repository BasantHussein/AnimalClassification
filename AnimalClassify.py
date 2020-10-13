####################Cleaning#############################

dir = 'animals'
ls=[]
labels = []
subdirs = [x[1] for x in os.walk(dir)]
k=0
for subdir in subdirs[0]:
    k+=1
    print(str(subdir))
    for filename in glob.glob('animals/'+str(subdir) +'/*.jpg'):
        image = Image.open(filename)
        size = (50,50)#(28, 28)
        image = ImageOps.fit(image, size, Image.ANTIALIAS).convert('LA')
        pix = np.array(image)
        pix = pix.flatten()
        pix = pix.tolist()
        ls.append(pix)
        labels.append(k)
		

# this function to read all dirs of dataset classes
def rec_walk(dir):
    contents = os.listdir(dir) # read the contents of dir
    lst = []
    for item in contents:      # loop over those contents
        lst.append(item) 
    return lst

def resizeImage(path, width, height):
    basewidth = width
    img = Image.open(path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,height), Image.ANTIALIAS)
    img.save(path) 


allClasses =  rec_walk("image/") #read all classes folders
classBear = rec_walk("image/BearHead") # read first class file "dirs Only"

#fitAllImage()
def renameDataSetFiles():
    for cls in range(len(allClasses)-1):
        i = 1;
        currentClassImages = rec_walk("image/"+allClasses[cls])
        for image in range(len(currentClassImages)):
            imagePath = "image/"+allClasses[cls]+"/" + currentClassImages[image]
            newDir = "image/"+allClasses[cls]+"/" +allClasses[cls] +'__'+ str(i)+'.jpg';
            i+=1;
            os.rename(imagePath, newDir)
            print(newDir)

        
def fitAllImage():
    for cls in range(len(allClasses)):
        currentClassImages = rec_walk("image/"+allClasses[cls])
        for image in range(len(currentClassImages)):
            imagePath = "image/"+allClasses[cls]+"/" + currentClassImages[image]
            jpgfile = Image.open(imagePath)
            if (jpgfile.size[0]!=150 or jpgfile.size[1]!=150 ):
                print(imagePath)
                resizeImage(imagePath,150,150)
				
allDatasetPhotos = [] #all dataset images as Dirs
def loadFilePaths():
    for cls in range(len(allClasses)):
        i = 1;
        currentClassImages = rec_walk("image/"+allClasses[cls])
        for image in range(len(currentClassImages)):
            allDatasetPhotos.append("image/"+allClasses[cls]+"/" + currentClassImages[image])
            
loadFilePaths()



alls = []
flattenAll = []
def readDatasetInNormalArray():
    for i in range(len(allDatasetPhotos)):
        I = np.asarray(Image.open(allDatasetPhotos[i]))
        #print()
        
        alls.append(I)
        flattenAll.append(I.flatten())
        #plt.imshow(rgb2gray(I),cmap=plt.get_cmap('gray'), vmin=1, vmax=100)
        ظplt.imshow(I)
        #I.flatten
        #img = Image.open('image.png').convert('LA')
        #print(len(I.flatten()))
    
readDatasetInNormalArray()

dataset = np.ndarray(shape=(len(flattenAll[0]),len(flattenAll[0])),dtype=np.float32)
dataset[0] = flattenAll[0]
print(dataset)
#print(len(alls[0][0][0]))
def setDatasetInNumpyArray():
    
    for i in range(len(alls)):
        dataset[i] = alls[i]
		
		
#######################################CNN#####################################


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

x = tf.placeholder(tf.float32,shape=[None,10800])
y_true = tf.placeholder(tf.float32,shape=[None,20])

x_image = tf.reshape(x,[-1,60,60,1])

convo_1 = convolutional_layer(x_image,shape=[6,6,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,15*15*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,20)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x , batch_y = train_X[i:i+50],train_y[i:i+50]
        #batch_x , batch_y= my_func(batch_x),my_func(batch_y)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:test_X,y_true:test_y,hold_prob:1.0}))
            print('\n')
##########################################################KNN########################################3
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Test set prediction:\n {}".format(y_pred))
########################################################SVM###########################################
clf = svm.SVC(C=43,gamma=0.001)
clf.fit(X_train,y_train)
print(str(clf.score(X_test,y_test)*100))