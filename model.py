from glob import glob
import matplotlib.pyplot as plot
#from IPython.display import Image
from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16 #Import all the necessary modules
#import os



#with pneumonia train image
# img_trainp=[]
# img_trainn=[]
# for img in glob.glob("./chest-xray/train/PNEUMONIA/*"):
#     jpg = cv2.imread(img)
#     image_bw = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)

#     clahe = cv2.createCLAHE(clipLimit=2.5)
#     final_img = clahe.apply(image_bw) + 12
#     img_trainp.append(final_img)
#     #print(img_trainp)

# for img1 in glob.glob("./chest-xray/train/NORMAL/*"):
#     jpg1 = cv2.imread(img1)
#     image_bw1 = cv2.cvtColor(jpg1, cv2.COLOR_BGR2GRAY)

#     clahe1 = cv2.createCLAHE(clipLimit=2.5)
#     final_img1 = clahe1.apply(image_bw1) + 12
#     img_trainn.append(final_img1)
#     #print(img_trainp)
    
# #Test image Processing
# img_testp=[]
# img_testn=[]
# for img in glob.glob("./chest-xray/test/PNEUMONIA/*"):
#     jpg = cv2.imread(img)
#     image_bw = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)

#     clahe = cv2.createCLAHE(clipLimit=2.5)
#     final_img = clahe.apply(image_bw) + 12
#     img_testp.append(final_img)
#     #print(img_trainp)

# for img1 in glob.glob("./chest-xray/test/NORMAL/*"):
#     jpg1 = cv2.imread(img1)
#     image_bw1 = cv2.cvtColor(jpg1, cv2.COLOR_BGR2GRAY)

#     clahe1 = cv2.createCLAHE(clipLimit=2.5)
#     final_img1 = clahe1.apply(image_bw1) + 12
#     img_testn.append(final_img1)
#     #print(img_trainp)


# #Validation Image Processing
# img_valp=[]
# img_valn=[]
# for img in glob.glob("./chest-xray/val/PNEUMONIA/*"):
#     jpg = cv2.imread(img)
#     image_bw = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)

#     clahe = cv2.createCLAHE(clipLimit=2.5)
#     final_img = clahe.apply(image_bw) + 12
#     img_valp.append(final_img)
#     #print(img_trainp)

# for img1 in glob.glob("./chest-xray/val/NORMAL/*"):
#     jpg1 = cv2.imread(img1)
#     image_bw1 = cv2.cvtColor(jpg1, cv2.COLOR_BGR2GRAY)

#     clahe1 = cv2.createCLAHE(clipLimit=2.5)
#     final_img1 = clahe1.apply(image_bw1) + 12
#     img_valn.append(final_img1)


# #Cropping image

# img_valcp=[]
# img_valcn=[]
# for img in img_valp:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_valcp.append(crop_img)
# for img in img_valn:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_valcn.append(crop_img)
# #print(len(img_valcn))
# #print(len(img_valcp))

# img_testcn=[]
# img_testcp=[]
# for img in img_testp:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_testcp.append(crop_img)
# for img in img_testn:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_testcn.append(crop_img)
# #print(len(img_testcn))
# #print(len(img_testcp))
# test_all = img_testcp+img_testcn

# img_traincn=[]
# img_traincp=[]
# for img in img_trainp:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_traincp.append(crop_img)
# for img in img_trainn:
#     crop_img = img[:,int((img.shape[1])*.06):(img.shape[1])-int((img.shape[1])*.06)]
#     img_traincn.append(crop_img)
# #print(len(img_traincn))
# #print(len(img_traincp))
# train_all = img_traincn+img_traincp
# i=1
# for img in train_all:
#     filename = "train_crop_img_"+str(i)+".jpeg"
#     path = r"C:/Users\abakf\Desktop\pneumonia-model\chest-xray\update_train"
#     cv2.imwrite(os.path.join(path , filename), img)
#     i=i+1
IMAGESHAPE = [224, 224, 3] 
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)
training_data = 'chest_xray/update_train_c'
testing_data = 'chest_xray/test' 
for each_layer in vgg_model.layers:
	each_layer.trainable = False 
classes = glob('chest_xray/update_train_c/*') 
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction) 
final_model.summary() 
final_model.compile( 
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range = 30,  
        zoom_range = 0.2, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip = True,  
        vertical_flip=False)
testing_datagen = ImageDataGenerator(rescale =1. / 255)
training_set = train_datagen.flow_from_directory('chest_xray/update_train_c', 
						target_size = (224, 224),
						 batch_size = 25,
						class_mode = 'categorical')
test_set = testing_datagen.flow_from_directory('chest_xray/test',
						target_size = (224, 224),
						batch_size = 25,
						class_mode = 'categorical')
fitted_model = final_model.fit(
training_set,
validation_data=test_set,
epochs=10,
steps_per_epoch=len(training_set),
validation_steps=len(test_set)
)
plot.plot(fitted_model.history['loss'], label='training loss') 
plot.plot(fitted_model.history['val_loss'], label='validation loss')
plot.legend()
plot.show()
plot.savefig('LossVal_loss')
plot.plot(fitted_model.history['accuracy'], label='training accuracy')
plot.plot(fitted_model.history['val_accuracy'], label='validation accuracy')
plot.legend()
plot.show()
plot.savefig('AccVal_acc')
final_model.save('our_model.h5') 