 
# Importing the required libraries.
library(tensorflow)
library(keras)

# Loading the images into the workspace and converting the images to a vector of pixels.
image_names = image_names <- c('img1.jpg','img2.jpg','img3.jpg','img4.jpg','img5.jpg','img6.jpg','img7.jpg','img8.jpg','img9.jpg','img10.jpg','img11.jpg')
num_images = length (image_names)
x <- array(dim = c(num_images,224, 224, 3))
for (i in 1:num_images) {
    img_path = paste('/content', image_names[i], sep = "/")
    img = image_load(img_path, target_size = c(224, 224))
    x[i,,, ] = image_to_array(img)
}

# Inputting the array to the imagenet model.
x = imagenet_preprocess_input(x)

# Initializing the model.
model = application_resnet50()
summary(model)

# Model: "resnet50"
# ________________________________________________________________________________
#  Layer (type)         Output Shape   Param #  Connected to           Trainable  
# ================================================================================
#  input_3 (InputLayer)  [(None, 224,   0       []                     Y          
#                       224, 3)]                                                  
#  conv1_pad (ZeroPaddi  (None, 230, 2  0       ['input_3[0][0]']      Y          
#  ng2D)                30, 3)                                                    
#  conv1_conv (Conv2D)  (None, 112, 1  9472     ['conv1_pad[0][0]']    Y          
#                       12, 64)                                                   
#  conv1_bn (BatchNorma  (None, 112, 1  256     ['conv1_conv[0][0]']   Y          
#  lization)            12, 64)                                                   
#  conv1_relu (Activati  (None, 112, 1  0       ['conv1_bn[0][0]']     Y          
#  on)                  12, 64)                                                   
#  pool1_pad (ZeroPaddi  (None, 114, 1  0       ['conv1_relu[0][0]']   Y          
#  ng2D)                14, 64)                                                   
#  pool1_pool (MaxPooli  (None, 56, 56  0       ['pool1_pad[0][0]']    Y          
#  ng2D)                , 64)                                                     
#  conv2_block1_1_conv   (None, 56, 56  4160    ['pool1_pool[0][0]']   Y          
#  (Conv2D)             , 64)                                                     
#  conv2_block1_1_bn (B  (None, 56, 56  256     ['conv2_block1_1_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block1_1_relu   (None, 56, 56  0       ['conv2_block1_1_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block1_2_conv   (None, 56, 56  36928   ['conv2_block1_1_relu  Y          
#  (Conv2D)             , 64)                   [0][0]']                          
#  conv2_block1_2_bn (B  (None, 56, 56  256     ['conv2_block1_2_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block1_2_relu   (None, 56, 56  0       ['conv2_block1_2_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block1_0_conv   (None, 56, 56  16640   ['pool1_pool[0][0]']   Y          
#  (Conv2D)             , 256)                                                    
#  conv2_block1_3_conv   (None, 56, 56  16640   ['conv2_block1_2_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv2_block1_0_bn (B  (None, 56, 56  1024    ['conv2_block1_0_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv2_block1_3_bn (B  (None, 56, 56  1024    ['conv2_block1_3_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv2_block1_add (Ad  (None, 56, 56  0       ['conv2_block1_0_bn[0  Y          
#  d)                   , 256)                  ][0]',                            
#                                                'conv2_block1_3_bn[0             
#                                               ][0]']                            
#  conv2_block1_out (Ac  (None, 56, 56  0       ['conv2_block1_add[0]  Y          
#  tivation)            , 256)                  [0]']                             
#  conv2_block2_1_conv   (None, 56, 56  16448   ['conv2_block1_out[0]  Y          
#  (Conv2D)             , 64)                   [0]']                             
#  conv2_block2_1_bn (B  (None, 56, 56  256     ['conv2_block2_1_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block2_1_relu   (None, 56, 56  0       ['conv2_block2_1_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block2_2_conv   (None, 56, 56  36928   ['conv2_block2_1_relu  Y          
#  (Conv2D)             , 64)                   [0][0]']                          
#  conv2_block2_2_bn (B  (None, 56, 56  256     ['conv2_block2_2_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block2_2_relu   (None, 56, 56  0       ['conv2_block2_2_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block2_3_conv   (None, 56, 56  16640   ['conv2_block2_2_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv2_block2_3_bn (B  (None, 56, 56  1024    ['conv2_block2_3_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv2_block2_add (Ad  (None, 56, 56  0       ['conv2_block1_out[0]  Y          
#  d)                   , 256)                  [0]',                             
#                                                'conv2_block2_3_bn[0             
#                                               ][0]']                            
#  conv2_block2_out (Ac  (None, 56, 56  0       ['conv2_block2_add[0]  Y          
#  tivation)            , 256)                  [0]']                             
#  conv2_block3_1_conv   (None, 56, 56  16448   ['conv2_block2_out[0]  Y          
#  (Conv2D)             , 64)                   [0]']                             
#  conv2_block3_1_bn (B  (None, 56, 56  256     ['conv2_block3_1_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block3_1_relu   (None, 56, 56  0       ['conv2_block3_1_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block3_2_conv   (None, 56, 56  36928   ['conv2_block3_1_relu  Y          
#  (Conv2D)             , 64)                   [0][0]']                          
#  conv2_block3_2_bn (B  (None, 56, 56  256     ['conv2_block3_2_conv  Y          
#  atchNormalization)   , 64)                   [0][0]']                          
#  conv2_block3_2_relu   (None, 56, 56  0       ['conv2_block3_2_bn[0  Y          
#  (Activation)         , 64)                   ][0]']                            
#  conv2_block3_3_conv   (None, 56, 56  16640   ['conv2_block3_2_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv2_block3_3_bn (B  (None, 56, 56  1024    ['conv2_block3_3_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv2_block3_add (Ad  (None, 56, 56  0       ['conv2_block2_out[0]  Y          
#  d)                   , 256)                  [0]',                             
#                                                'conv2_block3_3_bn[0             
#                                               ][0]']                            
#  conv2_block3_out (Ac  (None, 56, 56  0       ['conv2_block3_add[0]  Y          
#  tivation)            , 256)                  [0]']                             
#  conv3_block1_1_conv   (None, 28, 28  32896   ['conv2_block3_out[0]  Y          
#  (Conv2D)             , 128)                  [0]']                             
#  conv3_block1_1_bn (B  (None, 28, 28  512     ['conv3_block1_1_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block1_1_relu   (None, 28, 28  0       ['conv3_block1_1_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block1_2_conv   (None, 28, 28  147584  ['conv3_block1_1_relu  Y          
#  (Conv2D)             , 128)                  [0][0]']                          
#  conv3_block1_2_bn (B  (None, 28, 28  512     ['conv3_block1_2_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block1_2_relu   (None, 28, 28  0       ['conv3_block1_2_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block1_0_conv   (None, 28, 28  131584  ['conv2_block3_out[0]  Y          
#  (Conv2D)             , 512)                  [0]']                             
#  conv3_block1_3_conv   (None, 28, 28  66048   ['conv3_block1_2_relu  Y          
#  (Conv2D)             , 512)                  [0][0]']                          
#  conv3_block1_0_bn (B  (None, 28, 28  2048    ['conv3_block1_0_conv  Y          
#  atchNormalization)   , 512)                  [0][0]']                          
#  conv3_block1_3_bn (B  (None, 28, 28  2048    ['conv3_block1_3_conv  Y          
#  atchNormalization)   , 512)                  [0][0]']                          
#  conv3_block1_add (Ad  (None, 28, 28  0       ['conv3_block1_0_bn[0  Y          
#  d)                   , 512)                  ][0]',                            
#                                                'conv3_block1_3_bn[0             
#                                               ][0]']                            
#  conv3_block1_out (Ac  (None, 28, 28  0       ['conv3_block1_add[0]  Y          
#  tivation)            , 512)                  [0]']                             
#  conv3_block2_1_conv   (None, 28, 28  65664   ['conv3_block1_out[0]  Y          
#  (Conv2D)             , 128)                  [0]']                             
#  conv3_block2_1_bn (B  (None, 28, 28  512     ['conv3_block2_1_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block2_1_relu   (None, 28, 28  0       ['conv3_block2_1_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block2_2_conv   (None, 28, 28  147584  ['conv3_block2_1_relu  Y          
#  (Conv2D)             , 128)                  [0][0]']                          
#  conv3_block2_2_bn (B  (None, 28, 28  512     ['conv3_block2_2_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block2_2_relu   (None, 28, 28  0       ['conv3_block2_2_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block2_3_conv   (None, 28, 28  66048   ['conv3_block2_2_relu  Y          
#  (Conv2D)             , 512)                  [0][0]']                          
#  conv3_block2_3_bn (B  (None, 28, 28  2048    ['conv3_block2_3_conv  Y          
#  atchNormalization)   , 512)                  [0][0]']                          
#  conv3_block2_add (Ad  (None, 28, 28  0       ['conv3_block1_out[0]  Y          
#  d)                   , 512)                  [0]',                             
#                                                'conv3_block2_3_bn[0             
#                                               ][0]']                            
#  conv3_block2_out (Ac  (None, 28, 28  0       ['conv3_block2_add[0]  Y          
#  tivation)            , 512)                  [0]']                             
#  conv3_block3_1_conv   (None, 28, 28  65664   ['conv3_block2_out[0]  Y          
#  (Conv2D)             , 128)                  [0]']                             
#  conv3_block3_1_bn (B  (None, 28, 28  512     ['conv3_block3_1_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block3_1_relu   (None, 28, 28  0       ['conv3_block3_1_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block3_2_conv   (None, 28, 28  147584  ['conv3_block3_1_relu  Y          
#  (Conv2D)             , 128)                  [0][0]']                          
#  conv3_block3_2_bn (B  (None, 28, 28  512     ['conv3_block3_2_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block3_2_relu   (None, 28, 28  0       ['conv3_block3_2_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block3_3_conv   (None, 28, 28  66048   ['conv3_block3_2_relu  Y          
#  (Conv2D)             , 512)                  [0][0]']                          
#  conv3_block3_3_bn (B  (None, 28, 28  2048    ['conv3_block3_3_conv  Y          
#  atchNormalization)   , 512)                  [0][0]']                          
#  conv3_block3_add (Ad  (None, 28, 28  0       ['conv3_block2_out[0]  Y          
#  d)                   , 512)                  [0]',                             
#                                                'conv3_block3_3_bn[0             
#                                               ][0]']                            
#  conv3_block3_out (Ac  (None, 28, 28  0       ['conv3_block3_add[0]  Y          
#  tivation)            , 512)                  [0]']                             
#  conv3_block4_1_conv   (None, 28, 28  65664   ['conv3_block3_out[0]  Y          
#  (Conv2D)             , 128)                  [0]']                             
#  conv3_block4_1_bn (B  (None, 28, 28  512     ['conv3_block4_1_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block4_1_relu   (None, 28, 28  0       ['conv3_block4_1_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block4_2_conv   (None, 28, 28  147584  ['conv3_block4_1_relu  Y          
#  (Conv2D)             , 128)                  [0][0]']                          
#  conv3_block4_2_bn (B  (None, 28, 28  512     ['conv3_block4_2_conv  Y          
#  atchNormalization)   , 128)                  [0][0]']                          
#  conv3_block4_2_relu   (None, 28, 28  0       ['conv3_block4_2_bn[0  Y          
#  (Activation)         , 128)                  ][0]']                            
#  conv3_block4_3_conv   (None, 28, 28  66048   ['conv3_block4_2_relu  Y          
#  (Conv2D)             , 512)                  [0][0]']                          
#  conv3_block4_3_bn (B  (None, 28, 28  2048    ['conv3_block4_3_conv  Y          
#  atchNormalization)   , 512)                  [0][0]']                          
#  conv3_block4_add (Ad  (None, 28, 28  0       ['conv3_block3_out[0]  Y          
#  d)                   , 512)                  [0]',                             
#                                                'conv3_block4_3_bn[0             
#                                               ][0]']                            
#  conv3_block4_out (Ac  (None, 28, 28  0       ['conv3_block4_add[0]  Y          
#  tivation)            , 512)                  [0]']                             
#  conv4_block1_1_conv   (None, 14, 14  131328  ['conv3_block4_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block1_1_bn (B  (None, 14, 14  1024    ['conv4_block1_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block1_1_relu   (None, 14, 14  0       ['conv4_block1_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block1_2_conv   (None, 14, 14  590080  ['conv4_block1_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block1_2_bn (B  (None, 14, 14  1024    ['conv4_block1_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block1_2_relu   (None, 14, 14  0       ['conv4_block1_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block1_0_conv   (None, 14, 14  525312  ['conv3_block4_out[0]  Y          
#  (Conv2D)             , 1024)                 [0]']                             
#  conv4_block1_3_conv   (None, 14, 14  263168  ['conv4_block1_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block1_0_bn (B  (None, 14, 14  4096    ['conv4_block1_0_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block1_3_bn (B  (None, 14, 14  4096    ['conv4_block1_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block1_add (Ad  (None, 14, 14  0       ['conv4_block1_0_bn[0  Y          
#  d)                   , 1024)                 ][0]',                            
#                                                'conv4_block1_3_bn[0             
#                                               ][0]']                            
#  conv4_block1_out (Ac  (None, 14, 14  0       ['conv4_block1_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv4_block2_1_conv   (None, 14, 14  262400  ['conv4_block1_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block2_1_bn (B  (None, 14, 14  1024    ['conv4_block2_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block2_1_relu   (None, 14, 14  0       ['conv4_block2_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block2_2_conv   (None, 14, 14  590080  ['conv4_block2_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block2_2_bn (B  (None, 14, 14  1024    ['conv4_block2_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block2_2_relu   (None, 14, 14  0       ['conv4_block2_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block2_3_conv   (None, 14, 14  263168  ['conv4_block2_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block2_3_bn (B  (None, 14, 14  4096    ['conv4_block2_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block2_add (Ad  (None, 14, 14  0       ['conv4_block1_out[0]  Y          
#  d)                   , 1024)                 [0]',                             
#                                                'conv4_block2_3_bn[0             
#                                               ][0]']                            
#  conv4_block2_out (Ac  (None, 14, 14  0       ['conv4_block2_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv4_block3_1_conv   (None, 14, 14  262400  ['conv4_block2_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block3_1_bn (B  (None, 14, 14  1024    ['conv4_block3_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block3_1_relu   (None, 14, 14  0       ['conv4_block3_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block3_2_conv   (None, 14, 14  590080  ['conv4_block3_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block3_2_bn (B  (None, 14, 14  1024    ['conv4_block3_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block3_2_relu   (None, 14, 14  0       ['conv4_block3_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block3_3_conv   (None, 14, 14  263168  ['conv4_block3_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block3_3_bn (B  (None, 14, 14  4096    ['conv4_block3_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block3_add (Ad  (None, 14, 14  0       ['conv4_block2_out[0]  Y          
#  d)                   , 1024)                 [0]',                             
#                                                'conv4_block3_3_bn[0             
#                                               ][0]']                            
#  conv4_block3_out (Ac  (None, 14, 14  0       ['conv4_block3_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv4_block4_1_conv   (None, 14, 14  262400  ['conv4_block3_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block4_1_bn (B  (None, 14, 14  1024    ['conv4_block4_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block4_1_relu   (None, 14, 14  0       ['conv4_block4_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block4_2_conv   (None, 14, 14  590080  ['conv4_block4_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block4_2_bn (B  (None, 14, 14  1024    ['conv4_block4_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block4_2_relu   (None, 14, 14  0       ['conv4_block4_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block4_3_conv   (None, 14, 14  263168  ['conv4_block4_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block4_3_bn (B  (None, 14, 14  4096    ['conv4_block4_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block4_add (Ad  (None, 14, 14  0       ['conv4_block3_out[0]  Y          
#  d)                   , 1024)                 [0]',                             
#                                                'conv4_block4_3_bn[0             
#                                               ][0]']                            
#  conv4_block4_out (Ac  (None, 14, 14  0       ['conv4_block4_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv4_block5_1_conv   (None, 14, 14  262400  ['conv4_block4_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block5_1_bn (B  (None, 14, 14  1024    ['conv4_block5_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block5_1_relu   (None, 14, 14  0       ['conv4_block5_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block5_2_conv   (None, 14, 14  590080  ['conv4_block5_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block5_2_bn (B  (None, 14, 14  1024    ['conv4_block5_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block5_2_relu   (None, 14, 14  0       ['conv4_block5_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block5_3_conv   (None, 14, 14  263168  ['conv4_block5_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block5_3_bn (B  (None, 14, 14  4096    ['conv4_block5_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block5_add (Ad  (None, 14, 14  0       ['conv4_block4_out[0]  Y          
#  d)                   , 1024)                 [0]',                             
#                                                'conv4_block5_3_bn[0             
#                                               ][0]']                            
#  conv4_block5_out (Ac  (None, 14, 14  0       ['conv4_block5_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv4_block6_1_conv   (None, 14, 14  262400  ['conv4_block5_out[0]  Y          
#  (Conv2D)             , 256)                  [0]']                             
#  conv4_block6_1_bn (B  (None, 14, 14  1024    ['conv4_block6_1_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block6_1_relu   (None, 14, 14  0       ['conv4_block6_1_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block6_2_conv   (None, 14, 14  590080  ['conv4_block6_1_relu  Y          
#  (Conv2D)             , 256)                  [0][0]']                          
#  conv4_block6_2_bn (B  (None, 14, 14  1024    ['conv4_block6_2_conv  Y          
#  atchNormalization)   , 256)                  [0][0]']                          
#  conv4_block6_2_relu   (None, 14, 14  0       ['conv4_block6_2_bn[0  Y          
#  (Activation)         , 256)                  ][0]']                            
#  conv4_block6_3_conv   (None, 14, 14  263168  ['conv4_block6_2_relu  Y          
#  (Conv2D)             , 1024)                 [0][0]']                          
#  conv4_block6_3_bn (B  (None, 14, 14  4096    ['conv4_block6_3_conv  Y          
#  atchNormalization)   , 1024)                 [0][0]']                          
#  conv4_block6_add (Ad  (None, 14, 14  0       ['conv4_block5_out[0]  Y          
#  d)                   , 1024)                 [0]',                             
#                                                'conv4_block6_3_bn[0             
#                                               ][0]']                            
#  conv4_block6_out (Ac  (None, 14, 14  0       ['conv4_block6_add[0]  Y          
#  tivation)            , 1024)                 [0]']                             
#  conv5_block1_1_conv   (None, 7, 7,   524800  ['conv4_block6_out[0]  Y          
#  (Conv2D)             512)                    [0]']                             
#  conv5_block1_1_bn (B  (None, 7, 7,   2048    ['conv5_block1_1_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block1_1_relu   (None, 7, 7,   0       ['conv5_block1_1_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block1_2_conv   (None, 7, 7,   2359808  ['conv5_block1_1_relu  Y         
#  (Conv2D)             512)                    [0][0]']                          
#  conv5_block1_2_bn (B  (None, 7, 7,   2048    ['conv5_block1_2_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block1_2_relu   (None, 7, 7,   0       ['conv5_block1_2_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block1_0_conv   (None, 7, 7,   2099200  ['conv4_block6_out[0]  Y         
#  (Conv2D)             2048)                   [0]']                             
#  conv5_block1_3_conv   (None, 7, 7,   1050624  ['conv5_block1_2_relu  Y         
#  (Conv2D)             2048)                   [0][0]']                          
#  conv5_block1_0_bn (B  (None, 7, 7,   8192    ['conv5_block1_0_conv  Y          
#  atchNormalization)   2048)                   [0][0]']                          
#  conv5_block1_3_bn (B  (None, 7, 7,   8192    ['conv5_block1_3_conv  Y          
#  atchNormalization)   2048)                   [0][0]']                          
#  conv5_block1_add (Ad  (None, 7, 7,   0       ['conv5_block1_0_bn[0  Y          
#  d)                   2048)                   ][0]',                            
#                                                'conv5_block1_3_bn[0             
#                                               ][0]']                            
#  conv5_block1_out (Ac  (None, 7, 7,   0       ['conv5_block1_add[0]  Y          
#  tivation)            2048)                   [0]']                             
#  conv5_block2_1_conv   (None, 7, 7,   1049088  ['conv5_block1_out[0]  Y         
#  (Conv2D)             512)                    [0]']                             
#  conv5_block2_1_bn (B  (None, 7, 7,   2048    ['conv5_block2_1_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block2_1_relu   (None, 7, 7,   0       ['conv5_block2_1_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block2_2_conv   (None, 7, 7,   2359808  ['conv5_block2_1_relu  Y         
#  (Conv2D)             512)                    [0][0]']                          
#  conv5_block2_2_bn (B  (None, 7, 7,   2048    ['conv5_block2_2_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block2_2_relu   (None, 7, 7,   0       ['conv5_block2_2_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block2_3_conv   (None, 7, 7,   1050624  ['conv5_block2_2_relu  Y         
#  (Conv2D)             2048)                   [0][0]']                          
#  conv5_block2_3_bn (B  (None, 7, 7,   8192    ['conv5_block2_3_conv  Y          
#  atchNormalization)   2048)                   [0][0]']                          
#  conv5_block2_add (Ad  (None, 7, 7,   0       ['conv5_block1_out[0]  Y          
#  d)                   2048)                   [0]',                             
#                                                'conv5_block2_3_bn[0             
#                                               ][0]']                            
#  conv5_block2_out (Ac  (None, 7, 7,   0       ['conv5_block2_add[0]  Y          
#  tivation)            2048)                   [0]']                             
#  conv5_block3_1_conv   (None, 7, 7,   1049088  ['conv5_block2_out[0]  Y         
#  (Conv2D)             512)                    [0]']                             
#  conv5_block3_1_bn (B  (None, 7, 7,   2048    ['conv5_block3_1_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block3_1_relu   (None, 7, 7,   0       ['conv5_block3_1_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block3_2_conv   (None, 7, 7,   2359808  ['conv5_block3_1_relu  Y         
#  (Conv2D)             512)                    [0][0]']                          
#  conv5_block3_2_bn (B  (None, 7, 7,   2048    ['conv5_block3_2_conv  Y          
#  atchNormalization)   512)                    [0][0]']                          
#  conv5_block3_2_relu   (None, 7, 7,   0       ['conv5_block3_2_bn[0  Y          
#  (Activation)         512)                    ][0]']                            
#  conv5_block3_3_conv   (None, 7, 7,   1050624  ['conv5_block3_2_relu  Y         
#  (Conv2D)             2048)                   [0][0]']                          
#  conv5_block3_3_bn (B  (None, 7, 7,   8192    ['conv5_block3_3_conv  Y          
#  atchNormalization)   2048)                   [0][0]']                          
#  conv5_block3_add (Ad  (None, 7, 7,   0       ['conv5_block2_out[0]  Y          
#  d)                   2048)                   [0]',                             
#                                                'conv5_block3_3_bn[0             
#                                               ][0]']                            
#  conv5_block3_out (Ac  (None, 7, 7,   0       ['conv5_block3_add[0]  Y          
#  tivation)            2048)                   [0]']                             
#  avg_pool (GlobalAver  (None, 2048)  0        ['conv5_block3_out[0]  Y          
#  agePooling2D)                                [0]']                             
#  predictions (Dense)  (None, 1000)   2049000  ['avg_pool[0][0]']     Y          
# ================================================================================
# Total params: 25,636,712
# Trainable params: 25,583,592
# Non-trainable params: 53,120
# ________________________________________________________________________________

# Predicting the classes of the new images.
pred = model %>% predict (x) %>%
imagenet_decode_predictions(top = 5)

# Taking a look at the top 5 predicted classes.
names(pred) = image_names
print(pred)

# $img1.jpg
#   class_name class_description       score
# 1  n02487347           macaque 0.847332537
# 2  n02480495         orangutan 0.087277815
# 3  n02484975            guenon 0.029793324
# 4  n02493509              titi 0.011118809
# 5  n02493793     spider_monkey 0.007158784

# $img2.jpg
#   class_name class_description        score
# 1  n02504458  African_elephant 5.154622e-01
# 2  n01871265            tusker 4.715194e-01
# 3  n02504013   Indian_elephant 1.283774e-02
# 4  n02397096           warthog 2.121060e-05
# 5  n02391049             zebra 1.508214e-05

# $img3.jpg
#   class_name   class_description        score
# 1  n02132136          brown_bear 9.996605e-01
# 2  n02133161 American_black_bear 3.107583e-04
# 3  n01883070              wombat 7.252698e-06
# 4  n02117135               hyena 2.591740e-06
# 5  n02134418          sloth_bear 2.314227e-06

# $img4.jpg
#   class_name class_description        score
# 1  n02510455       giant_panda 9.997714e-01
# 2  n02509815      lesser_panda 1.395200e-04
# 3  n02447366            badger 4.193358e-05
# 4  n02134084          ice_bear 2.198698e-05
# 5  n02132136        brown_bear 7.121393e-06

# $img5.jpg
#   class_name  class_description        score
# 1  n01843383             toucan 9.986794e-01
# 2  n01829413           hornbill 1.290908e-03
# 3  n02056570       king_penguin 7.787835e-06
# 4  n02017213 European_gallinule 3.430432e-06
# 5  n01806567              quail 2.848779e-06

# $img6.jpg
#   class_name   class_description       score
# 1  n02099712  Labrador_retriever 0.722283900
# 2  n02099601    golden_retriever 0.190184966
# 3  n02087394 Rhodesian_ridgeback 0.037294745
# 4  n02088466          bloodhound 0.009720366
# 5  n02108551     Tibetan_mastiff 0.008164544

# $img7.jpg
#   class_name class_description      score
# 1  n02130308           cheetah 0.43274033
# 2  n02423022           gazelle 0.17040773
# 3  n01798484   prairie_chicken 0.10398047
# 4  n02018795           bustard 0.04691753
# 5  n02013706           limpkin 0.04554260

# $img8.jpg
#   class_name class_description        score
# 1  n02391049             zebra 9.980556e-01
# 2  n02423022           gazelle 6.932941e-04
# 3  n02422699            impala 4.003343e-04
# 4  n01518878           ostrich 1.408840e-04
# 5  n02422106        hartebeest 7.173959e-05

# $img9.jpg
#   class_name class_description        score
# 1  n02129604             tiger 9.206469e-01
# 2  n02123159         tiger_cat 7.859027e-02
# 3  n02127052              lynx 3.874848e-04
# 4  n02128925            jaguar 1.233026e-04
# 5  n02128385           leopard 4.457178e-05

# $img10.jpg
#   class_name class_description      score
# 1  n02397096           warthog 0.59179085
# 2  n02410509             bison 0.17515907
# 3  n01871265            tusker 0.08330985
# 4  n02504458  African_elephant 0.06870038
# 5  n02408429     water_buffalo 0.03099908

# $img11.jpg
#   class_name class_description        score
# 1  n02114367       timber_wolf 0.8871910572
# 2  n02114712          red_wolf 0.1019534841
# 3  n02114855            coyote 0.0094019976
# 4  n02114548        white_wolf 0.0008022708
# 5  n02115641             dingo 0.0004270469



