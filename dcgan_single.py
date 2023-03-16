# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from tensorflow.keras.models import Sequential # for assembling a Neural Network model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout # adding layers to the Neural Network model
# from tensorflow.keras.utils import plot_model # for plotting model diagram
from tensorflow.keras.optimizers import Adam # for model optimization 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data manipulation
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
# import sklearn
# print('sklearn: %s' % sklearn.__version__) # print version
# from sklearn.preprocessing import MinMaxScaler # for scaling inputs used in the generator and discriminator


# Visualization
# import cv2 # for ingesting images
# print('OpenCV: %s' % cv2.__version__) # print version
#import matplotlib 
#import matplotlib.pyplot as plt # or data visualizationa
#print('matplotlib: %s' % matplotlib.__version__) # print version
#import graphviz # for showing model diagram
#print('graphviz: %s' % graphviz.__version__) # print version


# Other utilities
import os
import argparse



def generator(latent_dim, image_dim):
    nod = int(image_dim/8)
    
    model = Sequential(name="Generator") # Model
    
    # Hidden Layer 1: Start with 8 x 8 image
    n_nodes = nod * nod * 128 # number of nodes in the first hidden layer
    model.add(Dense(n_nodes, input_dim=latent_dim, name='Generator-Hidden-Layer-1'))
    model.add(Reshape((nod, nod, 128), name='Generator-Hidden-Layer-Reshape-1'))
    
    # Hidden Layer 2: Upsample to 24 x 24
    model.add(Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-Hidden-Layer-2'))
    model.add(LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-2'))
                              
    # Hidden Layer 3: Upsample to 48 x 48
    model.add(Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-Hidden-Layer-3'))
    model.add(LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-3'))
    
    # Hidden Layer 4: Upsample to 96 x 96
    model.add(Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-Hidden-Layer-4'))
    model.add(LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-4'))
    
    # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have only 1 channel)
    model.add(Conv2D(filters=3, kernel_size=(5,5), activation='tanh', padding='same', name='Generator-Output-Layer'))
    return model

def discriminator(in_shape):
    model = Sequential(name="Discriminator") # Model
    
    # Hidden Layer 1
    model.add(Conv2D(filters=96, kernel_size=(4,4), strides=(2, 2), padding='same', input_shape=in_shape, name='Discriminator-Hidden-Layer-1'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))
    
    # Hidden Layer 2
    model.add(Conv2D(filters=192, kernel_size=(4,4), strides=(2, 2), padding='same', name='Discriminator-Hidden-Layer-2'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))
    
    # Hidden Layer 3
    model.add(Conv2D(filters=192, kernel_size=(4,4), strides=(2, 2), padding='same', name='Discriminator-Hidden-Layer-3'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))
    
    # Flatten and Output Layers
    model.add(Flatten(name='Discriminator-Flatten-Layer')) # Flatten the shape
    model.add(Dropout(0.3, name='Discriminator-Flatten-Layer-Dropout')) # Randomly drop some connections for better generalization
    model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer')) # Output Layer
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model



def def_gan(generator, discriminator):
    
    # We don't want to train the weights of discriminator at this stage. Hence, make it not trainable
    discriminator.trainable = False
    
    # Combine
    model = Sequential(name="DCGAN") # GAN Model
    model.add(generator) # Add Generator
    model.add(discriminator) # Add Disriminator
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model



def real_samples(generator):
    # Samples of real data
    X = generator.next()

    # Class labels
    y = np.ones((X.shape[0], 1))

    return X, y
    
    
def latent_vector(latent_dim, n):
    
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * n)
    
    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input
  
    
def fake_samples(generator, latent_dim, n):
    
    # Generate points in latent space
    latent_output = latent_vector(latent_dim, n)
    
    # Predict outputs (i.e., generate fake samples)
    X = generator.predict(latent_output)
    
    # Create class labels
    y = np.zeros((n, 1))
    return X, y    
    
def performance_summary(generator, discriminator, dataset, latent_dim, n=200):
    
    # Get samples of the real data
    x_real, y_real = real_samples(dataset)
    # Evaluate the descriminator on real data
    _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)
    
    # Get fake (generated) samples
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    # Evaluate the descriminator on fake (generated) data
    _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)
    
    # summarize discriminator performance
    print("*** Evaluation ***")
    print("Discriminator Accuracy on REAL images: ", real_accuracy)
    print("Discriminator Accuracy on FAKE (generated) images: ", fake_accuracy)
    

    
def train(g_model, d_model, gan_model, dataset_generator, latent_dim, n_epochs, n_batch, n_eval, out_path):
    
    # Our batch to train the discriminator will consist of half real images and half fake (generated) images
    half_batch = int(n_batch / 2)
    
    # We will manually enumare epochs 
    for i in range(n_epochs):
    
    # Discriminator training
        # Prep real samples
        x_real, y_real = real_samples(dataset_generator)
        # Prep fake (generated) samples
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)
        
        # Train the discriminator using real and fake samples
        X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        discriminator_loss, _ = d_model.train_on_batch(X, y)
    
    # Generator training
        # Get values from the latent space to be used as inputs for the generator
        x_gan = latent_vector(latent_dim, n_batch)
        # While we are generating fake samples, 
        # we want GAN generator model to create examples that resemble the real ones,
        # hence we want to pass labels corresponding to real samples, i.e. y=1, not 0.
        y_gan = np.ones((n_batch, 1))
        
        # Train the generator via a composite GAN model
        generator_loss = gan_model.train_on_batch(x_gan, y_gan)
        
        # Evaluate the model at every n_eval epochs
        if (i) % n_eval == 0:
            print("Epoch number: ", i)
            print("*** Training ***")
            print("Discriminator Loss ", discriminator_loss)
            print("Generator Loss: ", generator_loss)
            performance_summary(g_model, d_model, dataset_generator, latent_dim)
            
        if (i) % SAVE_EVERY_X_EPOCH == 0:
            print(f"Saved weights for iteration {i}")
            g_model.save(out_path+"/generator_" + str(i) + ".h5")
            d_model.save(out_path+"/discriminator_" + str(i) + ".h5")
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description ='Train a DCGAN image generator') 
    parser.add_argument('-dataset', action = 'store', dest = 'dataset', required = True, help = 'Path to dataset')
    parser.add_argument('-epochs', action = 'store', dest = 'epochs', type=int, required = True, help = 'Epochs to train')
    parser.add_argument('-batch', action = 'store', dest = 'batch', type=int, required = True, help = 'Batch size')
    parser.add_argument('-eval', action = 'store', dest = 'eval', type=int, default=100, required = False, help = 'Evaluate loss every X epochs')
    parser.add_argument('-save', action = 'store', dest = 'save', type=int, default=1000, required = False, help = 'Save models every X epochs')
    parser.add_argument('-lat-dim', action = 'store', dest = 'lat', type=int, default=100, required = False, help = 'Latent dimension scale (default to 100)')
    parser.add_argument('-img-dim', action = 'store', dest = 'dim', type=int, required = True, help = 'Input image width and height')
    parser.add_argument('-img-channels', action = 'store', dest = 'channels', type=int, default=3, required = False, help = 'Input image channels (1 for grayscale)')
    
    parser.add_argument('-output', action = 'store', dest = 'output', required = True, help = 'Model output path')
    	  
    args = parser.parse_args()
    
    N_EPOCH = args.epochs+1
    N_BATCH = args.batch
    N_EVAL = args.eval
    SAVE_EVERY_X_EPOCH = args.save
    
    IMAGE_SHAPE = (args.dim,args.dim,args.channels)
    IMAGE_DIM = IMAGE_SHAPE[0]
    
    print()
    print("Dataset path: ", args.dataset)
    print("Batch size: ", N_BATCH)
    print()
    
    OUT_PATH = args.output
    print(OUT_PATH)
    # Create the destination folder if it doesn't already exist
    if not os.path.exists(OUT_PATH):
        print("Creating local directory: ", OUT_PATH)
        os.makedirs(OUT_PATH)
    
    # # Create a list to store image paths
    # ImagePaths=[]
    # for image in list(os.listdir(ImgLocation)):
    #     ImagePaths=ImagePaths+[ImgLocation+"/"+image]
        
    # image_gen = ImageDataGenerator(rotation_range=45, #rotate images up to X degrees
    #                                brightness_range=[0.3, 1], #brigthness range change
    #                                horizontal_flip=True,
    #                                shear_range=0.2,
    #                                rescale=1./255)
    
    image_gen = ImageDataGenerator(rescale=1./255)
    
    COLOR_MODE = "rgb" if IMAGE_SHAPE[2] == 3 else "grayscale"
    
    train_data_gen = image_gen.flow_from_directory(
    directory=args.dataset,
    target_size=(IMAGE_DIM,IMAGE_DIM),
    batch_size=N_BATCH,
    class_mode=None,
    color_mode=COLOR_MODE
    )
            
    
    # # Load images and resize to 64 x 64 (DCGAN is 64x64, I did not use this)
    # data_lowres=[]
    # for img in ImagePaths:
    #     image = cv2.imread(img)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Not required, but to use matplolib to check is required
    #     #image_lowres = cv2.resize(image, (64, 64))
    #     data_lowres.append(image)
        
    # # Convert image data to numpy array and standardize values (divide by 255 since RGB values ranges from 0 to 255)
    # data_lowres = np.array(data_lowres, dtype="float") / 255.0
    
    # # Show data shape
    # print("Shape of data_lowres: ", data_lowres.shape)
    
    # # Scaler
    # scaler=MinMaxScaler(feature_range=(-1, 1))
    
    # # Select images that we want to use for model trainng
    # data=data_lowres.copy()
    # print("Original shape of the data: ", data.shape)
    
    # # Reshape array
    # data=data.reshape(-1, 1)
    # print("Reshaped data: ", data.shape)
    
    # # Fit the scaler
    # scaler.fit(data)
    
    # # Scale the array
    # data=scaler.transform(data)
    
    # # Reshape back to the original shape
    # data=data.reshape(data_lowres.shape[0], IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
    # print("Shape of the scaled array: ", data.shape)
    
    # Instantiate
    latent_dim=args.lat # Our latent space has 100 dimensions. We can change it to any number
    gen_model = generator(latent_dim, IMAGE_DIM)
    
    # # Show model summary and plot model diagram
    # gen_model.summary()
    # plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)
    
    # Instantiate
    dis_model = discriminator(IMAGE_SHAPE)
    
    # Show model summary and plot model diagram
    #dis_model.summary()
    #plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)
    
    # Instantiate
    gan_model = def_gan(gen_model, dis_model)
    
    # Show model summary and plot model diagram
    # gan_model.summary()
    # plot_model(gan_model, show_shapes=True, show_layer_names=True, dpi=400)
    
    train(gen_model, dis_model, gan_model, train_data_gen, latent_dim, N_EPOCH, N_BATCH, N_EVAL, OUT_PATH)
