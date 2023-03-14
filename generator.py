import numpy as np
import cv2
from keras.models import load_model
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description ='Generate Images') 
    parser.add_argument('-generator', action = 'store', dest = 'gen', required = True, help = 'Generator model path')
    parser.add_argument('-category', action = 'store', dest = 'cat', required = True, help = 'Category option')
    parser.add_argument('-count', action = 'store', dest = 'count', required = True, help = 'How many images to generate')
    parser.add_argument('-output', action = 'store', dest = 'output', required = True, help = 'Output path')
    	  
    args = parser.parse_args()


    # Load the trained DCGAN model
    gen_model = load_model(args.gen)
    
    
    category = args.cat
    n_images = int(args.count)
    path = args.output
    
    # Create the destination folder if it doesn't already exist
    if not os.path.exists(path):
        print(f"Creating local directory {path}")
        os.makedirs(path)
    
        # Generate images
    latent_dim = 100 # Same as in the training script
    latent_input = np.random.randn(latent_dim * n_images)
    latent_input = latent_input.reshape(n_images, latent_dim)
    generated_images = gen_model.predict(latent_input)
    
        # Rescale generated images back to original scale
    generated_images = (generated_images + 1) / 2
    
        # Save generated images
    for i in range(n_images):
        image = cv2.cvtColor(generated_images[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path}/{category}_{i}.jpg", image * 255)

