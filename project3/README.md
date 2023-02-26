
# Individual Project 3:
# Image Generation with GAN

#### Due Date
* Tuesday Apr 4, 2023 (23:59)

#### Total Points 
* 100 (One Hundred)

## Goal
In this assignment you will be asked to implement a Generative Adversarial Networks (GAN) with [MNIST data set](http://yann.lecun.com/exdb/mnist/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/tutorials/). 

<img src="https://github.com/yanhuata/DS504CS586-S20/blob/master/project3/pic/goal.png" width="80%">


## Project Guidelines

#### Data set

MNIST is a dataset composed of handwrite numbers and their labels. Each MNIST image is a 28\*28 grey-scale image, which is labeled with an integer value from 0 and 9, corresponding to the actual value in the image. MNIST is provided in Pytorch as 28\*28 matrices containing numbers ranging from 0 to 255. There are 60000 images and labels in the training data set and 10000 images and labels in the test data set. Since this project is an unsupervised learning project, you can only use the 60000 images for your training. 

#### Installing Software and Dependencies 

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Create virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Install packages (e.g. pip install torch)

#### Building and Compiling Generator and Discriminator

In Pytorch, you can try different layers, such as “Conv2D”, different activation functions, such as “tanh”, “leakyRelu”. You can apply different optimizers, such as stochastic gradient descent or Adam, and different loss functions. The following is the sample code of how to build the model.


```python
# Create a Generator class.
class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Generator.
netG = Generator(*args)

# Create a Discriminator class.
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Discriminator.
netD = Discriminator(*args)

# Setup Generator optimizer.
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Setup Discriminator optimizer.
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Define loss function. 
criterion = torch.nn.BCELoss()
```

#### Training GAN

You have the option of changing how many epochs to train your model for and how large your batch size is. The following is the sample code of how to train GAN. You can add self-defined parameters such as #epoch, learning rate scheduler to the train function.



```python
# Training
def train():
    for _ in range(batchCount):  
	
        # Create a batch by drawing random index numbers from the training set
       
        # Create noise vectors for the generator
        
        # Generate the images from the noise

        # Create labels

        # Train discriminator on generated images

        # Train generator

```

#### Saving Generator

Please use the following code to save the model and weights of your generator.



```python
# save model with Pytorch
torch.save(netG.state_dict(), 'PATH_TO_SAVED_GENERATOR')
torch.save(netD.state_dict(), 'PATH_TO_SAVED_DISCRIMINATOR')
```

#### Plotting

Please use the following code to plot the generated images. As for the loss plot of your generator and discriminator during the training, you can plot with your own style. 


```python
# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.random.normal(size=[num_gen, z_dim])
generated_images = netG(z)

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
plt.show()
```

## Deliverables

Please compress all the below files into a zipped file and submit the zip file (firstName_lastName_GAN.zip) to Canvas. 

#### PDF Report
* Set of Experiments Performed: Include a section describing the set of experiments that you performed, what structures you experimented with (i.e., number of layers, number of neurons in each layer), what hyperparameters you varied (e.g., number of epochs of training, batch size and any other parameter values, weight initialization schema, activation function), what kind of loss function you used and what kind of optimizer you used. 
* Special skills: Include the skills which can improve the generation quality. Here are some [tips](https://github.com/soumith/ganhacks) may help.   
* Visualization: Include 25 (5\*5) final generated images which formatted as the example in Goal and a loss plot of the generator and discriminator during your training. For generated images, you need to generated at least one image for each digit. 

#### Python code
* Include model creation, model training, plotting code.

#### Generator Model
* Turn in your best generator saved as “generator.json” and the weights of your generator saved as “generator.h5”.


## Grading

#### Report (70%)

* Set of experiments performed: 30 points
* Special skills: 20 points
* Visualization: 20 points

#### Code (20%) 

You can get full credits if the scripts can run successfully (i.e., TA will test your code with a small data set to see if images can be generated), otherwise you may loss some points based on your error. Similar to project 2, you should submit a evaluation.py file.

#### Model (10%)

You can get full credits if all the generated images can be recognized, otherwise you may loss some points. Also, the code you submitted should be able to generate all 10 different digits.

## Bonus (10 points)

Generate images from other data source.

* Data set

  Here are some images you may interest. Note other data sets are also allowed.
  
  [Face](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)
  
  [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
  
  [Anime](https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I)
  
* Package

  You are allowed to use any deep learning package, such as Tensorflow, Pytorch, etc.
  
* Deliverable

  * Code
  
  * Model
  
  * README file (How to  compile and load the model to generate images)
  
  * 25 generated images

## Tips of Using GPU on ACE Server

* Set up environment on ACE server 
    1. Connect to ACE server
    2. Open remote folder (your own root folder on the server) 
    3. Create a new terminal
    4. Load anaconda3, CUDA and cudnn using “module load” command
    5. Create new conda env using “conda create –n NAME”
    6. Activate new env using “source activate NAME”
    7. Install Pytorch using “conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch ”

* Submit job on Turing server
   ```shell
   #!/bin/bash
   #SBATCH -N 1
   #SBATCH -n 4
   #SBATCH --gres=gpu:1

   module load cuda92/toolkit
   module load cudnn
   python torch_test.py
   ```

