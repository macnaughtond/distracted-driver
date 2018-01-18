# distracted-driver
Machine learning and deep learning models for classifying driving as good or 1 of 9 distraction categories

This model achieved a cross-entropy loss of [0.493,0.54] (31%,69%) when graded by the Kaggle.com website which places the result in the top quartile of final results on the public leader board (somewhere between [21.9%,23.9%]). This good result was achieved using only resources/methods that were available in 2015, including:
Transfer learning from Vgg16 convolutional layers
Lightweight hidden dense layers (128 filters + 128 filters) with batch normalisation to improve convergence spd
Using data augmentation (no dropout) and high quality pseudo labeling data to prevent over fitting
Being selective with training data. Removal of training data which appeared to be incorrectly classified.
Essentially, I used my machine learning knowledge of the curse of dimensionality to avoid overfitting. I created a 'model top' that was approximately sized for the amount of training data available. However, I haven't optimised the architecture of the top hat.
To improve the result, I would first proceed with further quick experiments including:
Increase the number of filters in the first dense layer
Combining Vgg16 (or Vgg19) convolutional layers with Resnet or Inception blocks.
Try adding more high quality pseudo labeled data (>0.99 probability cases only), though..
Other than that, I would expect that some time consuming efforts would be required to get into the top 10% of results. This is because the distracted driver training dataset only provides around 50 subjects (disregard the total number of training images for now). Skin tone, hair colour/style and clothing colour/style are hugely variable amongst the test subjects, whilst the variation between posture-detection categories (within-subject) is quite small. I could not visually distinguish or agree with the classification boundary between many of the radio and grooming distinctions; radio vs reaching behind vs talking to passenger; good driving looking right vs talking to passenger; good driving looking at rear view mirror vs makeup.
To improve the performance on this relatively small training data set, the next logical step is direct the focus of the model to key features: to incorporate hand, steering wheel, face, phone, make-up mirror/gaze direction bounding boxes (or segmentation) into a multi-label neural network model to improve performance. For example, a model bounding boxes for hands could be trained on an annotated hand dataset (e.g. http://www.robots.ox.ac.uk/~vgg/data/hands/) then used to predict the bounding boxes for hands in the images of the distracted-driver data set.
Similarly, the bounding boxes for steering wheel, face, phone, make-up mirror and gaze direction (can just be a two points forming a vector from the estimated centre of the eyeball to the centre of the pupil). Using the functional model API of Keras, the outputs of the bounding boxes can be connected with regression activation function to the second last layer of the model for the classification output.
I am not proceeding to improve my distracted driver classifier at this time as I wish to work on some other types of data sets to demonstrate my skills there, including:
Structured data
Time series data
GIS data (including time dependent factors)
However, in this series of notebooks I have demonstrated that I can get a top quartile results without employing time consuming machine learning specialist models.
n.b. Posture detection has commerically valuable applications - I may come back to this modeling work at a later stage to demonstrate top class modelling ability.
