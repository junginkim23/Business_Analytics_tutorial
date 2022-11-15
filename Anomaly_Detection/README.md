# Anomaly Detection

1. Density-based Anomaly Detection
2. Distance-based Anomaly Detection
3. **Model-based Anomaly Detection** (●)

---

This time, we are going to proceed with a tutorial on anomaly detection using autoencoder.

Before diving into the tutorial, what is an autoencoder?

- An autoencoder is an artificial neural network that compresses the input data as much as possible when an input is received and then restores the compressed data back to the original input form. 
- The part that compresses the data is called an encoder, and the part that restores the data is called a decoder. The meaningful data z extracted during the compression process is called a latent vector.

<p align='center'><img src="./img/autoencoder.jpg" width='1000' height='200'></p>

---

**python tutorial**

