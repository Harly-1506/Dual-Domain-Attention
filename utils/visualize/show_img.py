import matplotlib.pyplot as plt 
import numpy as np



EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

def show_image_dataset(train_ds):
    samples, labels = iter(train_ds).next()

    fig = plt.figure(figsize=(16,24))
    for i in range(24):
        a = fig.add_subplot(4,6,i+1)
        a.set_title(EMOTION_DICT[labels[i].item()])
        a.axis('off')
        a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)