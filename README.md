### Video understanding
Use Llama 4 models to intepret scenes from a movie and perform summarization 
## Implementation
Image frames are split from videos and sent to the Llama 4 Scout model to be intepreted when a scene change is detected.

A scene change is detected when the histogram between frames has a difference beyond a threshold. The key is the cv2.HISTCMP_BHATTACHARYYA metric used when doing compareHist()

The inidividual frames descriptions are contatenated and sent to the Llama 3.3 70b model where it performs a summarization over all the frame descriptions

## Future work

Experiment with different thresholds
Try batching frames together with one call
Try performing intermediate summarizations on frame descriptions, especially if you have a lot of frames
Try different models for frame descriptions and summarization
Try different platforms ( Groq, Lambda, etc... ) for speed and throughput
