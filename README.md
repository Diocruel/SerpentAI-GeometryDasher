# Playing GeometryDash with the SerpentAI framework.
Final assignment for CS4180: Deep Learning at the TU Delft

This project uses the SerpentAI framework to record frames from the Steam game GeometryDash and predicts the next action (jump/no-jump). 
A separate Audio network is also trained to use raw audio to predict the same. 
The two resulting networks are combined in the combineAgent to see if their combination improves prediction.

This project uses a special version of PyAudio that lets you record directly from speakers,
as well as several other python libraries. For setup instruction see [the wiki](../../wiki/Setup-instructions).