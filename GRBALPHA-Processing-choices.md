If no processing is present the FSPL corrected signal power looks like this
![alt text](screenshots/image-33.png)
Removing the FSPL
![alt text](screenshots/image-32.png)


Simple thresholding analysis
from
![alt text](screenshots/image-35.png)
to
![alt text](screenshots/image-34.png)

Done for all using the threshold -138dBm
![alt text](screenshots/image-36.png)

Only issue is that the threshold might not work for all...
![alt text](screenshots/image-37.png)

Therefore it could make sense to play around with histogram. Just delete all the points under the one with the most points, or do some sick ass machine learning
![alt text](screenshots/image-38.png)
![alt text](screenshots/image-39.png)
![alt text](screenshots/image-40.png)
It works for most, but not for 
![alt text](screenshots/image-41.png)
as the signal is just foookin transmitting at all times
![alt text](screenshots/image-42.png)
But that is very rare and it might still work

Apparently it is a bit messy... We fix by making histogram

I still think the histogram should be a great possibility. Another possibility is taken a mean of all the histograms, but it is not that cool

I found out it was possible to change the variance of the clusters. It seem some improvements

![alt text](screenshots/image-43.png) 
this is too much, but it is still rather interesting
![alt text](screenshots/image-44.png)
It also misses this one.


Now we cooking
![alt text](screenshots/image-45.png)
It makes these histograms for all, and for some it is really good, others are more wack. may
![alt text](screenshots/image-46.png)


I still think the simple linear threshold with a -138dBm might still be the best, but I am cooking

This is a bit stupider way, one would think
![alt text](screenshots/image-47.png)

I think there is something like LDA where it tries to minimise the variance between each group while maximising the spread between two groups. That is exactly what i want, and the following is just a bad version
![alt text](screenshots/image-48.png)
But it is still cooking territory. The better we remove the noise, the better the model will be i think. I am also allowed not to train on dataset that brings trouble

I did it with a threshold on a moving average, and it is working really nice.

After removing the data points where it is not transmitting, i tried to see the relation of each of the features. Most of them are completely flat, which means that there is nothing to gain from making linear regression for example.

![alt text](screenshots/image-49.png)
![alt text](screenshots/image-50.png)
![alt text](screenshots/image-51.png)


There might be something to see from the FSPL, where accoring to the trend, there is a 2 db gain in signal strength not explained by the FSPL when the satellite is closest. If there was something like atmosphere attenuation, this is something we could see, where the signal is better than just explained by the shorter distance. 
There is a quite clear trend that pointing error equals a lower signal. Here the points where the pointing error is below 0 should probably just die as it can change the bias.

The Elevation and FSPL looks equal, as it is equal if the orbit is circular. It is.

Where the noise power os biggest the signal is also biggest, but that is maybe just shit.

The pointing error comes at higher elevation as the satellite is closer, but it is not due to a direct overhead pass. Instead it is due to the azimuth being slower
![alt text](screenshots/image-52.png)
![alt text](screenshots/image-57.png)
![alt text](screenshots/image-53.png)
The elevation is only ever 2 degrees off.

The straight lines taking 40 seconds to fix seems like the azimuth goes from 0 to 360 the wrong way, and afterwards follow like it is no problem.
It always happened at the top point of the specific pass. It looks like a coding error and thereby something that could be fixed.

There are also mistakes happening due to the very first

It looks like the waterfall plots change very quickly when the station is moving
![alt text](screenshots/image-54.png)
![alt text](screenshots/image-55.png)
![alt text](screenshots/image-56.png)


The set azimuth is interpolated by a simple method, but the control system definitely choosing the wrong way around the azimuth 0-360 change. It 
![alt text](screenshots/image-58.png)
It even shows how the control system is easily able to catch up to the quickest section of the pass. Correction, when it is directly overhead, it is not possible to keep up
![alt text](screenshots/image-59.png)

However,
