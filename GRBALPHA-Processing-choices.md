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